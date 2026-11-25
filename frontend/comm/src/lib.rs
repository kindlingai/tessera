//! Opens the ranks' connections and hands them to the engine.

mod roce;

pub use roce::{port, ports};

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tessera_engine::comm::{Error, Fabric, Shm, Sockets, Transport, Verbs, VerbsLink};
use tessera_engine::p2p;
use tessera_engine::sums::{self, Sum};
use tessera_engine::whole;
use tessera_engine::{Grid, Settings};

/// How long a rank waits for the others.
const PATIENCE: Duration = Duration::from_secs(10);

/// Room for every token's logits, and then some.
pub const CAPACITY: usize = 64 << 20;

/// Binds `listen`, or this rank's address. Joins the ranks.
pub fn meet(rank: u32, addrs: &[SocketAddr], listen: Option<SocketAddr>) -> Result<Sockets, Error> {
    let listen = listen.unwrap_or(addrs[rank as usize]);
    let listener = TcpListener::bind(listen)?;
    eprintln!("rank {rank}: listening on {listen}");
    sockets(rank, listener, addrs)
}

/// One transport with two holders. They take turns.
struct Shared(Rc<RefCell<Box<dyn Transport>>>);

impl Transport for Shared {
    fn rank(&self) -> u32 {
        self.0.borrow().rank()
    }

    fn ranks(&self) -> u32 {
        self.0.borrow().ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        self.0.borrow_mut().exchange(mine)
    }
}

/// The sums' transport, then control's. `fabric` sums over RDMA.
pub fn join(
    mut met: Box<dyn Transport>,
    fabric: Option<&Fabric>,
) -> Result<[Box<dyn Transport>; 2], Error> {
    let Some(fabric) = fabric else {
        let met = Rc::new(RefCell::new(met));
        return Ok([Box::new(Shared(met.clone())), Box::new(Shared(met))]);
    };
    let (rank, ranks) = (met.rank(), met.ranks());
    let verbs = Verbs::connect(rank, ranks, fabric, CAPACITY, &mut *met)?;
    let lanes = fabric.lanes();
    eprintln!(
        "rank {rank}: sums over {lanes} RDMA lanes: {}",
        roce::describe(fabric)
    );
    Ok([Box::new(verbs), met])
}

/// How ranks sum activations across devices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sums {
    #[default]
    Host,
    Grid,
    P2p,
    P2pBf16,
}

/// Parses `--sums host|grid|p2p|p2p-bf16`.
pub fn sums_flag(value: &str) -> Option<Sums> {
    match value {
        "host" => Some(Sums::Host),
        "grid" => Some(Sums::Grid),
        "p2p" => Some(Sums::P2p),
        "p2p-bf16" => Some(Sums::P2pBf16),
        _ => None,
    }
}

/// Direct GPU-to-GPU sums over IPC and SM stores.
pub fn p2p(
    met: &mut dyn Transport,
    settings: &Settings,
) -> Result<Box<dyn Sum>, Box<dyn std::error::Error>> {
    p2p_with(met, settings, false)
}

/// Direct GPU-to-GPU sums over IPC with BF16 wire transport.
pub fn p2p_bf16(
    met: &mut dyn Transport,
    settings: &Settings,
) -> Result<Box<dyn Sum>, Box<dyn std::error::Error>> {
    p2p_with(met, settings, true)
}

fn p2p_with(
    met: &mut dyn Transport,
    settings: &Settings,
    bf16: bool,
) -> Result<Box<dyn Sum>, Box<dyn std::error::Error>> {
    let rank = met.rank();
    let device = settings.device.unwrap_or(rank as usize);
    tessera_engine::select(device);
    let sums = match bf16 {
        true => p2p::open_bf16(settings.gpu(), met)?,
        false => p2p::open(settings.gpu(), met)?,
    };
    let kind = if bf16 { "bf16" } else { "f32" };
    eprintln!("rank {rank}: sums over direct GPU P2P ({kind}, device {device})");
    Ok(sums)
}

/// A persistent grid that sums over `fabric`, met through `met`.
pub fn grid(
    met: &mut dyn Transport,
    fabric: &Fabric,
    settings: &Settings,
) -> Result<Box<dyn Sum>, Box<dyn std::error::Error>> {
    let link = link(met, fabric, settings, (sums::regions, sums::RELAY), "sums")?;
    Ok(sums::open(settings.gpu(), link)?)
}

/// A recipe's resident grid, its sums over `fabric`.
pub fn resident(
    met: &mut dyn Transport,
    fabric: &Fabric,
    settings: &Settings,
) -> Result<Box<dyn Grid>, Box<dyn std::error::Error>> {
    let link = link(
        met,
        fabric,
        settings,
        (whole::regions, whole::RELAY),
        "steps",
    )?;
    Ok(whole::open(settings.gpu(), link, 1)?)
}

/// The fabric link a persistent grid of `what` runs over.
fn link(
    met: &mut dyn Transport,
    fabric: &Fabric,
    settings: &Settings,
    (regions, relay): (fn(u32) -> Vec<usize>, usize),
    what: &str,
) -> Result<VerbsLink, Error> {
    let (rank, ranks) = (met.rank(), met.ranks());
    let pinning = tessera_engine::pinning(settings);
    let link = VerbsLink::connect(rank, ranks, fabric, &regions(ranks), relay, met, pinning)?;
    eprintln!(
        "rank {rank}: {what} in a persistent grid over {}",
        roce::describe(fabric)
    );
    Ok(link)
}

/// Joins rank `rank` of `addrs` on `listener`. Lower ranks call.
pub fn sockets(rank: u32, listener: TcpListener, addrs: &[SocketAddr]) -> Result<Sockets, Error> {
    sockets_within(rank, listener, addrs, PATIENCE, None)
}

/// Joins `rank` within `wait`, stopping when `abort` flips.
pub fn sockets_within(
    rank: u32,
    listener: TcpListener,
    addrs: &[SocketAddr],
    wait: Duration,
    abort: Option<&AtomicBool>,
) -> Result<Sockets, Error> {
    let deadline = Instant::now() + wait;
    let mut peers: Vec<Option<TcpStream>> = (0..addrs.len()).map(|_| None).collect();
    for (r, addr) in addrs.iter().enumerate().take(rank as usize) {
        let mut stream = call(*addr, deadline, abort)?;
        stream.set_write_timeout(Some(remaining(deadline, abort)?))?;
        stream.write_all(&rank.to_le_bytes())?;
        stream.set_write_timeout(None)?;
        peers[r] = Some(stream);
    }
    if rank as usize + 1 < addrs.len() {
        listener.set_nonblocking(true)?;
    }
    for _ in rank as usize + 1..addrs.len() {
        let mut stream = accept(&listener, deadline, abort)?;
        stream.set_read_timeout(Some(remaining(deadline, abort)?))?;
        let mut who = [0u8; 4];
        stream.read_exact(&mut who)?;
        stream.set_read_timeout(None)?;
        let who = u32::from_le_bytes(who) as usize;
        if who <= rank as usize || who >= addrs.len() || peers[who].is_some() {
            return Err(Error::Fit(format!("rank {who} called rank {rank}")));
        }
        peers[who] = Some(stream);
    }
    Sockets::over(rank, peers)
}

/// Whether a connect failure may clear once the peer binds.
fn transient(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::ConnectionRefused | ErrorKind::Interrupted
    )
}

fn remaining(deadline: Instant, abort: Option<&AtomicBool>) -> io::Result<Duration> {
    if abort.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(ErrorKind::ConnectionAborted.into());
    }
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(ErrorKind::TimedOut.into());
    }
    Ok(left)
}

/// Accepts one stream on nonblocking `listener` before `deadline`.
fn accept(
    listener: &TcpListener,
    deadline: Instant,
    abort: Option<&AtomicBool>,
) -> io::Result<TcpStream> {
    loop {
        let _ = remaining(deadline, abort)?;
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false)?;
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => {
                let left = remaining(deadline, abort)?;
                thread::sleep(Duration::from_millis(5).min(left));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Connects to `addr`, retrying while it is not up yet.
fn call(addr: SocketAddr, deadline: Instant, abort: Option<&AtomicBool>) -> io::Result<TcpStream> {
    loop {
        let left = remaining(deadline, abort)?;
        match TcpStream::connect_timeout(&addr, left) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(e) if transient(&e) => {
                let left = remaining(deadline, abort)?;
                thread::sleep(Duration::from_millis(20).min(left));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Rank 0 lays `path` out. The others wait for it.
pub fn shm(path: &Path, rank: u32, ranks: u32, capacity: usize) -> Result<Shm, Error> {
    let file = shared(path, rank, ranks, Shm::room(ranks, capacity))?;
    Shm::over(rank, ranks, file, capacity)
}

/// `len` zeroed bytes, then one mark byte per rank.
pub fn shared(path: &Path, rank: u32, ranks: u32, len: usize) -> Result<File, Error> {
    let room = (len + ranks as usize) as u64;
    let file = match rank {
        0 => lay_out(path, ranks, len, room)?,
        _ => open(path, rank, len, room)?,
    };
    Ok(file)
}

/// Rank 0 waits for the other ranks' marks, then unlinks.
fn lay_out(path: &Path, ranks: u32, len: usize, room: u64) -> Result<File, Error> {
    let mut fresh = path.as_os_str().to_owned();
    fresh.push(".new");
    let _ = std::fs::remove_file(&fresh);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&fresh)?;
    file.set_len(room)?;
    std::fs::rename(&fresh, path)?;
    let start = Instant::now();
    let mut marks = vec![0u8; ranks as usize];
    loop {
        file.read_exact_at(&mut marks, len as u64)?;
        let missing: Vec<usize> = (1..marks.len()).filter(|&r| marks[r] == 0).collect();
        if missing.is_empty() {
            break;
        }
        if start.elapsed() > PATIENCE {
            let what = format!("ranks {missing:?} did not open {}", path.display());
            return Err(Error::Fit(what));
        }
        thread::sleep(Duration::from_millis(1));
    }
    std::fs::remove_file(path)?;
    file.write_all_at(&[1], len as u64)?;
    Ok(file)
}

/// Marks this rank's byte. Waits for rank 0's mark.
fn open(path: &Path, rank: u32, len: usize, room: u64) -> Result<File, Error> {
    let start = Instant::now();
    let mut held: Option<File> = None;
    while start.elapsed() < PATIENCE {
        held = match held.take() {
            Some(file) => {
                let mut go = [0u8];
                file.read_exact_at(&mut go, len as u64)?;
                if go[0] == 1 {
                    return Ok(file);
                }
                // Rank 0 replaced an earlier run's file.
                let now = std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()));
                let mine = file.metadata()?;
                let stale = now.is_some_and(|now| now != (mine.dev(), mine.ino()));
                (!stale).then_some(file)
            }
            None => match OpenOptions::new().read(true).write(true).open(path) {
                Ok(file) if file.metadata()?.len() == room => {
                    file.write_all_at(&[1], (len + rank as usize) as u64)?;
                    Some(file)
                }
                _ => None,
            },
        };
        thread::sleep(Duration::from_millis(1));
    }
    let what = format!("rank 0 did not lay out {}", path.display());
    Err(Error::Fit(what))
}

#[cfg(test)]
mod lib_test;
#[cfg(test)]
mod roce_test;
