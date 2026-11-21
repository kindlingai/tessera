//! Opens the ranks' connections and hands them to the engine.

mod roce;

pub use roce::{port, ports};

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::rc::Rc;
use std::thread;
use std::time::{Duration, Instant};
use tessera_engine::comm::{Error, Fabric, Shm, Sockets, Transport, Verbs, VerbsLink};
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

/// `--sums grid` or `host`: whether sums run in a grid.
pub fn grid_flag(value: &str) -> Option<bool> {
    match value {
        "grid" => Some(true),
        "host" => Some(false),
        _ => None,
    }
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
    let mut peers: Vec<Option<TcpStream>> = (0..addrs.len()).map(|_| None).collect();
    for (r, addr) in addrs.iter().enumerate().take(rank as usize) {
        let mut stream = call(*addr)?;
        stream.write_all(&rank.to_le_bytes())?;
        peers[r] = Some(stream);
    }
    for _ in rank as usize + 1..addrs.len() {
        let (mut stream, _) = listener.accept()?;
        stream.set_nodelay(true)?;
        let mut who = [0u8; 4];
        stream.read_exact(&mut who)?;
        let who = u32::from_le_bytes(who) as usize;
        if who <= rank as usize || who >= addrs.len() || peers[who].is_some() {
            return Err(Error::Fit(format!("rank {who} called rank {rank}")));
        }
        peers[who] = Some(stream);
    }
    Sockets::over(rank, peers)
}

/// Connects to `addr`, retrying while it is not up yet.
fn call(addr: SocketAddr) -> io::Result<TcpStream> {
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(e) if start.elapsed() < PATIENCE => {
                let _ = e;
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Rank 0 lays `path` out. The others wait for it.
pub fn shm(path: &Path, rank: u32, ranks: u32, capacity: usize) -> Result<Shm, Error> {
    let file = shared(path, rank, Shm::room(ranks, capacity))?;
    Shm::over(rank, ranks, file, capacity)
}

/// A zeroed file of `len` bytes that rank 0 makes.
pub fn shared(path: &Path, rank: u32, len: usize) -> Result<File, Error> {
    let start = Instant::now();
    let file = loop {
        if rank == 0 {
            let _ = std::fs::remove_file(path);
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(path)?;
            file.set_len(len as u64)?;
            break file;
        }
        match OpenOptions::new().read(true).write(true).open(path) {
            Ok(file) if file.metadata()?.len() as usize == len => break file,
            Ok(_) | Err(_) if start.elapsed() < PATIENCE => thread::sleep(Duration::from_millis(5)),
            Ok(_) => {
                let what = format!("{} has the wrong size", path.display());
                return Err(Error::Fit(what));
            }
            Err(e) => return Err(e.into()),
        }
    };
    Ok(file)
}

#[cfg(test)]
mod lib_test;
#[cfg(test)]
mod roce_test;
