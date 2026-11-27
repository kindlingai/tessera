//! Dual-root RoCE link driven by a host proxy thread.

use crate::fabric::Fabric;
use crate::pace::{Pace, Spinning, Step};
use crate::pin::Pinning;
use crate::verbs::{Engine, Region};
use crate::{Error, Transport};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Send buffer bytes per parity.
pub const SEND_BYTES: usize = 32 << 20;
/// Receive slot bytes per peer per parity.
pub const SLOT_BYTES: usize = 16 << 20;
/// Maximum ranks supported by arx.
pub const MAX_RANKS: usize = 8;

/// Control word indexes in region 3.
const CTL_STOP: usize = 5;
const CTL_ERR: usize = 7;
const CTL_GPU_ERR: usize = 8;
/// Word offset of the 64-entry work ring in region 3.
const CTL_RING: usize = 16;
const RING_CAP: usize = 64;
const WORK_WORDS: usize = 7;

/// Scatter piece op posted by the GPU.
const MODE_SCATTER_PIECE: u64 = 3;

/// Flag value encoding matching arx.cu (`seq * 16 + piece + 1`).
const FLAG_STRIDE: u64 = 16;
const PIECES: u64 = 8;

/// Device addresses and topology exposed to GPU kernels.
#[derive(Clone, Copy, Debug)]
pub struct ArxMem {
    pub send: u64,
    pub recv: u64,
    pub flag: u64,
    pub ctl: u64,
    pub rank: u32,
    pub ranks: u32,
    pub lanes: u32,
    pub mesh: bool,
}

/// Pinned RDMA buffers with a dedicated proxy thread.
pub struct ArxLink {
    mem: ArxMem,
    ctl: *const AtomicU64,
    fault: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
}

// The proxy thread owns the regions until drop joins it.
unsafe impl Send for ArxLink {}

impl ArxLink {
    pub const SEND_BYTES: usize = SEND_BYTES;
    pub const SLOT_BYTES: usize = SLOT_BYTES;
    pub const MAX_RANKS: usize = MAX_RANKS;

    /// Connects RDMA lanes over pinned memory and spawns the proxy.
    pub fn connect(
        rank: u32,
        ranks: u32,
        fabric: &Fabric,
        boot: &mut dyn Transport,
        pinning: Option<Arc<dyn Pinning>>,
    ) -> Result<Self, Error> {
        if ranks < 2 || ranks as usize > MAX_RANKS || rank >= ranks {
            let what = format!("arx rank {rank} of {ranks} (supports 2..={MAX_RANKS})");
            return Err(Error::Fit(what));
        }
        let pin = pinning
            .as_ref()
            .ok_or_else(|| Error::Fit("arx needs GPU pinned memory".into()))?;
        let sizes = [2 * SEND_BYTES, 2 * ranks as usize * SLOT_BYTES, 4096, 4096];
        let regions: Result<Vec<Region>, Error> =
            sizes.iter().map(|&n| Region::new(n, Some(pin))).collect();
        let engine = Engine::connect(rank, ranks, fabric, SLOT_BYTES, regions?, boot)?;
        let dev = |i: usize| {
            engine.regions[i]
                .device()
                .ok_or_else(|| Error::Fit(format!("arx region {i} has no device address")))
        };
        let mem = ArxMem {
            send: dev(0)?,
            recv: dev(1)?,
            flag: dev(2)?,
            ctl: dev(3)?,
            rank,
            ranks,
            lanes: engine.lanes as u32,
            mesh: engine.wiring.relay.is_none(),
        };
        let ctl = engine.regions[3].base as *const AtomicU64;
        let fault = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&fault);
        let thread = thread::Builder::new()
            .name(format!("arx-proxy-{rank}"))
            .spawn(move || run_proxy(engine, shared))?;
        Ok(Self {
            mem,
            ctl,
            fault,
            thread: Some(thread),
        })
    }

    pub fn mem(&self) -> ArxMem {
        self.mem
    }

    /// Returns any proxy or GPU wait error.
    pub fn check(&self) -> Result<(), Error> {
        let ctl = |i: usize| unsafe { &*self.ctl.add(i) };
        if ctl(CTL_ERR).load(Ordering::Acquire) != 0 {
            let msg = self
                .fault
                .lock()
                .ok()
                .and_then(|g| g.clone())
                .unwrap_or_else(|| "arx proxy stopped".into());
            return Err(Error::Link(msg));
        }
        let gpu = ctl(CTL_GPU_ERR).load(Ordering::Acquire);
        if gpu != 0 {
            let why = match gpu {
                1 => "saw proxy error",
                2 => "timed out waiting for peers",
                _ => "failed",
            };
            return Err(Error::Link(format!("arx rank {} GPU {why}", self.mem.rank)));
        }
        Ok(())
    }
}

impl Drop for ArxLink {
    fn drop(&mut self) {
        let stop = unsafe { &*self.ctl.add(CTL_STOP) };
        stop.store(1, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn record_fault(engine: &Engine, fault: &Mutex<Option<String>>, err: Error) {
    let msg = format!("arx rank {} proxy: {err}", engine.rank);
    eprintln!("{msg}");
    if let Ok(mut slot) = fault.lock() {
        *slot = Some(msg);
    }
    engine.regions[3].word(CTL_ERR).store(1, Ordering::Release);
}

fn run_proxy(mut engine: Engine, fault: Arc<Mutex<Option<String>>>) {
    if let Err(err) = proxy_loop(&mut engine, Spinning::default()) {
        record_fault(&engine, &fault, err);
    }
}

fn flag_val(seq: u64, piece: u64) -> u64 {
    seq * FLAG_STRIDE + piece + 1
}

fn post_work(engine: &mut Engine, sends: &[usize], base: usize) -> Result<(u64, usize), Error> {
    let w = |i: usize| engine.regions[3].word(base + i).load(Ordering::Relaxed);
    let op = w(0);
    let seq = w(1);
    let piece = w(2);
    let slice = w(3) as usize;
    let lo = w(4) as usize;
    let hi = w(5) as usize;
    let par = (seq & 1) as usize;
    let len = hi.saturating_sub(lo);
    let rank = engine.rank;
    let slot = engine.slot(par, rank) + lo;
    let flag = flag_val(seq, piece);
    for &c in sends {
        let src = match op == MODE_SCATTER_PIECE {
            true => engine.wiring.channels[c].peer as usize * slice + lo,
            false => lo,
        };
        engine.post(c, (0, par * SEND_BYTES + src), len, slot, rank, flag)?;
    }
    engine.poll()?;
    Ok((seq, len))
}

fn proxy_loop(engine: &mut Engine, mut pace: impl Pace) -> Result<(), Error> {
    let mut seq = 0u64;
    let mut ring_done = 0u64;
    let mut relayed = 0u64;
    let mut own_bytes = [0usize; 2];
    let sends = engine.wiring.sends.clone();
    let relay = engine.wiring.relay;
    while engine.regions[3].word(CTL_STOP).load(Ordering::Acquire) == 0 {
        if let Some((from, to)) = relay {
            let prev = engine.wiring.channels[from].peer;
            while relayed < seq && engine.arrived(prev, flag_val(relayed + 1, PIECES - 1)) {
                relayed += 1;
                let par = (relayed & 1) as usize;
                let slot = engine.slot(par, prev);
                let flag = flag_val(relayed, PIECES - 1);
                engine.post(to, (1, slot), own_bytes[par], slot, prev, flag)?;
                pace.worked();
            }
        }
        let ring_base = CTL_RING + (ring_done as usize % RING_CAP) * WORK_WORDS;
        let tag = engine.regions[3]
            .word(ring_base + WORK_WORDS - 1)
            .load(Ordering::Acquire);
        if tag == ring_done + 1 {
            pace.worked();
            ring_done += 1;
            let (w_seq, len) = post_work(engine, &sends, ring_base)?;
            own_bytes[(w_seq & 1) as usize] = len;
            seq = seq.max(w_seq);
            continue;
        }
        match pace.idle(relay.is_some() && relayed < seq) {
            Step::Spin => std::hint::spin_loop(),
            Step::Poll => engine.poll()?,
            Step::Nap(nap) => {
                engine.poll()?;
                thread::sleep(nap);
            }
        }
    }
    Ok(())
}
