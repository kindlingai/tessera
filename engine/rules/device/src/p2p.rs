//! Direct GPU-to-GPU all-reduce over IPC and SM stores.

use crate::resident::{Error, Status, backend, persistent};
use crate::{Options, flatten};
use std::time::{Duration, Instant};
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_gpukit::backend::{Backend as Gpu, OnGpu};
use tessera_gpukit::device::{Device, DeviceBuffer, IpcHandle, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_run::Sum;

const KERNEL: &str = include_str!("p2p/p2p.kernel");
const ENTRY: &str = "tessera_p2p_sum";

/// Maximum ranks in one P2P group.
const MAX_RANKS: usize = 8;
/// Floats per chunk. Larger sums loop over chunks.
const MAX_FLOATS: usize = 2 << 20;
const MAX_BYTES: usize = MAX_FLOATS * 4;
/// Slot stride: max payload plus a fixed tail page.
pub(crate) const SLOT_BYTES: usize = MAX_BYTES + 4096;
/// Byte threshold switching from fan-out to pairwise rounds.
pub(crate) const CROSSOVER: usize = 24 << 10;

const BASE: u64 = 3;
const RING: usize = 4;
const CNT_STRIDE: usize = MAX_RANKS + 2;
const MAX_THREADS: u32 = 1024;
const LARGE_GRID: u32 = 4;
/// Ranks may still load when the first sum starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Schedule choice for P2P transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Schedule {
    /// Fan-out below `CROSSOVER`, butterfly or rounds from `CROSSOVER` up.
    #[default]
    Auto,
    /// Every rank writes all peers at once.
    Fanout,
    /// One peer per round: XOR for power-of-two ranks, shift otherwise.
    Rounds,
    /// Recursive doubling for power-of-two ranks, rounds otherwise.
    Butterfly,
}

/// Bytes in one rank's IPC area.
pub(crate) const fn area_bytes(ranks: u32) -> usize {
    ack_offset(ranks) + 4096
}

/// Ack array offset in one rank's IPC area.
pub(crate) const fn ack_offset(ranks: u32) -> usize {
    2 * ranks as usize * SLOT_BYTES
}

/// Picks `(sched_code, grid_blocks)` for `bytes` across `ranks`.
pub(crate) const fn pick(bytes: usize, ranks: u32, schedule: Schedule) -> (u32, u32) {
    let large = bytes >= CROSSOVER;
    let pow2 = ranks.is_power_of_two();
    let sched = match schedule {
        Schedule::Fanout => 0,
        Schedule::Rounds => {
            if ranks > 2 {
                1
            } else {
                0
            }
        }
        Schedule::Butterfly => {
            if ranks >= 4 && pow2 {
                2
            } else if ranks > 2 {
                1
            } else {
                0
            }
        }
        Schedule::Auto => {
            if large && ranks >= 4 && pow2 {
                2
            } else if large && ranks > 2 {
                1
            } else {
                0
            }
        }
    };
    let grid = if large { LARGE_GRID } else { 1 };
    (sched, grid)
}

/// Peer IPC mappings, closed on drop.
struct Peers<D: Device> {
    ctx: D,
    opened: Vec<u64>,
}

impl<D: Device> Drop for Peers<D> {
    fn drop(&mut self) {
        for ptr in self.opened.drain(..) {
            self.ctx.ipc_close(ptr);
        }
    }
}

/// Direct P2P all-reduce over CUDA or HIP IPC.
pub struct P2p<D: Device> {
    ctx: D,
    kernel: D::Kernel,
    _peers: Peers<D>,
    _area: DeviceBuffer<D>,
    areas: DeviceBuffer<D>,
    counters: DeviceBuffer<D>,
    fault: DeviceBuffer<D>,
    seq: DeviceBuffer<D>,
    rank: u32,
    ranks: u32,
    threads: u32,
    timeout_ticks: u64,
    schedule: Schedule,
    broken: bool,
}

/// How long drop waits for peers' final acks.
const DRAIN: Duration = Duration::from_secs(2);

impl<D: Device> Drop for P2p<D> {
    fn drop(&mut self) {
        let _ = self.ctx.finish();
        if self.broken {
            return;
        }
        let mut seq_raw = [0u8; 8];
        if self.seq.read_bytes(&mut seq_raw).is_err() {
            return;
        }
        let seq = u64::from_ne_bytes(seq_raw);
        if seq <= BASE {
            return;
        }
        let want = seq - 1;
        let ptr = self._area.device_ptr() + ack_offset(self.ranks) as u64;
        let mut raw = vec![0u8; self.ranks as usize * 8];
        let start = Instant::now();
        while start.elapsed() < DRAIN {
            if self.ctx.download(ptr, &mut raw).is_err() {
                break;
            }
            let (chunks, _) = raw.as_chunks::<8>();
            let done = chunks
                .iter()
                .enumerate()
                .all(|(p, b)| p == self.rank as usize || u64::from_ne_bytes(*b) >= want);
            if done {
                break;
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

impl<D> P2p<D>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    /// Opens direct P2P on `ctx`, exchanging IPC handles over `boot`.
    pub fn on(ctx: &D, boot: &mut dyn Transport) -> Result<Self, Error> {
        let (rank, ranks) = (boot.rank(), boot.ranks());
        if ranks < 2 || ranks as usize > MAX_RANKS || rank >= ranks {
            let what = format!("p2p rank {rank} of {ranks} (supports 2..={MAX_RANKS})");
            return Err(Error::Comm(CommError::Fit(what)));
        }
        let options = Options {
            backend: backend::<D>()?,
            remote_kv: false,
            marks: false,
        };
        let text = flatten("p2p.kernel", KERNEL, &|_| None, &options)?.text;
        let kernel = ctx.compile(&text, ENTRY)?;

        let area = DeviceBuffer::alloc(ctx, area_bytes(ranks))?;
        area.zero()?;
        let acks = vec![BASE - 1; ranks as usize];
        ctx.upload(
            area.device_ptr() + ack_offset(ranks) as u64,
            bytes_of(&acks),
        )?;
        ctx.finish()?;

        let handle = ctx.ipc_export(area.device_ptr())?;
        let raw_handles = boot.exchange(&handle.0)?;
        if raw_handles.len() != ranks as usize {
            return Err(Error::Comm(CommError::Fit("handle count mismatch".into())));
        }
        let mut table = [0u64; MAX_RANKS];
        table[rank as usize] = area.device_ptr();
        let mut peers = Peers {
            ctx: ctx.clone(),
            opened: Vec::with_capacity(ranks as usize - 1),
        };
        for (p, raw) in raw_handles.iter().enumerate() {
            if p == rank as usize {
                continue;
            }
            let bytes: [u8; 64] = raw
                .as_slice()
                .try_into()
                .map_err(|_| Error::Comm(CommError::Fit(format!("rank {p} handle size"))))?;
            let ptr = ctx.ipc_open(IpcHandle(bytes))?;
            peers.opened.push(ptr);
            table[p] = ptr;
        }
        let areas = DeviceBuffer::alloc(ctx, size_of_val(&table))?;
        areas.write_bytes(bytes_of(&table))?;
        let counters = DeviceBuffer::alloc(ctx, RING * CNT_STRIDE * size_of::<u32>())?;
        counters.zero()?;
        let fault = DeviceBuffer::alloc(ctx, size_of::<u32>())?;
        fault.zero()?;
        let seq = DeviceBuffer::alloc(ctx, size_of::<u64>())?;
        seq.write_bytes(&BASE.to_ne_bytes())?;
        let timeout_ticks = (TIMEOUT.as_secs_f64() * ctx.ticks()) as u64;
        // Wait until every peer opened all handles.
        boot.exchange(&[])?;

        let threads = (ctx.platform().limits.group_threads as u32).clamp(1, MAX_THREADS);
        Ok(P2p {
            ctx: ctx.clone(),
            kernel,
            _peers: peers,
            _area: area,
            areas,
            counters,
            fault,
            seq,
            rank,
            ranks,
            threads,
            timeout_ticks,
            schedule: Schedule::Auto,
            broken: false,
        })
    }

    /// Overrides the schedule for tests and benchmarks.
    pub fn with_schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = schedule;
        self
    }

    fn read_seq(&self) -> Result<u64, Error> {
        let mut raw = [0u8; 8];
        self.seq.read_bytes(&mut raw)?;
        Ok(u64::from_ne_bytes(raw))
    }

    fn launch_sum(&mut self, from: u64, to: u64, n: u32) -> Result<(), Error> {
        if self.broken {
            return Err(Error::Plan("an earlier p2p sum failed".into()));
        }
        let (sched, grid) = pick(n as usize * 4, self.ranks, self.schedule);
        let mut args = KernelArgs::new();
        args.device_ptr(self.areas.device_ptr())
            .device_ptr(self.counters.device_ptr())
            .device_ptr(self.fault.device_ptr())
            .device_ptr(self.seq.device_ptr())
            .u64(from)
            .u64(to)
            .u32(n)
            .u32(self.ranks)
            .u32(self.rank)
            .u32(sched)
            .u64(self.timeout_ticks);
        self.ctx.programmatic(false);
        let blocks = (grid, 1, 1);
        let threads = (self.threads, 1, 1);
        Ok(self
            .ctx
            .launch(&self.kernel, blocks, threads, 0, &mut args)?)
    }

    fn status(&mut self) -> Result<(), Error> {
        self.ctx.finish()?;
        let mut raw = [0u8; 4];
        self.fault.read_bytes(&mut raw)?;
        let code = u32::from_ne_bytes(raw);
        if code != 0 {
            self.broken = true;
            return Err(Error::Fault {
                code: Status(code),
                pool: 0,
                ordinal: self.read_seq().unwrap_or(BASE),
            });
        }
        Ok(())
    }
}

impl<D> Sum for P2p<D>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    fn sum(&mut self, from: u64, to: u64, len: usize) -> Result<(), tessera_run::Error> {
        if self.broken {
            return Err(tessera_run::Error::Sums(Box::new(Error::Plan(
                "an earlier p2p sum failed".into(),
            ))));
        }
        let n = u32::try_from(len).map_err(|_| tessera_run::Error::SumTooLong(len))?;
        if n == 0 {
            return Ok(());
        }
        self.launch_sum(from, to, n)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn capturable(&self) -> bool {
        true
    }

    fn check(&mut self) -> Result<(), tessera_run::Error> {
        self.status()
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }
}

struct Start<'a>(&'a mut dyn Transport);

impl OnGpu for Start<'_> {
    type Out = Result<Box<dyn Sum>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Ok(Box::new(P2p::on(cached_context::<D>()?, self.0)?))
    }
}

/// Direct P2P sums on the GPU `gpu` names.
pub fn open(gpu: Option<Gpu>, boot: &mut dyn Transport) -> Result<Box<dyn Sum>, Error> {
    persistent(gpu, Start(boot))
}
