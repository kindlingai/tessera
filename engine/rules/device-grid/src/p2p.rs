//! Direct GPU-to-GPU all-reduce over IPC and SM stores.

use std::time::{Duration, Instant};
use tessera_device::resident::{Error, Status, backend, persistent};
use tessera_device::{Options, flatten};
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_gpukit::backend::{Backend as Gpu, OnGpu};
use tessera_gpukit::device::{Device, DeviceBuffer, IpcHandle, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_lower::Knobs;
use tessera_mem::DType;
use tessera_run::Sum;

const KERNEL: &str = include_str!("p2p/p2p.kernel");
const COLLECT: &str = include_str!("p2p/collect.kernel");

/// Full P2P kernel source combining sum and collective kernels.
pub(crate) fn source() -> String {
    format!("{KERNEL}\n{COLLECT}")
}
const ENTRY: &str = "tessera_p2p_sum";
const GATHER_ENTRY: &str = "tessera_p2p_gather";
const GATHER_ROWS_ENTRY: &str = "tessera_p2p_gather_rows";
const BF16_ENTRY: &str = "tessera_p2p_sum_bf16";
const FP16_ENTRY: &str = "tessera_p2p_sum_fp16";
const FP8_ENTRY: &str = "tessera_p2p_sum_fp8";
const RS_ENTRY: &str = "tessera_p2p_rs";
const RS_BF16_ENTRY: &str = "tessera_p2p_rs_bf16";
const RS_FP16_ENTRY: &str = "tessera_p2p_rs_fp16";
const RS_FP8_ENTRY: &str = "tessera_p2p_rs_fp8";

/// Maximum ranks in one P2P group.
const MAX_RANKS: usize = 8;
/// Floats per chunk. Larger sums loop over chunks.
const MAX_FLOATS: usize = 4 << 20;
const MAX_BYTES: usize = MAX_FLOATS * 4;
/// Slot stride: max payload plus a fixed tail page.
pub(crate) const SLOT_BYTES: usize = MAX_BYTES + 4096;
/// Byte threshold switching from fan-out to pairwise rounds.
pub(crate) const CROSSOVER: usize = 24 << 10;
/// Byte threshold switching four-rank sums to reduce-scatter and all-gather.
const SCATTER_CROSSOVER: usize = 256 << 10;

const BASE: u64 = 3;
const RING: usize = 4;
const CNT_STRIDE: usize = MAX_RANKS + 2;
const MAX_THREADS: u32 = 256;
/// Payload bytes where `Auto` leaves fan-out, then picks scatter.
pub(crate) const CROSS: [usize; 2] = [CROSSOVER, SCATTER_CROSSOVER];

tessera_lower::knob!("p2p_sum", "threads", 32, 1024);
tessera_lower::knob!("p2p_sum", "grid", 1, 64);
tessera_lower::knob!("p2p_sum", "min_blocks", 1, 16);
tessera_lower::knob!("p2p_sum", "crossover", 0, MAX_BYTES as u64);
tessera_lower::knob!("p2p_sum", "scatter_crossover", 0, MAX_BYTES as u64);
pub(crate) const LARGE_GRID: u32 = 16;
/// Ranks may still load when the first sum starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// Schedule choice for P2P transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Schedule {
    /// Fan-out, butterfly, or halving-doubling by payload size.
    #[default]
    Auto,
    /// Every rank writes all peers at once.
    Fanout,
    /// One peer per round: XOR for power-of-two ranks, shift otherwise.
    Rounds,
    /// Recursive doubling for power-of-two ranks, rounds otherwise.
    Butterfly,
    /// Reduce-scatter and all-gather for four ranks, butterfly otherwise.
    Scatter,
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
pub(crate) const fn pick(
    bytes: usize,
    ranks: u32,
    schedule: Schedule,
    cross: [usize; 2],
) -> (u32, u32) {
    let large = bytes >= cross[0];
    let huge = bytes >= cross[1];
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
        Schedule::Scatter => {
            if ranks == 4 {
                3
            } else if ranks >= 4 && pow2 {
                2
            } else if ranks > 2 {
                1
            } else {
                0
            }
        }
        Schedule::Auto => {
            if huge && ranks == 4 {
                3
            } else if large && ranks >= 4 && pow2 {
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
    gather_kernel: D::Kernel,
    gather_rows_kernel: D::Kernel,
    bf16_kernel: D::Kernel,
    fp16_kernel: D::Kernel,
    fp8_kernel: D::Kernel,
    rs_kernel: D::Kernel,
    rs_bf16_kernel: D::Kernel,
    rs_fp16_kernel: D::Kernel,
    rs_fp8_kernel: D::Kernel,
    _peers: Peers<D>,
    _area: DeviceBuffer<D>,
    areas: DeviceBuffer<D>,
    counters: DeviceBuffer<D>,
    fault: DeviceBuffer<D>,
    seq: DeviceBuffer<D>,
    rank: u32,
    ranks: u32,
    threads: u32,
    large_grid: u32,
    cross: [usize; 2],
    timeout_ticks: u64,
    schedule: Schedule,
    timed: bool,
    last: [u64; 2],
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
    pub fn on(ctx: &D, boot: &mut dyn Transport, knobs: &Knobs) -> Result<Self, Error> {
        let (rank, ranks) = (boot.rank(), boot.ranks());
        if ranks < 2 || ranks as usize > MAX_RANKS || rank >= ranks {
            let what = format!("p2p rank {rank} of {ranks} (supports 2..={MAX_RANKS})");
            return Err(Error::Comm(CommError::Fit(what)));
        }
        let options = Options::plain(backend::<D>()?);
        let most = knobs
            .get("p2p_sum", "threads")
            .map_or(MAX_THREADS, |t| t as u32);
        let limit = ctx.platform().limits.group_threads as u32;
        let threads = (limit.min(most) & !31).max(32);
        let large_grid = knobs
            .get("p2p_sum", "grid")
            .map_or(LARGE_GRID, |g| g as u32);
        let cross = [
            knobs.get("p2p_sum", "crossover").unwrap_or(CROSS[0]),
            knobs
                .get("p2p_sum", "scatter_crossover")
                .unwrap_or(CROSS[1]),
        ];
        let min_blocks = knobs.get("p2p_sum", "min_blocks");
        let source = source();
        let text = flatten("p2p.kernel", &source, &|_| None, &options)?.text;
        let text = match min_blocks {
            Some(mb) => {
                format!("#define P2P_THREADS {threads}\n#define P2P_MIN_BLOCKS {mb}\n{text}")
            }
            None => format!("#define P2P_THREADS {threads}\n{text}"),
        };
        let kernel = ctx.compile(&text, ENTRY)?;
        let gather_kernel = ctx.compile(&text, GATHER_ENTRY)?;
        let gather_rows_kernel = ctx.compile(&text, GATHER_ROWS_ENTRY)?;
        let bf16_kernel = ctx.compile(&text, BF16_ENTRY)?;
        let fp16_kernel = ctx.compile(&text, FP16_ENTRY)?;
        let fp8_kernel = ctx.compile(&text, FP8_ENTRY)?;
        let rs_kernel = ctx.compile(&text, RS_ENTRY)?;
        let rs_bf16_kernel = ctx.compile(&text, RS_BF16_ENTRY)?;
        let rs_fp16_kernel = ctx.compile(&text, RS_FP16_ENTRY)?;
        let rs_fp8_kernel = ctx.compile(&text, RS_FP8_ENTRY)?;

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
        let seq = DeviceBuffer::alloc(ctx, size_of::<u64>() + crate::TIMING_BYTES)?;
        seq.zero()?;
        ctx.upload(seq.device_ptr(), &BASE.to_ne_bytes())?;
        let timeout_ticks = (TIMEOUT.as_secs_f64() * ctx.ticks()) as u64;
        // Wait until every peer opened all handles.
        boot.exchange(&[])?;

        Ok(P2p {
            ctx: ctx.clone(),
            kernel,
            gather_kernel,
            gather_rows_kernel,
            bf16_kernel,
            fp16_kernel,
            fp8_kernel,
            rs_kernel,
            rs_bf16_kernel,
            rs_fp16_kernel,
            rs_fp8_kernel,
            _peers: peers,
            _area: area,
            areas,
            counters,
            fault,
            seq,
            rank,
            ranks,
            threads,
            large_grid,
            cross,
            timeout_ticks,
            schedule: Schedule::Auto,
            timed: false,
            last: [0; 2],
            broken: false,
        })
    }

    /// Overrides the schedule for tests and benchmarks.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub fn with_schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = schedule;
        self
    }

    fn read_seq(&self) -> Result<u64, Error> {
        let mut raw = [0u8; 8];
        self.ctx.download(self.seq.device_ptr(), &mut raw)?;
        Ok(u64::from_ne_bytes(raw))
    }

    fn prepare(
        &self,
        bytes: usize,
        from: u64,
        to: u64,
        large_grid: Option<u32>,
    ) -> Result<(u32, u32, KernelArgs), Error> {
        if self.broken {
            return Err(Error::Plan("an earlier p2p sum failed".into()));
        }
        let (sched, mut grid) = pick(bytes, self.ranks, self.schedule, self.cross);
        if grid > 1 {
            grid = large_grid.unwrap_or(grid);
            if self.ctx.active_lane() != 0 {
                let cap = (self.ctx.platform().units as u32 / 6).clamp(4, 16);
                grid = grid.min(cap);
            }
        }
        let mut args = KernelArgs::new();
        args.device_ptr(self.areas.device_ptr())
            .device_ptr(self.counters.device_ptr())
            .device_ptr(self.fault.device_ptr())
            .device_ptr(self.seq.device_ptr())
            .u64(from)
            .u64(to);
        Ok((sched, grid, args))
    }

    fn finish_launch(
        &self,
        kernel: &<D as Device>::Kernel,
        sched: u32,
        grid: u32,
        mut args: KernelArgs,
    ) -> Result<(), Error> {
        args.u32(self.ranks)
            .u32(self.rank)
            .u32(sched)
            .u64(self.timeout_ticks);
        self.ctx.programmatic(false);
        Ok(self
            .ctx
            .launch(kernel, (grid, 1, 1), (self.threads, 1, 1), 0, &mut args)?)
    }

    fn launch_sum(&mut self, from: u64, to: u64, n: u32, wire: DType) -> Result<(), Error> {
        let (sched, grid, mut args) =
            self.prepare(n as usize * 4, from, to, Some(self.large_grid))?;
        args.u32(n);
        let kernel = match wire {
            DType::Bf16 => &self.bf16_kernel,
            DType::F16 => &self.fp16_kernel,
            DType::Fp8 => &self.fp8_kernel,
            _ => &self.kernel,
        };
        self.finish_launch(kernel, sched, grid, args)
    }

    fn launch_gather(
        &mut self,
        from: u64,
        to: u64,
        part: u32,
        total: u32,
        rows: u32,
    ) -> Result<(), Error> {
        let (sched, grid, mut args) = self.prepare(total as usize * 4, from, to, None)?;
        args.u32(part).u32(total).u32(rows);
        self.finish_launch(&self.gather_kernel, sched, grid, args)
    }

    fn launch_rs(
        &mut self,
        from: u64,
        to: u64,
        start: u32,
        len: u32,
        total: u32,
        wire: DType,
    ) -> Result<(), Error> {
        let even = len > 0
            && u64::from(start) == u64::from(self.rank) * u64::from(len)
            && u64::from(len) * u64::from(self.ranks) == u64::from(total);
        let bytes = if even { len as usize } else { total as usize } * 4;
        let (sched, grid, mut args) = self.prepare(bytes, from, to, Some(self.large_grid))?;
        args.u32(start).u32(len).u32(total);
        let kernel = match wire {
            DType::Bf16 => &self.rs_bf16_kernel,
            DType::F16 => &self.rs_fp16_kernel,
            DType::Fp8 => &self.rs_fp8_kernel,
            _ => &self.rs_kernel,
        };
        self.finish_launch(kernel, sched, grid, args)
    }

    fn launch_gather_rows(
        &mut self,
        from: u64,
        to: u64,
        start: u32,
        part: u32,
        limit: u32,
        bytes: usize,
    ) -> Result<(), Error> {
        let (sched, grid, mut args) = self.prepare(bytes, from, to, Some(self.large_grid))?;
        args.u32(start).u32(part).u32(limit);
        self.finish_launch(&self.gather_rows_kernel, sched, grid, args)
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
    fn sum(
        &mut self,
        from: u64,
        to: u64,
        len: usize,
        wire: DType,
    ) -> Result<(), tessera_run::Error> {
        if !self.carries(wire) {
            return Err(tessera_run::Error::Wire(wire));
        }
        if self.broken {
            return Err(tessera_run::Error::Sums(Box::new(Error::Plan(
                "an earlier p2p sum failed".into(),
            ))));
        }
        let n = u32::try_from(len).map_err(|_| tessera_run::Error::SumTooLong(len))?;
        if n == 0 {
            return Ok(());
        }
        self.launch_sum(from, to, n, wire)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn carries(&self, wire: DType) -> bool {
        matches!(wire, DType::F32 | DType::Bf16 | DType::F16 | DType::Fp8)
    }

    fn gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        total: usize,
        rows: usize,
    ) -> Result<bool, tessera_run::Error> {
        if self.broken {
            return Err(tessera_run::Error::Sums(Box::new(Error::Plan(
                "an earlier p2p sum failed".into(),
            ))));
        }
        let fits = rows > 0
            && part > 0
            && total > 0
            && part.is_multiple_of(rows)
            && total.is_multiple_of(rows)
            && total / rows <= MAX_FLOATS;
        if !fits {
            return Ok(false);
        }
        let (Ok(part), Ok(total), Ok(rows)) = (
            u32::try_from(part),
            u32::try_from(total),
            u32::try_from(rows),
        ) else {
            return Ok(false);
        };
        self.launch_gather(from, to, part, total, rows)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        Ok(true)
    }

    fn reduce_scatter(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        len: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, tessera_run::Error> {
        if !self.carries(wire) {
            return Err(tessera_run::Error::Wire(wire));
        }
        if self.broken {
            return Err(tessera_run::Error::Sums(Box::new(Error::Plan(
                "an earlier p2p sum failed".into(),
            ))));
        }
        if total == 0 {
            return Ok(true);
        }
        if start > total || len > total - start {
            return Ok(false);
        }
        let (Ok(start), Ok(len), Ok(total)) = (
            u32::try_from(start),
            u32::try_from(len),
            u32::try_from(total),
        ) else {
            return Ok(false);
        };
        self.launch_rs(from, to, start, len, total, wire)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        Ok(true)
    }

    fn gather_rows(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        part: usize,
        total: usize,
        _wire: DType,
    ) -> Result<bool, tessera_run::Error> {
        if self.broken {
            return Err(tessera_run::Error::Sums(Box::new(Error::Plan(
                "an earlier p2p sum failed".into(),
            ))));
        }
        if total == 0 {
            return Ok(false);
        }
        let ranks = self.ranks as usize;
        let even = part > 0 && start == self.rank as usize * part && part * ranks == total;
        if !even && total > MAX_FLOATS {
            return Ok(false);
        }
        let limit = if even { part } else { 1 };
        let (Ok(start_u), Ok(part_u), Ok(limit_u)) = (
            u32::try_from(start),
            u32::try_from(part),
            u32::try_from(limit),
        ) else {
            return Ok(false);
        };
        self.launch_gather_rows(from, to, start_u, part_u, limit_u, total * 4)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        Ok(true)
    }

    fn capturable(&self) -> bool {
        true
    }

    fn check(&mut self) -> Result<(), tessera_run::Error> {
        self.status()
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn timing(&mut self, on: bool) {
        if self.timed != on {
            self.timed = on;
            crate::write_timed(&self.ctx, self.seq.device_ptr() + 8, on);
        }
    }

    fn spent(&mut self) -> Result<Option<[f64; 2]>, tessera_run::Error> {
        if !self.timed {
            return Ok(None);
        }
        crate::read_ticks(&self.ctx, self.seq.device_ptr() + 16, &mut self.last)
            .map(Some)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }
}

struct Start<'a>(&'a mut dyn Transport, &'a Knobs);

impl OnGpu for Start<'_> {
    type Out = Result<Box<dyn Sum>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Ok(Box::new(P2p::on(cached_context::<D>()?, self.0, self.1)?))
    }
}

/// Direct P2P sums on the GPU `gpu` names.
pub fn open(
    gpu: Option<Gpu>,
    boot: &mut dyn Transport,
    knobs: &Knobs,
) -> Result<Box<dyn Sum>, Error> {
    persistent(gpu, Start(boot, knobs))
}
