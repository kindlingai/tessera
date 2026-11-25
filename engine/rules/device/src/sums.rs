//! A plan's sums through a persistent grid and its proxy.

use crate::proxy::Op;
use crate::resident::{Error, LanePlan, Plan, PoolPlan, Resident, persistent};
use crate::{Options, flatten};
use std::time::Duration;
use tessera_engine_comm::Link;
use tessera_gpukit::backend::{Backend as Gpu, OnGpu};
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_mem::DType;

pub use tessera_run::Sum;

const KERNEL: &str = include_str!("sum.cu");
const ENTRY: &str = "tessera_sums";
const HAND: &str = "tessera_hand";
/// Floats per round. A longer sum takes several.
const ROOM: u32 = 256 << 10;
const THREADS: u32 = 256;
/// The grid stays up, so it keeps few blocks.
const SHARE: u32 = 64;
/// Bytes of the job the hand kernel passes.
const JOB: usize = 72;
/// Where the job keeps its summed wait and move ticks.
const TICKS: usize = 56;
/// Ranks may still load when the first sum starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The most bytes in one relayed put.
pub const RELAY: usize = ROOM as usize * 4;

/// One two-way pool of every rank, slots of `RELAY` bytes.
pub(crate) fn pool(ranks: u32) -> PoolPlan {
    PoolPlan {
        depth: 2,
        one_way: false,
        group: (0..ranks).collect(),
        send_bytes: RELAY as u32,
        recv_bytes: RELAY as u32,
    }
}

pub(crate) fn plan(ranks: u32) -> Plan {
    Plan {
        pools: vec![pool(ranks)],
        lanes: vec![
            LanePlan {
                op: Op::Sum,
                elem: crate::layout::BF16,
                pool: 0,
                block: 0,
            },
            LanePlan {
                op: Op::Gather,
                elem: crate::layout::F32,
                pool: 0,
                block: 0,
            },
            LanePlan {
                op: Op::Sum,
                elem: crate::layout::F32,
                pool: 0,
                block: 0,
            },
        ],
        items: Vec::new(),
        forms: Vec::new(),
        kept: Vec::new(),
        pages: Vec::new(),
        arenas: Vec::new(),
        threads: THREADS,
        shared: 0,
        share: SHARE,
    }
}

/// The regions a link must hold for `ranks`.
pub fn regions(ranks: u32) -> Vec<usize> {
    plan(ranks).regions().to_vec()
}

/// A grid that stays up, and its hand kernel.
struct Grid<D: Device, L> {
    resident: Resident<D, L>,
    ctx: D,
    hand: <D as Device>::Kernel,
    job: DeviceBuffer<D>,
    /// The grid stays running across unprofiled submits until drop.
    running: bool,
    /// Whether sums take a barrier round and keep ticks.
    timed: bool,
    /// Wait and move ticks at the last `spent`.
    last: [u64; 2],
}

impl<D, L> Grid<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    /// Queues the hand kernel. Zero floats stop the grid.
    fn hand(
        &mut self,
        from: u64,
        to: u64,
        n: u32,
        (rows, total): (u32, u32),
        wide: bool,
    ) -> Result<(), Error> {
        let timeout = (TIMEOUT.as_secs_f64() * self.ctx.ticks()) as u64;
        let mut args = KernelArgs::new();
        args.device_ptr(self.resident.table())
            .device_ptr(self.job.device_ptr())
            .u64(from)
            .u64(to)
            .u32(n)
            .u32(self.timed as u32)
            .u32(rows)
            .u32(total)
            .u32(wide as u32)
            .u64(timeout);
        self.ctx.programmatic(false);
        let one = (1, 1, 1);
        Ok(self.ctx.launch(&self.hand, one, one, 0, &mut args)?)
    }

    /// Starts the grid unless it runs.
    fn start(&mut self) -> Result<(), Error> {
        if self.running {
            return match self.resident.poll()? {
                true => Ok(()),
                false => {
                    self.running = false;
                    self.resident.join()
                }
            };
        }
        let at = self.job.device_ptr();
        let words = [at as u32, (at >> 32) as u32];
        self.resident.start(&[ROOM, words[0], words[1]])?;
        self.running = true;
        Ok(())
    }

    /// Stops a running grid and reads its fault.
    fn stop(&mut self) -> Result<(), Error> {
        if !std::mem::take(&mut self.running) {
            return self.resident.status();
        }
        self.hand(0, 0, 0, (0, 0), false)?;
        self.resident.join()
    }
}

impl<D: Device, L> Drop for Grid<D, L> {
    fn drop(&mut self) {
        if std::mem::take(&mut self.running) {
            self.resident.abort();
            let _ = self.ctx.finish();
        }
    }
}

impl<D, L> Sum for Grid<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn sum(
        &mut self,
        from: u64,
        to: u64,
        len: usize,
        wire: DType,
    ) -> Result<(), tessera_run::Error> {
        let wide = match wire {
            DType::F32 => true,
            DType::Bf16 => false,
            other => return Err(tessera_run::Error::Wire(other)),
        };
        let n = u32::try_from(len).map_err(|_| tessera_run::Error::SumTooLong(len))?;
        let queued = match n {
            0 => self.start(),
            n if self.ctx.capturing() => self.hand(from, to, n, (0, 0), wide),
            n => (self.start()).and_then(|_| self.hand(from, to, n, (0, 0), wide)),
        };
        queued.map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn wire(&self) -> DType {
        DType::Bf16
    }

    fn gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        total: usize,
        rows: usize,
    ) -> Result<bool, tessera_run::Error> {
        let fits = rows > 0
            && part > 0
            && total > 0
            && part.is_multiple_of(rows)
            && total.is_multiple_of(rows)
            && total / rows <= ROOM as usize;
        if !fits {
            return Ok(false);
        }
        let (Ok(n), Ok(rows), Ok(total)) = (
            u32::try_from(part),
            u32::try_from(rows),
            u32::try_from(total),
        ) else {
            return Ok(false);
        };
        let queued = match self.ctx.capturing() {
            true => self.hand(from, to, n, (rows, total), false),
            false => self
                .start()
                .and_then(|_| self.hand(from, to, n, (rows, total), false)),
        };
        queued.map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        Ok(true)
    }

    fn capturable(&self) -> bool {
        true
    }

    fn pause(&mut self) -> Result<(), tessera_run::Error> {
        if self.running {
            self.stop()
                .map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        }
        Ok(())
    }

    fn check(&mut self) -> Result<(), tessera_run::Error> {
        let status = match (self.running, self.timed) {
            (false, _) => self.resident.status(),
            (true, true) => self.stop(),
            (true, false) => match self.resident.poll() {
                Ok(true) => Ok(()),
                Ok(false) => {
                    self.running = false;
                    self.resident.join()
                }
                Err(e) => {
                    self.running = false;
                    self.resident.abort();
                    Err(e)
                }
            },
        };
        status.map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn timing(&mut self, on: bool) {
        if self.timed != on && self.running {
            let _ = self.stop();
        }
        self.timed = on;
    }

    fn spent(&mut self) -> Result<Option<[f64; 2]>, tessera_run::Error> {
        let mut bytes = [0u8; JOB];
        let read = self.job.read_bytes(&mut bytes);
        read.map_err(|e| tessera_run::Error::Sums(Box::new(e)))?;
        let (words, _) = bytes[TICKS..].as_chunks::<8>();
        let now = [u64::from_le_bytes(words[0]), u64::from_le_bytes(words[1])];
        let ticks = self.ctx.ticks();
        let spent = [0, 1].map(|i| now[i].wrapping_sub(self.last[i]) as f64 / ticks);
        self.last = now;
        Ok(self.timed.then_some(spent))
    }
}

/// Sums over `link` in a grid on `ctx`.
pub fn on<D, L>(ctx: &D, link: L) -> Result<Box<dyn Sum>, Error>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    let options = Options {
        backend: crate::resident::backend::<D>()?,
        remote_kv: false,
        marks: false,
    };
    let text = flatten("sum.cu", KERNEL, &|_| None, &options)?.text;
    let plan = plan(link.ranks());
    let hand = ctx.compile(&text, HAND)?;
    let job = DeviceBuffer::alloc(ctx, JOB)?;
    ctx.zero(job.device_ptr(), JOB)?;
    let resident = Resident::open(ctx, &text, ENTRY, link, &plan)?;
    Ok(Box::new(Grid {
        resident,
        ctx: ctx.clone(),
        hand,
        job,
        running: false,
        timed: false,
        last: [0; 2],
    }))
}

struct Start<L>(L);

impl<L: Link + 'static> OnGpu for Start<L> {
    type Out = Result<Box<dyn Sum>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        on(cached_context::<D>()?, self.0)
    }
}

/// Sums over `link` on the GPU `gpu` names.
pub fn open<L: Link + 'static>(gpu: Option<Gpu>, link: L) -> Result<Box<dyn Sum>, Error> {
    persistent(gpu, Start(link))
}
