//! Dual-root RoCE all-reduce over pinned memory and a host proxy.

use crate::resident::{Error, backend, persistent};
use crate::{Options, flatten};
use std::time::Duration;
use tessera_engine_comm::{ArxLink, ArxMem};
use tessera_gpukit::backend::{Backend as Gpu, OnGpu};
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_lower::Knobs;
use tessera_mem::DType;
use tessera_run::Sum;

const KERNEL: &str = include_str!("arx.cu");
const THREADS: u32 = 256;
/// Byte threshold switching from one-shot to two-shot RS/AG.
const TWO_SHOT_BYTES: usize = 256 << 10;
/// Wire-byte threshold switching gather_rows to 8-piece pipelining.
const PIPE_GATHER_BYTES: usize = 1 << 20;
/// Ranks may still load when the first sum starts.
const TIMEOUT: Duration = Duration::from_secs(300);

tessera_lower::knob!("arx_sum", "max_grid", 1, 256);
tessera_lower::knob!("arx_sum", "gather_max_grid", 1, 256);
tessera_lower::knob!("arx_sum", "two_shot_bytes", 0, 128 << 20);
tessera_lower::knob!("arx_sum", "min_blocks", 1, 16);

/// Index of `wire` in a bf16, f32, fp8 kernel array.
fn slot(wire: DType) -> usize {
    match wire {
        DType::Bf16 => 0,
        DType::Fp8 => 2,
        _ => 1,
    }
}

/// Elements one pipelined block moves on `wire`.
fn pipe_step(wire: DType) -> u32 {
    if wire == DType::Fp8 {
        1024
    } else {
        THREADS * 8
    }
}

/// Element alignment a pipelined part needs on `wire`.
fn pipe_align(wire: DType) -> usize {
    if wire == DType::Fp8 { 1024 } else { 8 }
}

/// Bytes `elems` floats take on `wire`.
fn wire_bytes(wire: DType, elems: usize) -> usize {
    match wire {
        DType::Fp8 => (elems / 1024) * 1152 + (elems % 1024) * 4,
        DType::Bf16 => elems * 2,
        _ => elems * 4,
    }
}

/// Whole floats that `bytes` hold on `wire`.
fn wire_elems(wire: DType, bytes: usize) -> usize {
    match wire {
        DType::Fp8 => (bytes / 1152) * 1024,
        DType::Bf16 => bytes / 2,
        _ => bytes / 4,
    }
}

/// Floats that fit one send buffer and `ranks` slots.
fn send_fit(wire: DType, ranks: usize) -> usize {
    wire_elems(wire, ArxLink::SEND_BYTES).min(wire_elems(wire, ArxLink::SLOT_BYTES) * ranks)
}

/// GPU-side arx kernels paired with a host RDMA proxy link.
pub struct Arx<D: Device> {
    ctx: D,
    link: ArxLink,
    mem: ArxMem,
    one: [D::Kernel; 3],
    two: [D::Kernel; 3],
    rs: [D::Kernel; 3],
    gr: [D::Kernel; 3],
    allgather: D::Kernel,
    gather_rows: D::Kernel,
    state: DeviceBuffer<D>,
    blocks: DeviceBuffer<D>,
    max_grid_cap: u32,
    gather_max_grid: u32,
    two_shot_bytes: usize,
    timeout_ticks: u64,
    timed: bool,
    last: [u64; 2],
    broken: bool,
}

impl<D: Device> Drop for Arx<D> {
    fn drop(&mut self) {
        let _ = self.ctx.finish();
    }
}

impl<D> Arx<D>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    /// Compiles arx kernels on `ctx` over `link`.
    pub fn on(ctx: &D, link: ArxLink, knobs: &Knobs) -> Result<Self, Error> {
        let options = Options {
            backend: backend::<D>()?,
            remote_kv: false,
            marks: false,
        };
        let mut text = flatten("arx.cu", KERNEL, &|_| None, &options)?.text;
        if let Some(mb) = knobs.get("arx_sum", "min_blocks") {
            let v = mb.to_string();
            let prefix = tessera_lower::define_prefix([("ARX_MIN_BLOCKS", v.as_str())])
                .expect("valid cuda define");
            text = format!("{prefix}{text}");
        }
        let max_grid_cap = knobs.get("arx_sum", "max_grid").map_or(48, |v| v as u32);
        let gather_max_grid = knobs
            .get("arx_sum", "gather_max_grid")
            .map_or(64, |v| v as u32);
        let two_shot_bytes = knobs
            .get("arx_sum", "two_shot_bytes")
            .unwrap_or(TWO_SHOT_BYTES);
        let wired = |base: &str| -> Result<[<D as Device>::Kernel; 3], Error> {
            let k = |w: &str| ctx.compile(&text, &format!("tessera_arx_{base}_{w}"));
            Ok([k("bf16")?, k("f32")?, k("fp8")?])
        };
        let one = wired("one")?;
        let two = wired("two")?;
        let rs = wired("rs")?;
        let gr = wired("gr")?;
        let allgather = ctx.compile(&text, "tessera_arx_allgather")?;
        let gather_rows = ctx.compile(&text, "tessera_arx_gather_rows")?;
        let state = DeviceBuffer::alloc(ctx, 32 + crate::TIMING_BYTES)?;
        let blocks = DeviceBuffer::alloc(ctx, 96)?;
        state.zero()?;
        blocks.zero()?;
        ctx.finish()?;
        let mem = link.mem();
        let timeout_ticks = (TIMEOUT.as_secs_f64() * ctx.ticks()) as u64;
        Ok(Self {
            ctx: ctx.clone(),
            link,
            mem,
            one,
            two,
            rs,
            gr,
            allgather,
            gather_rows,
            state,
            blocks,
            max_grid_cap,
            gather_max_grid,
            two_shot_bytes,
            timeout_ticks,
            timed: false,
            last: [0; 2],
            broken: false,
        })
    }

    fn base_args(&self) -> KernelArgs {
        let mut args = KernelArgs::new();
        args.u64(self.mem.send)
            .u64(self.mem.recv)
            .u64(self.mem.flag)
            .u64(self.mem.ctl)
            .device_ptr(self.state.device_ptr())
            .device_ptr(self.blocks.device_ptr());
        args
    }

    fn finish_args(&self, args: &mut KernelArgs) {
        args.u32(self.mem.rank)
            .u32(self.mem.ranks)
            .u32(self.mem.lanes)
            .u64(self.timeout_ticks);
    }

    fn grid(&self, blocks: u32, full: u32) -> u32 {
        let cap = if self.ctx.active_lane() != 0 {
            (self.ctx.platform().units as u32 / 6).clamp(4, 8)
        } else {
            full
        };
        blocks.clamp(1, cap)
    }

    fn launch_sum_kernel(
        &mut self,
        from: u64,
        to: u64,
        n: u32,
        wire: DType,
        two: bool,
    ) -> Result<(), Error> {
        let mut args = self.base_args();
        args.u64(from).u64(to).u32(n);
        self.finish_args(&mut args);
        let grid = self.grid(n.div_ceil(pipe_step(wire)), self.max_grid_cap);
        let kernel = if two {
            &self.two[slot(wire)]
        } else {
            &self.one[slot(wire)]
        };
        self.ctx.programmatic(false);
        Ok(self
            .ctx
            .launch(kernel, (grid, 1, 1), (THREADS, 1, 1), 0, &mut args)?)
    }

    fn launch_sum(&mut self, from: u64, to: u64, len: usize, wire: DType) -> Result<(), Error> {
        let ranks = self.mem.ranks as usize;
        let align = match wire {
            DType::Fp8 => ranks * 1024,
            _ => ranks * 8,
        };
        let aligned = (from & 15) == 0 && (to & 15) == 0;
        let two_shot = self.mem.mesh
            && aligned
            && wire_bytes(wire, len) > self.two_shot_bytes
            && len.is_multiple_of(align);
        let max_elems = match (two_shot, wire) {
            (true, _) => (send_fit(wire, ranks) / align) * align,
            (false, DType::Fp8) => wire_elems(wire, ArxLink::SLOT_BYTES - 4096),
            (false, _) => wire_elems(wire, ArxLink::SLOT_BYTES),
        };
        let mut off = 0usize;
        while off < len {
            let step = (len - off).min(max_elems);
            let off4 = (off * 4) as u64;
            let n = u32::try_from(step).map_err(|_| Error::Plan("arx sum too long".into()))?;
            let use_two = two_shot
                && wire_bytes(wire, step) > self.two_shot_bytes
                && step.is_multiple_of(align);
            self.launch_sum_kernel(from + off4, to + off4, n, wire, use_two)?;
            off += step;
        }
        Ok(())
    }
    fn launch_gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        total: usize,
        rows: usize,
    ) -> Result<bool, Error> {
        let ranks = self.mem.ranks as usize;
        if self.broken || rows == 0 || part == 0 || !part.is_multiple_of(rows) {
            return Ok(false);
        }
        if part * ranks != total {
            return Ok(false);
        }
        let width = part / rows;
        let cols = total / rows;
        if width * 4 > ArxLink::SLOT_BYTES {
            return Ok(false);
        }
        let step_rows = (ArxLink::SLOT_BYTES / 4 / width).max(1);
        let mut r0 = 0usize;
        while r0 < rows {
            let r_cnt = (rows - r0).min(step_rows);
            let f = from + (r0 * width * 4) as u64;
            let t = to + (r0 * cols * 4) as u64;
            let mut args = self.base_args();
            args.u64(f).u64(t).u32(r_cnt as u32).u32(width as u32);
            self.finish_args(&mut args);
            let grid = self.grid((r_cnt * cols) as u32 / (THREADS * 4), 64);
            self.ctx.programmatic(false);
            self.ctx
                .launch(&self.allgather, (grid, 1, 1), (THREADS, 1, 1), 0, &mut args)?;
            r0 += r_cnt;
        }
        Ok(true)
    }

    fn launch_rs(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        len: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, Error> {
        let ranks = self.mem.ranks as usize;
        let part_align = pipe_align(wire);
        let aligned = (from & 15) == 0 && (to & 15) == 0;
        let fits = !self.broken
            && self.mem.mesh
            && aligned
            && len > 0
            && start == self.mem.rank as usize * len
            && len * ranks == total
            && len.is_multiple_of(part_align);
        if !fits {
            return Ok(false);
        }
        if total > send_fit(wire, ranks) {
            return Ok(false);
        }
        let Ok(n) = u32::try_from(total) else {
            return Ok(false);
        };
        let mut args = self.base_args();
        args.u64(from).u64(to).u32(n);
        self.finish_args(&mut args);
        let grid = self.grid(n.div_ceil(pipe_step(wire)), self.max_grid_cap);
        let kernel = &self.rs[slot(wire)];
        self.ctx.programmatic(false);
        self.ctx
            .launch(kernel, (grid, 1, 1), (THREADS, 1, 1), 0, &mut args)?;
        Ok(true)
    }

    fn launch_gather_rows(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        part: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, Error> {
        let ranks = self.mem.ranks as usize;
        if self.broken
            || part == 0
            || start != self.mem.rank as usize * part
            || part * ranks != total
        {
            return Ok(false);
        }
        let Ok(part_stride) = u32::try_from(part) else {
            return Ok(false);
        };
        let part_align = pipe_align(wire);
        let aligned = (from & 15) == 0 && (to & 15) == 0;
        let pipelined = self.mem.mesh
            && aligned
            && part.is_multiple_of(part_align)
            && wire_bytes(wire, total) > PIPE_GATHER_BYTES;
        let max_part = match pipelined {
            true => (wire_elems(wire, ArxLink::SLOT_BYTES) / part_align) * part_align,
            false => ArxLink::SLOT_BYTES / 4,
        };
        let mut off = 0usize;
        while off < part {
            let n = (part - off).min(max_part);
            let off4 = (off * 4) as u64;
            let mut args = self.base_args();
            args.u64(from + off4)
                .u64(to + off4)
                .u32(part_stride)
                .u32(n as u32);
            self.finish_args(&mut args);
            let use_pipe = pipelined
                && n.is_multiple_of(part_align)
                && wire_bytes(wire, n * ranks) > PIPE_GATHER_BYTES;
            let elems = (n * ranks) as u32;
            let grid = match use_pipe {
                true => self.grid(elems.div_ceil(pipe_step(wire)), self.max_grid_cap),
                false => self.grid(elems / (THREADS * 4), self.gather_max_grid),
            };
            let kernel = if use_pipe {
                &self.gr[slot(wire)]
            } else {
                &self.gather_rows
            };
            self.ctx.programmatic(false);
            self.ctx
                .launch(kernel, (grid, 1, 1), (THREADS, 1, 1), 0, &mut args)?;
            off += n;
        }
        Ok(true)
    }
}

impl<D> Sum for Arx<D>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    fn capturable(&self) -> bool {
        true
    }

    fn wire(&self) -> DType {
        DType::Bf16
    }

    fn carries(&self, wire: DType) -> bool {
        matches!(wire, DType::F32 | DType::Bf16 | DType::Fp8)
    }

    fn check(&mut self) -> Result<(), tessera_run::Error> {
        if let Err(e) = self.link.check() {
            self.broken = true;
            return Err(tessera_run::Error::Sums(Box::new(Error::Comm(e))));
        }
        Ok(())
    }

    fn timing(&mut self, on: bool) {
        if self.timed != on {
            self.timed = on;
            crate::write_timed(&self.ctx, self.state.device_ptr() + 32, on);
        }
    }

    fn spent(&mut self) -> Result<Option<[f64; 2]>, tessera_run::Error> {
        if !self.timed {
            return Ok(None);
        }
        crate::read_ticks(&self.ctx, self.state.device_ptr() + 40, &mut self.last)
            .map(Some)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

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
                "an earlier arx sum failed".into(),
            ))));
        }
        if len == 0 {
            return Ok(());
        }
        self.launch_sum(from, to, len, wire)
            .map_err(|e| tessera_run::Error::Sums(Box::new(e)))
    }

    fn gather(
        &mut self,
        from: u64,
        to: u64,
        part: usize,
        total: usize,
        rows: usize,
    ) -> Result<bool, tessera_run::Error> {
        arx_err(self.launch_gather(from, to, part, total, rows))
    }

    fn gather_rows(
        &mut self,
        from: u64,
        to: u64,
        start: usize,
        part: usize,
        total: usize,
        wire: DType,
    ) -> Result<bool, tessera_run::Error> {
        if !self.carries(wire) {
            return Err(tessera_run::Error::Wire(wire));
        }
        arx_err(self.launch_gather_rows(from, to, start, part, total, wire))
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
        arx_err(self.launch_rs(from, to, start, len, total, wire))
    }
}

fn arx_err<T>(res: Result<T, Error>) -> Result<T, tessera_run::Error> {
    res.map_err(|e| tessera_run::Error::Sums(Box::new(e)))
}

struct Start<'a>(ArxLink, &'a Knobs);

impl OnGpu for Start<'_> {
    type Out = Result<Box<dyn Sum>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Ok(Box::new(Arx::on(cached_context::<D>()?, self.0, self.1)?))
    }
}

/// Opens arx sums over `link` on the GPU `gpu` names.
pub fn open(gpu: Option<Gpu>, link: ArxLink, knobs: &Knobs) -> Result<Box<dyn Sum>, Error> {
    persistent(gpu, Start(link, knobs))
}
