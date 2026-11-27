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

/// GPU-side arx kernels paired with a host RDMA proxy link.
pub struct Arx<D: Device> {
    ctx: D,
    link: ArxLink,
    mem: ArxMem,
    one_bf16: D::Kernel,
    one_f32: D::Kernel,
    one_fp8: D::Kernel,
    two_bf16: D::Kernel,
    two_f32: D::Kernel,
    two_fp8: D::Kernel,
    rs_bf16: D::Kernel,
    rs_f32: D::Kernel,
    rs_fp8: D::Kernel,
    gr_bf16: D::Kernel,
    gr_f32: D::Kernel,
    gr_fp8: D::Kernel,
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
        let one_bf16 = ctx.compile(&text, "tessera_arx_one_bf16")?;
        let one_f32 = ctx.compile(&text, "tessera_arx_one_f32")?;
        let one_fp8 = ctx.compile(&text, "tessera_arx_one_fp8")?;
        let two_bf16 = ctx.compile(&text, "tessera_arx_two_bf16")?;
        let two_f32 = ctx.compile(&text, "tessera_arx_two_f32")?;
        let two_fp8 = ctx.compile(&text, "tessera_arx_two_fp8")?;
        let rs_bf16 = ctx.compile(&text, "tessera_arx_rs_bf16")?;
        let rs_f32 = ctx.compile(&text, "tessera_arx_rs_f32")?;
        let rs_fp8 = ctx.compile(&text, "tessera_arx_rs_fp8")?;
        let gr_bf16 = ctx.compile(&text, "tessera_arx_gr_bf16")?;
        let gr_f32 = ctx.compile(&text, "tessera_arx_gr_f32")?;
        let gr_fp8 = ctx.compile(&text, "tessera_arx_gr_fp8")?;
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
            one_bf16,
            one_f32,
            one_fp8,
            two_bf16,
            two_f32,
            two_fp8,
            rs_bf16,
            rs_f32,
            rs_fp8,
            gr_bf16,
            gr_f32,
            gr_fp8,
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

    fn max_grid(&self, full: u32) -> u32 {
        if self.ctx.active_lane() != 0 {
            (self.ctx.platform().units as u32 / 6).clamp(4, 8)
        } else {
            full
        }
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
        let step = if wire == DType::Fp8 {
            1024
        } else {
            THREADS * 8
        };
        let grid = n.div_ceil(step).clamp(1, self.max_grid(self.max_grid_cap));
        let kernel = match (two, wire) {
            (false, DType::Bf16) => &self.one_bf16,
            (false, DType::Fp8) => &self.one_fp8,
            (false, _) => &self.one_f32,
            (true, DType::Bf16) => &self.two_bf16,
            (true, DType::Fp8) => &self.two_fp8,
            (true, _) => &self.two_f32,
        };
        self.ctx.programmatic(false);
        Ok(self
            .ctx
            .launch(kernel, (grid, 1, 1), (THREADS, 1, 1), 0, &mut args)?)
    }

    fn launch_sum(&mut self, from: u64, to: u64, len: usize, wire: DType) -> Result<(), Error> {
        let ranks = self.mem.ranks as usize;
        let wire_bytes = |elems: usize| match wire {
            DType::Fp8 => (elems / 1024) * 1152 + (elems % 1024) * 4,
            DType::Bf16 => elems * 2,
            _ => elems * 4,
        };
        let align = match wire {
            DType::Fp8 => ranks * 1024,
            _ => ranks * 8,
        };
        let aligned = (from & 15) == 0 && (to & 15) == 0;
        let two_shot = self.mem.mesh
            && aligned
            && wire_bytes(len) > self.two_shot_bytes
            && len.is_multiple_of(align);
        let max_elems = match (two_shot, wire) {
            (true, DType::Fp8) => {
                let by_send = (ArxLink::SEND_BYTES / 1152) * 1024;
                let by_slot = (ArxLink::SLOT_BYTES / 1152) * 1024 * ranks;
                (by_send.min(by_slot) / align) * align
            }
            (false, DType::Fp8) => ((ArxLink::SLOT_BYTES - 4096) / 1152) * 1024,
            (true, _) => {
                let elem = if wire == DType::Bf16 { 2 } else { 4 };
                let by_send = ArxLink::SEND_BYTES / elem;
                let by_slot = (ArxLink::SLOT_BYTES / elem) * ranks;
                (by_send.min(by_slot) / align) * align
            }
            (false, _) => ArxLink::SLOT_BYTES / if wire == DType::Bf16 { 2 } else { 4 },
        };
        let mut off = 0usize;
        while off < len {
            let step = (len - off).min(max_elems);
            let f = from + (off * 4) as u64;
            let t = to + (off * 4) as u64;
            let n = u32::try_from(step).map_err(|_| Error::Plan("arx sum too long".into()))?;
            let use_two =
                two_shot && wire_bytes(step) > self.two_shot_bytes && step.is_multiple_of(align);
            self.launch_sum_kernel(f, t, n, wire, use_two)?;
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
            let grid = ((r_cnt * cols) as u32 / (THREADS * 4)).clamp(1, self.max_grid(64));
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
        let part_align = if wire == DType::Fp8 { 1024 } else { 8 };
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
        let max_total = match wire {
            DType::Fp8 => {
                let by_send = (ArxLink::SEND_BYTES / 1152) * 1024;
                let by_slot = (ArxLink::SLOT_BYTES / 1152) * 1024 * ranks;
                by_send.min(by_slot)
            }
            _ => {
                let elem = if wire == DType::Bf16 { 2 } else { 4 };
                (ArxLink::SEND_BYTES / elem).min((ArxLink::SLOT_BYTES / elem) * ranks)
            }
        };
        if total > max_total {
            return Ok(false);
        }
        let Ok(n) = u32::try_from(total) else {
            return Ok(false);
        };
        let mut args = self.base_args();
        args.u64(from).u64(to).u32(n);
        self.finish_args(&mut args);
        let step = if wire == DType::Fp8 {
            1024
        } else {
            THREADS * 8
        };
        let grid = n.div_ceil(step).clamp(1, self.max_grid(self.max_grid_cap));
        let kernel = match wire {
            DType::Bf16 => &self.rs_bf16,
            DType::Fp8 => &self.rs_fp8,
            _ => &self.rs_f32,
        };
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
        let wire_bytes = |elems: usize| match wire {
            DType::Fp8 => (elems / 1024) * 1152 + (elems % 1024) * 4,
            DType::Bf16 => elems * 2,
            _ => elems * 4,
        };
        let part_align = if wire == DType::Fp8 { 1024 } else { 8 };
        let aligned = (from & 15) == 0 && (to & 15) == 0;
        let pipelined = self.mem.mesh
            && aligned
            && part.is_multiple_of(part_align)
            && wire_bytes(total) > PIPE_GATHER_BYTES;
        let max_part = match (pipelined, wire) {
            (true, DType::Fp8) => (ArxLink::SLOT_BYTES / 1152) * 1024,
            (true, DType::Bf16) => (ArxLink::SLOT_BYTES / 2 / part_align) * part_align,
            (true, _) => (ArxLink::SLOT_BYTES / 4 / part_align) * part_align,
            (false, _) => ArxLink::SLOT_BYTES / 4,
        };
        let mut off = 0usize;
        while off < part {
            let n = (part - off).min(max_part);
            let f = from + (off * 4) as u64;
            let t = to + (off * 4) as u64;
            let mut args = self.base_args();
            args.u64(f).u64(t).u32(part_stride).u32(n as u32);
            self.finish_args(&mut args);
            let use_pipe = pipelined
                && n.is_multiple_of(part_align)
                && wire_bytes(n * ranks) > PIPE_GATHER_BYTES;
            let (kernel, grid) = if use_pipe {
                let step = if wire == DType::Fp8 {
                    1024
                } else {
                    THREADS * 8
                };
                let g = ((n * ranks) as u32)
                    .div_ceil(step)
                    .clamp(1, self.max_grid(self.max_grid_cap));
                let k = match wire {
                    DType::Bf16 => &self.gr_bf16,
                    DType::Fp8 => &self.gr_fp8,
                    _ => &self.gr_f32,
                };
                (k, g)
            } else {
                let g = ((n * ranks) as u32 / (THREADS * 4))
                    .clamp(1, self.max_grid(self.gather_max_grid));
                (&self.gather_rows, g)
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
