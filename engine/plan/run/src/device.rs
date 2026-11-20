//! Replays a plan on any GPU over the device traits.

use crate::Error;
use crate::kernels::Kernels;
use crate::run::{Base, Run};
use sensai_codegen::Dialect;
use sensai_engine_settings::Memory;
use sensai_gpukit::backend::Backend;
use sensai_gpukit::device::{Device, DeviceBuffer, cached_context, cached_source};
use sensai_gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use sensai_graph::DimName;
use sensai_lower::{Arg, Extent, Launch as Call, Launching, Plan, Shape, Slot};
use sensai_mem::Buffer;
use std::collections::HashMap;
use std::ops::Range;
use std::time::Instant;

pub(crate) struct Runner<D: Device> {
    ctx: &'static D,
    base: Base,
    kernels: Vec<D::Kernel>,
    compiled: Kernels<D::Kernel>,
    /// The whole plan's launches per binding, recorded once.
    graphs: HashMap<Vec<(DimName, u32)>, D::Graph>,
    launching: Launching,
    /// Whether the device overlaps dependent launches at all.
    overlaps: bool,
    /// Which steps' kernels wait on the one before them.
    waits: Vec<bool>,
}

/// The kernel text a backend compiles.
fn dialect(backend: Backend) -> Dialect {
    match backend {
        Backend::Metal => Dialect::Metal,
        Backend::Cuda | Backend::Rocm => Dialect::Cuda,
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    pub fn new(plan: Plan, memory: Memory) -> Result<Self, Error> {
        let ctx = cached_context::<D>()?;
        let mut runner = Self {
            ctx,
            base: Base::new(memory, &ctx.platform()),
            kernels: Vec::new(),
            compiled: Kernels::default(),
            graphs: HashMap::new(),
            launching: Launching::default(),
            // Asked once: asking may launch, and captures come later.
            overlaps: ctx.platform().limits.pdl,
            waits: Vec::new(),
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &DeviceBuffer<D> {
        let backing = self.base.held.get(slot).backing().as_any();
        backing
            .downcast_ref::<DeviceBuffer<D>>()
            .expect("this runner allocates device buffers")
    }

    /// Each argument by position, as the device takes it.
    fn bound(&self, args: &[Arg]) -> Result<KernelArgs, Error> {
        let mut bound = KernelArgs::new();
        for arg in args {
            match arg {
                Arg::Buffer(slot, Extent::Fixed(skip)) => {
                    let at = slot.at(*skip) as u64;
                    bound.device_ptr(self.buffer(*slot).device_ptr() + at)
                }
                Arg::Value(bytes) => bound.bytes(bytes),
                Arg::Size(_) | Arg::Buffer(..) => return Err(sensai_lower::Error::Unbound.into()),
            };
        }
        Ok(bound)
    }

    fn fire(
        &self,
        kernel: &<D as Device>::Kernel,
        shape: Shape,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        match shape {
            Shape::Line(count) => launch_1d(self.ctx, kernel, count, args)?,
            Shape::Grid {
                across,
                down,
                threads,
            } => {
                let threads = threads as u32;
                launch_grid(self.ctx, kernel, across, down, threads, args)?
            }
            Shape::Runtime(_) => return Err(sensai_lower::Error::Unbound.into()),
        }
        Ok(())
    }

    /// Passes each argument by position, then enqueues.
    fn launch(&self, step: usize, launch: &Call) -> Result<(), Error> {
        let overlap = self.launching.pdl && self.overlaps && self.waits[step];
        self.ctx.programmatic(overlap);
        let mut args = self.bound(&launch.args)?;
        self.fire(&self.kernels[step], launch.shape, &mut args)
    }

    /// Times each step by device stamps, else by waiting.
    fn profiled(&mut self, steps: Range<usize>) -> Result<(), Error> {
        let stamped = self.ctx.stamp()?;
        let mut waited = Vec::new();
        for i in steps.clone() {
            let start = Instant::now();
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
            match stamped {
                true => _ = self.ctx.stamp()?,
                false => {
                    let clocked = self.ctx.finish()?;
                    waited.push(clocked.unwrap_or_else(|| start.elapsed().as_secs_f64()));
                }
            }
        }
        let seconds = match stamped {
            true => self.ctx.stamps()?,
            false => waited,
        };
        for (i, s) in steps.zip(seconds) {
            let name = self.base.plan.steps[i].kernel.name();
            self.note(name, s);
        }
        Ok(())
    }

    /// Every launch of `steps`, in order.
    fn launch_all(&self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
        }
        Ok(())
    }
}

/// Pinned host memory from the shared context.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct Pin<D: Device>(pub(crate) &'static D);

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<D: Device> sensai_engine_comm::Pinning for Pin<D> {
    fn pin(&self, len: usize) -> Result<sensai_engine_comm::Pinned, sensai_engine_comm::Error> {
        let (host, device) = self
            .0
            .pin(len)
            .map_err(|e| sensai_engine_comm::Error::Link(e.to_string()))?;
        Ok(sensai_engine_comm::Pinned {
            host: host as *mut u8,
            device,
            len,
        })
    }

    fn unpin(&self, pinned: sensai_engine_comm::Pinned) {
        self.0.unpin(pinned.host as u64);
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Run for Runner<D> {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        if !self.base.fresh(plan)? {
            return Ok(());
        }
        // Buffers may move, so recorded launches go stale.
        self.graphs.clear();
        let ctx = self.ctx;
        self.base.held.bind(plan, |zone, bytes| {
            let buffer = DeviceBuffer::alloc(ctx, bytes.max(4))?;
            // Kernels may read what they are about to write.
            buffer.zero()?;
            Ok(Buffer::new(zone, buffer))
        })?;
        self.kernels.clear();
        self.waits.clear();
        for step in &plan.steps {
            let fixed = |text, entry| -> Result<_, Error> { Ok(cached_source::<D>(text, entry)?) };
            let made =
                |source: &str, entry: &str| -> Result<_, Error> { Ok(ctx.compile(source, entry)?) };
            // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
            let (kernel, generated) = self.compiled.get(step, dialect(D::BACKEND), fixed, made)?;
            self.kernels.push(kernel);
            self.waits.push(generated);
        }
        self.base.adopt(plan);
        Ok(())
    }

    fn launching(&mut self, launching: Launching) {
        self.launching = launching;
        self.graphs.clear();
    }

    /// Hand-written kernels are CUDA text.
    fn fixed(&self) -> bool {
        dialect(D::BACKEND) == Dialect::Cuda
    }

    fn launch_fixed(
        &self,
        source: &'static str,
        entry: &'static str,
        args: &[Arg],
        shape: Shape,
    ) -> Result<(), Error> {
        let kernel = cached_source::<D>(source, entry)?;
        self.ctx.programmatic(false);
        let mut args = self.bound(args)?;
        self.fire(&kernel, shape, &mut args)?;
        self.ctx.finish()?;
        Ok(())
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.ctx.finish()?;
        let start = Instant::now();
        let launches = self.base.launches(step)?;
        for _ in 0..times {
            for launch in &launches {
                self.launch(step, launch)?;
            }
        }
        let clocked = self.ctx.finish()?;
        let seconds = clocked.unwrap_or_else(|| start.elapsed().as_secs_f64());
        Ok(seconds / times as f64)
    }

    fn device(&self) -> String {
        format!("{} {}", D::BACKEND, self.ctx.name())
    }

    fn platform(&self) -> sensai_gpukit::Platform {
        self.ctx.platform()
    }

    fn precision(&self) -> sensai_codegen::Precision {
        crate::precision(match D::BACKEND {
            Backend::Metal => sensai_engine_settings::Backend::Metal,
            Backend::Cuda => sensai_engine_settings::Backend::Cuda,
            Backend::Rocm => sensai_engine_settings::Backend::Rocm,
        })
    }

    /// Launches every step, as a recorded graph when whole.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        if self.base.profile.is_some() {
            return self.profiled(steps);
        }
        let whole = self.launching.graphs
            && self.ctx.graphs()
            && self.base.plan.id != 0
            && steps == (0..self.base.plan.steps.len());
        if !whole {
            self.launch_all(steps)?;
            self.ctx.finish()?;
            return Ok(());
        }
        let mut key: Vec<(DimName, u32)> = self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
        key.sort_unstable();
        if !self.graphs.contains_key(&key) {
            let graph = self.ctx.capture(|| self.launch_all(steps.clone()))?;
            self.graphs.insert(key.clone(), graph);
        }
        self.ctx.replay(&self.graphs[&key])?;
        self.ctx.finish()?;
        Ok(())
    }
}
