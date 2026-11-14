//! Replays a plan on CUDA or ROCm, from one body.

macro_rules! device_runner {
    ($bk:ident, $variant:ident) => {
        pub mod $bk {
            use crate::kernels::Kernels;
            use crate::run::{Base, Run};
            use codegen::Dialect;
            use gpukit::device::Device;
            use gpukit::$bk::{
                Context, DeviceBuffer, Graph, Kernel, KernelArgs, cached_context,
                cached_source_kernel, launch_1d, launch_grid,
            };
            use graph::DimName;
            use lower::{Arg, Error, Extent, Launch, Launching, Plan, Shape, Slot};
            use mem::Buffer;
            use std::collections::HashMap;
            use std::ops::Range;
            use std::time::Instant;

            pub(crate) struct Runner {
                ctx: &'static Context,
                base: Base,
                kernels: Vec<Kernel>,
                compiled: Kernels<Kernel>,
                /// The whole plan's launches per binding, recorded once.
                graphs: HashMap<Vec<(DimName, u32)>, Graph>,
                launching: Launching,
                /// Whether the device overlaps dependent launches at all.
                overlaps: bool,
                /// Which steps' kernels wait on the one before them.
                waits: Vec<bool>,
            }

            impl Runner {
                pub fn new(plan: Plan) -> Result<Self, Error> {
                    let ctx = cached_context()?;
                    let mut runner = Self {
                        ctx,
                        base: Base::default(),
                        kernels: Vec::new(),
                        compiled: Kernels::default(),
                        graphs: HashMap::new(),
                        launching: Launching::default(),
                        // Asked once: asking may launch, and captures come later.
                        overlaps: ctx.limits().pdl,
                        waits: Vec::new(),
                    };
                    runner.replan(&plan)?;
                    Ok(runner)
                }

                fn buffer(&self, slot: Slot) -> &DeviceBuffer {
                    let backing = self.base.held.get(slot).backing().as_any();
                    backing
                        .downcast_ref::<DeviceBuffer>()
                        .expect("this runner allocates device buffers")
                }

                /// Passes each argument by position, then enqueues.
                fn launch(&self, step: usize, launch: &Launch) -> Result<(), Error> {
                    let kernel = &self.kernels[step];
                    let overlap = self.launching.pdl && self.overlaps && self.waits[step];
                    self.ctx.programmatic(overlap);
                    let mut args = KernelArgs::new();
                    for arg in &launch.args {
                        match arg {
                            Arg::Buffer(slot, Extent::Fixed(skip)) => {
                                let at = slot.at(*skip) as u64;
                                args.device_ptr(self.buffer(*slot).device_ptr() + at)
                            }
                            Arg::Value(bytes) => args.bytes(bytes),
                            Arg::Size(_) | Arg::Buffer(..) => {
                                unreachable!("bound before launching")
                            }
                        };
                    }
                    match launch.shape {
                        Shape::Line(count) => launch_1d(self.ctx, kernel, count, &mut args)?,
                        Shape::Grid {
                            across,
                            down,
                            threads,
                        } => {
                            let threads = threads as u32;
                            launch_grid(self.ctx, kernel, across, down, threads, &mut args)?
                        }
                        Shape::Runtime(_) => unreachable!("bound before launching"),
                    }
                    Ok(())
                }
            }

            impl Run for Runner {
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
                        let fixed = |text, entry| -> Result<_, Error> {
                            Ok(cached_source_kernel(text, entry)?)
                        };
                        let made = |source: &str, entry: &str| -> Result<_, Error> {
                            Ok(ctx.compile(source, entry)?)
                        };
                        // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
                        let (kernel, generated) =
                            self.compiled.get(step, Dialect::Cuda, fixed, made)?;
                        self.kernels.push(kernel);
                        self.waits.push(generated);
                    }
                    self.base.plan = plan.clone();
                    Ok(())
                }

                fn launching(&mut self, launching: Launching) {
                    self.launching = launching;
                    self.graphs.clear();
                }

                fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
                    self.ctx.synchronize()?;
                    let start = Instant::now();
                    let launches = self.base.launches(step)?;
                    for _ in 0..times {
                        for launch in &launches {
                            self.launch(step, launch)?;
                        }
                    }
                    self.ctx.synchronize()?;
                    Ok(start.elapsed().as_secs_f64() / times as f64)
                }

                fn device(&self) -> String {
                    format!("{} {}", stringify!($bk), self.ctx.name())
                }

                fn platform(&self) -> gpukit::Platform {
                    self.ctx.platform()
                }

                fn precision(&self) -> codegen::Precision {
                    crate::precision(gpukit::backend::Backend::$variant)
                }

                /// Launches every step, as a recorded graph when whole.
                fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
                    let whole = self.launching.graphs
                        && self.base.plan.id != 0
                        && steps == (0..self.base.plan.steps.len());
                    if !whole {
                        for i in steps {
                            for launch in self.base.launches(i)? {
                                self.launch(i, &launch)?;
                            }
                        }
                        return Ok(self.ctx.synchronize()?);
                    }
                    let mut key: Vec<(DimName, u32)> =
                        self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
                    key.sort_unstable();
                    if !self.graphs.contains_key(&key) {
                        let graph = self.ctx.capture(|| -> Result<(), Error> {
                            for i in steps.clone() {
                                for launch in self.base.launches(i)? {
                                    self.launch(i, &launch)?;
                                }
                            }
                            Ok(())
                        })?;
                        self.graphs.insert(key.clone(), graph);
                    }
                    self.ctx.replay(&self.graphs[&key])?;
                    Ok(self.ctx.synchronize()?)
                }
            }
        }
    };
}

#[cfg(feature = "cuda")]
device_runner!(cuda, Cuda);

#[cfg(feature = "rocm")]
device_runner!(rocm, Rocm);
