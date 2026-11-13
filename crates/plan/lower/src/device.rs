//! Replays a plan on CUDA or ROCm, from one body.

macro_rules! device_runner {
    ($bk:ident) => {
        pub mod $bk {
            use crate::Error;
            use crate::abi::{Arg, Launch, Shape, Source};
            use crate::plan::{Plan, Slot};
            use crate::run::{Held, Run};
            use codegen::Dialect;
            use gpukit::device::Device;
            use gpukit::$bk::{
                Context, DeviceBuffer, Kernel, KernelArgs, cached_context, cached_source_kernel,
                launch_1d, launch_grid, launch_rows,
            };
            use mem::Buffer;
            use std::collections::HashMap;
            use std::ops::Range;
            use std::time::Instant;

            pub struct Runner {
                ctx: &'static Context,
                plan: Plan,
                held: Held,
                kernels: Vec<Kernel>,
                /// Compiled generated kernels, by entry.
                made: HashMap<String, Kernel>,
            }

            impl Runner {
                pub fn new(plan: Plan) -> Result<Self, Error> {
                    let mut runner = Self {
                        ctx: cached_context()?,
                        plan: Plan::default(),
                        held: Held::default(),
                        kernels: Vec::new(),
                        made: HashMap::new(),
                    };
                    runner.replan(plan)?;
                    Ok(runner)
                }

                fn buffer(&self, slot: Slot) -> &DeviceBuffer {
                    let backing = self.held.get(slot).backing().as_any();
                    backing
                        .downcast_ref::<DeviceBuffer>()
                        .expect("this runner allocates device buffers")
                }

                /// Passes each argument by position, then enqueues.
                fn launch(&self, kernel: &Kernel, launch: &Launch) -> Result<(), Error> {
                    let mut args = KernelArgs::new();
                    for arg in &launch.args {
                        match arg {
                            Arg::Buffer(slot, skip) => {
                                let at = slot.start() + skip * slot.dtype.bytes();
                                args.device_ptr(self.buffer(*slot).device_ptr() + at as u64)
                            }
                            Arg::U32(v) => args.u32(*v),
                            Arg::F32(v) => args.f32(*v),
                            Arg::Pair(a, b) => args.u32(*a).u32(*b),
                            Arg::Bytes(bytes) => args.bytes(bytes),
                        };
                    }
                    match launch.shape {
                        Shape::Line(count) => launch_1d(self.ctx, kernel, count, &mut args)?,
                        Shape::Rows(rows) => launch_rows(self.ctx, kernel, rows, 256, &mut args)?,
                        Shape::Grid {
                            across,
                            down,
                            threads,
                        } => {
                            let threads = threads as u32;
                            launch_grid(self.ctx, kernel, across, down, threads, &mut args)?
                        }
                    }
                    Ok(())
                }
            }

            impl Run for Runner {
                fn plan(&self) -> &Plan {
                    &self.plan
                }

                fn replan(&mut self, plan: Plan) -> Result<(), Error> {
                    if plan.zoned() {
                        return Err(Error::Unsupported("a plan over several zones".into()));
                    }
                    let ctx = self.ctx;
                    self.held.bind(&plan, |zone, bytes| {
                        let buffer = DeviceBuffer::alloc(ctx, bytes.max(4))?;
                        // Kernels may read what they are about to write.
                        buffer.zero()?;
                        Ok(Buffer::new(zone, buffer))
                    })?;
                    self.kernels.clear();
                    for step in &plan.steps {
                        let text = match step.kernel.source() {
                            Source::Fixed { cuda, entry, .. } => {
                                self.kernels.push(cached_source_kernel(cuda, entry)?);
                                continue;
                            }
                            Source::Made(text) => text,
                        };
                        if !self.made.contains_key(&text.entry) {
                            let compiled = ctx.compile(&text.render(Dialect::Cuda), &text.entry)?;
                            self.made.insert(text.entry.clone(), compiled);
                        }
                        self.kernels.push(self.made[&text.entry].clone());
                    }
                    self.plan = plan;
                    Ok(())
                }

                fn held(&self) -> &Held {
                    &self.held
                }

                fn held_mut(&mut self) -> &mut Held {
                    &mut self.held
                }

                fn repeat(&self, step: usize, times: usize) -> Result<f64, Error> {
                    self.ctx.synchronize()?;
                    let start = Instant::now();
                    for _ in 0..times {
                        for launch in self.plan.steps[step].launches() {
                            self.launch(&self.kernels[step], &launch)?;
                        }
                    }
                    self.ctx.synchronize()?;
                    Ok(start.elapsed().as_secs_f64() / times as f64)
                }

                fn device(&self) -> String {
                    format!("{} {}", stringify!($bk), self.ctx.name())
                }

                /// Enqueues every launch, then waits once.
                fn submit(&self, steps: Range<usize>) -> Result<(), Error> {
                    for i in steps {
                        for launch in self.plan.steps[i].launches() {
                            self.launch(&self.kernels[i], &launch)?;
                        }
                    }
                    Ok(self.ctx.synchronize()?)
                }
            }
        }
    };
}

#[cfg(feature = "cuda")]
device_runner!(cuda);

#[cfg(feature = "rocm")]
device_runner!(rocm);
