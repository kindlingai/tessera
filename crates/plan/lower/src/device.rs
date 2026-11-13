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
            use graph::DimName;
            use mem::Buffer;
            use std::collections::HashMap;
            use std::ops::Range;
            use std::time::Instant;

            pub struct Runner {
                ctx: &'static Context,
                plan: Plan,
                held: Held,
                kernels: Vec<Kernel>,
                /// Compiled kernels, by entry, and by how a kernel prints.
                made: HashMap<String, Kernel>,
                known: HashMap<String, Kernel>,
                /// Runtime dimensions, as last bound.
                bound: HashMap<DimName, u32>,
            }

            impl Runner {
                pub fn new(plan: Plan) -> Result<Self, Error> {
                    let mut runner = Self {
                        ctx: cached_context()?,
                        plan: Plan::default(),
                        held: Held::default(),
                        kernels: Vec::new(),
                        made: HashMap::new(),
                        known: HashMap::new(),
                        bound: HashMap::new(),
                    };
                    runner.replan(&plan)?;
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
                            Arg::Size(_) | Arg::Offset(..) | Arg::Span(..) => {
                                unreachable!("bound before launching")
                            }
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
                        Shape::Runtime(_) => unreachable!("bound before launching"),
                    }
                    Ok(())
                }

                /// Each launch of a step, its runtime sizes filled in.
                fn launches(&self, step: usize) -> Result<Vec<Launch>, Error> {
                    let size = |name| self.bound.get(&name).copied();
                    let bound = |l: Launch| l.bound(&size).ok_or(Error::Unbound);
                    self.plan.steps[step]
                        .launches()
                        .into_iter()
                        .map(bound)
                        .collect()
                }
            }

            impl Run for Runner {
                fn plan(&self) -> &Plan {
                    &self.plan
                }

                fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
                    if plan.id != 0 && plan.id == self.plan.id {
                        return Ok(());
                    }
                    if plan.zoned() {
                        return Err(Error::Unsupported("a plan over several zones".into()));
                    }
                    let ctx = self.ctx;
                    self.held.bind(plan, |zone, bytes| {
                        let buffer = DeviceBuffer::alloc(ctx, bytes.max(4))?;
                        // Kernels may read what they are about to write.
                        buffer.zero()?;
                        Ok(Buffer::new(zone, buffer))
                    })?;
                    self.kernels.clear();
                    for step in &plan.steps {
                        // Printing a kernel is cheaper than writing its source.
                        let key = format!("{:?}", step.kernel);
                        if let Some(kernel) = self.known.get(&key) {
                            self.kernels.push(kernel.clone());
                            continue;
                        }
                        let kernel = match step.kernel.source() {
                            Source::Fixed { cuda, entry, .. } => cached_source_kernel(cuda, entry)?,
                            Source::Made(text) => match self.made.get(&text.entry) {
                                Some(compiled) => compiled.clone(),
                                None => {
                                    let source = text.render(Dialect::Cuda);
                                    let compiled = ctx.compile(&source, &text.entry)?;
                                    self.made.insert(text.entry.clone(), compiled.clone());
                                    compiled
                                }
                            },
                        };
                        self.known.insert(key, kernel.clone());
                        self.kernels.push(kernel);
                    }
                    self.plan = plan.clone();
                    Ok(())
                }

                fn bind(&mut self, name: DimName, value: u32) {
                    self.bound.insert(name, value);
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
                    let launches = self.launches(step)?;
                    for _ in 0..times {
                        for launch in &launches {
                            self.launch(&self.kernels[step], launch)?;
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
                        for launch in self.launches(i)? {
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
