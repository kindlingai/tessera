//! Replays a plan on CUDA or ROCm, from one body.

macro_rules! device_runner {
    ($bk:ident, $variant:ident) => {
        pub mod $bk {
            use crate::Error;
            use crate::abi::{Arg, Launch, Shape, Source};
            use crate::extent::Extent;
            use crate::plan::{Plan, Slot};
            use crate::recipe::Launching;
            use crate::run::{Held, Run};
            use codegen::Dialect;
            use gpukit::device::Device;
            use gpukit::$bk::{
                Context, DeviceBuffer, Graph, Kernel, KernelArgs, cached_context,
                cached_source_kernel, launch_1d, launch_grid,
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
                /// Compiled kernels, by entry.
                made: HashMap<String, Kernel>,
                /// By how a kernel prints, with whether it was generated.
                known: HashMap<String, (Kernel, bool)>,
                /// Runtime dimensions, as last bound.
                bound: HashMap<DimName, u32>,
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
                        plan: Plan::default(),
                        held: Held::default(),
                        kernels: Vec::new(),
                        made: HashMap::new(),
                        known: HashMap::new(),
                        bound: HashMap::new(),
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
                    let backing = self.held.get(slot).backing().as_any();
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
                    // Buffers may move, so recorded launches go stale.
                    self.graphs.clear();
                    let ctx = self.ctx;
                    self.held.bind(plan, |zone, bytes| {
                        let buffer = DeviceBuffer::alloc(ctx, bytes.max(4))?;
                        // Kernels may read what they are about to write.
                        buffer.zero()?;
                        Ok(Buffer::new(zone, buffer))
                    })?;
                    self.kernels.clear();
                    self.waits.clear();
                    for step in &plan.steps {
                        // Printing a kernel is cheaper than writing its source.
                        let key = format!("{:?}", step.kernel);
                        if let Some((kernel, generated)) = self.known.get(&key) {
                            self.kernels.push(kernel.clone());
                            self.waits.push(*generated);
                            continue;
                        }
                        let source = step.kernel.source();
                        // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
                        let generated = matches!(source, Source::Made(_));
                        let kernel = match source {
                            Source::Fixed { cuda, entry, .. } => cached_source_kernel(cuda, entry)?,
                            Source::Made(body) => match self.made.get(&body.entry) {
                                Some(compiled) => compiled.clone(),
                                None => {
                                    let source = body.text().render(Dialect::Cuda);
                                    let compiled = ctx.compile(&source, &body.entry)?;
                                    self.made.insert(body.entry.clone(), compiled.clone());
                                    compiled
                                }
                            },
                        };
                        self.known.insert(key, (kernel.clone(), generated));
                        self.kernels.push(kernel);
                        self.waits.push(generated);
                    }
                    self.plan = plan.clone();
                    Ok(())
                }

                fn bind(&mut self, name: DimName, value: u32) {
                    self.bound.insert(name, value);
                }

                fn launching(&mut self, launching: Launching) {
                    self.launching = launching;
                    self.graphs.clear();
                }

                fn held(&self) -> &Held {
                    &self.held
                }

                fn held_mut(&mut self) -> &mut Held {
                    &mut self.held
                }

                fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
                    self.ctx.synchronize()?;
                    let start = Instant::now();
                    let launches = self.launches(step)?;
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

                fn limits(&self) -> gpukit::Limits {
                    self.ctx.limits()
                }

                fn precision(&self) -> codegen::Precision {
                    crate::run::precision(gpukit::backend::Backend::$variant)
                }

                /// Launches every step, as a recorded graph when whole.
                fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
                    let whole = self.launching.graphs
                        && self.plan.id != 0
                        && steps == (0..self.plan.steps.len());
                    if !whole {
                        for i in steps {
                            for launch in self.launches(i)? {
                                self.launch(i, &launch)?;
                            }
                        }
                        return Ok(self.ctx.synchronize()?);
                    }
                    let mut key: Vec<(DimName, u32)> =
                        self.bound.iter().map(|(k, v)| (*k, *v)).collect();
                    key.sort_unstable();
                    if !self.graphs.contains_key(&key) {
                        let graph = self.ctx.capture(|| -> Result<(), Error> {
                            for i in steps.clone() {
                                for launch in self.launches(i)? {
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
