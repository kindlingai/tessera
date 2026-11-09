//! Replays a plan on CUDA or ROCm, from one body.

macro_rules! device_runner {
    ($bk:ident) => {
        pub mod $bk {
            use crate::Error;
            use crate::abi::{Arg, Launch, Shape, Source};
            use crate::emit::Dialect;
            use crate::plan::{Plan, Slot};
            use crate::run::Run;
            use gpukit::device::Device;
            use gpukit::$bk::{
                Context, DeviceBuffer, Kernel, KernelArgs, cached_context, cached_source_kernel,
                launch_1d, launch_grid, launch_rows,
            };
            use std::collections::HashMap;
            use std::ops::Range;

            pub struct Runner {
                ctx: &'static Context,
                plan: Plan,
                buffers: Vec<DeviceBuffer>,
                kernels: Vec<Kernel>,
            }

            impl Runner {
                pub fn new(plan: Plan) -> Result<Self, Error> {
                    let ctx = cached_context()?;
                    let mut buffers = Vec::with_capacity(plan.buffers.len());
                    for len in &plan.buffers {
                        let buffer = DeviceBuffer::alloc(ctx, (len * 4).max(4))?;
                        // Kernels may read what they are about to write.
                        buffer.zero()?;
                        buffers.push(buffer);
                    }
                    let mut made: HashMap<String, Kernel> = HashMap::new();
                    let mut kernels = Vec::with_capacity(plan.steps.len());
                    for step in &plan.steps {
                        let text = match step.kernel.source() {
                            Source::Fixed { cuda, entry, .. } => {
                                kernels.push(cached_source_kernel(cuda, entry)?);
                                continue;
                            }
                            Source::Made(text) => text,
                        };
                        if !made.contains_key(&text.entry) {
                            let compiled = ctx.compile(&text.render(Dialect::Cuda), &text.entry)?;
                            made.insert(text.entry.clone(), compiled);
                        }
                        kernels.push(made[&text.entry].clone());
                    }
                    Ok(Self {
                        ctx,
                        plan,
                        buffers,
                        kernels,
                    })
                }

                fn buffer(&self, slot: Slot) -> &DeviceBuffer {
                    &self.buffers[slot.buffer as usize]
                }

                /// Passes each argument by position, then enqueues.
                fn launch(&self, kernel: &Kernel, launch: &Launch) -> Result<(), Error> {
                    let mut args = KernelArgs::new();
                    for arg in &launch.args {
                        match arg {
                            Arg::Buffer(slot, skip) => {
                                let at = self.buffer(*slot).device_ptr() + (skip * 4) as u64;
                                args.device_ptr(at)
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

                fn write(&self, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
                    Ok(self.buffer(slot).write_bytes(bytes)?)
                }

                fn read(&self, slot: Slot) -> Result<Vec<f32>, Error> {
                    let mut out = vec![0.0; slot.len];
                    self.buffer(slot).read_f32(&mut out)?;
                    Ok(out)
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
