//! Replays a plan on Metal. Buffers stay on the device.

use crate::run::{Base, Kernels, Run, line};
use codegen::Dialect;
use gpukit::metal::{CacheConfig, ComputePipeline, Context, Shared};
use lower::{Arg, Error, Extent, Launch, Plan, Shape, Slot};
use mem::Buffer;
use objc2::msg_send;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLSize,
};
use std::ops::Range;
use std::rc::Rc;

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub(crate) struct Runner {
    ctx: Rc<Context>,
    base: Base,
    pipelines: Vec<Rc<ComputePipeline>>,
    compiled: Kernels<Rc<ComputePipeline>>,
}

fn gpu(what: &'static str) -> Error {
    Error::Gpu(gpukit::Error::Gpu(what))
}

fn grid(enc: &Encoder, width: usize, height: usize, threads: usize) {
    let groups = MTLSize {
        width,
        height,
        depth: 1,
    };
    let group = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(groups, group);
}

impl Runner {
    pub fn new(plan: Plan) -> Result<Self, Error> {
        let mut runner = Self {
            ctx: gpukit::metal::cached_context(CacheConfig::MEMORY)?,
            base: Base::default(),
            pipelines: Vec::new(),
            compiled: Kernels::default(),
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &ProtocolObject<dyn MTLBuffer> {
        let shared = self.base.held.get(slot).backing().as_any();
        shared
            .downcast_ref::<Shared>()
            .expect("this runner allocates shared buffers")
            .buffer()
    }

    /// Binds each argument at its index, then dispatches.
    fn encode(&self, enc: &Encoder, launch: &Launch) {
        for (index, arg) in launch.args.iter().enumerate() {
            match arg {
                Arg::Buffer(slot, Extent::Fixed(skip)) => {
                    let buffer = Some(self.buffer(*slot));
                    let at = slot.at(*skip);
                    unsafe { enc.setBuffer_offset_atIndex(buffer, at, index) };
                }
                Arg::Value(bytes) => {
                    let at = std::ptr::NonNull::from(&bytes[..]).cast();
                    unsafe { enc.setBytes_length_atIndex(at, bytes.len(), index) };
                }
                Arg::Size(_) | Arg::Buffer(..) => {
                    unreachable!("bound before encoding")
                }
            }
        }
        match launch.shape {
            Shape::Line(count) => {
                let (groups, threads) = line(count);
                grid(enc, groups, 1, threads);
            }
            Shape::Grid {
                across,
                down,
                threads,
            } => grid(enc, across, down, threads),
            Shape::Runtime(_) => unreachable!("bound before encoding"),
        }
    }
}

impl Runner {
    /// Runs the steps in one command buffer. Returns GPU seconds.
    fn commit(&self, steps: impl Iterator<Item = usize>) -> Result<f64, Error> {
        let cmd = self.ctx.queue.commandBuffer();
        let cmd = cmd.ok_or_else(|| gpu("command buffer"))?;
        let enc = cmd.computeCommandEncoder();
        let enc = enc.ok_or_else(|| gpu("encoder"))?;
        for i in steps {
            enc.setComputePipelineState(&self.pipelines[i].pipeline);
            for launch in self.base.launches(i)? {
                self.encode(&enc, &launch);
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        // The GPU's own clock leaves out the scheduling wait.
        let (start, end): (f64, f64) =
            unsafe { (msg_send![&*cmd, GPUStartTime], msg_send![&*cmd, GPUEndTime]) };
        Ok(end - start)
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
        let ctx = &self.ctx;
        self.base.held.bind(plan, |zone, bytes| {
            let shared = Shared::new(&ctx.device, bytes).ok_or_else(|| gpu("buffer allocation"));
            Ok(Buffer::new(zone, shared?))
        })?;
        self.pipelines.clear();
        for step in &plan.steps {
            let fixed = |text, entry| -> Result<_, Error> {
                Ok(gpukit::metal::cached_pipeline(ctx, text, entry)?)
            };
            let made = |source: &str, entry: &str| -> Result<_, Error> {
                Ok(Rc::new(ctx.compile_kernel(source, entry)?))
            };
            let (pipeline, _) = self.compiled.get(step, Dialect::Metal, fixed, made)?;
            self.pipelines.push(pipeline);
        }
        self.base.plan = plan.clone();
        Ok(())
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.commit(steps).map(drop)
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        let again = std::iter::repeat_n(step, times);
        Ok(self.commit(again)? / times as f64)
    }

    fn device(&self) -> String {
        format!("metal {}", self.ctx.name())
    }

    fn platform(&self) -> gpukit::Platform {
        self.ctx.platform()
    }

    fn precision(&self) -> codegen::Precision {
        codegen::Precision::METAL
    }
}
