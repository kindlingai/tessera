//! Replays a plan on Metal. Buffers stay on the device.

use crate::Error;
use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::Dialect;
use crate::plan::{Plan, Slot};
use crate::run::{Held, Run};
use gpukit::metal::{BufferPool, CacheConfig, ComputePipeline, Context, set_bytes};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

pub struct Runner {
    ctx: Rc<Context>,
    plan: Plan,
    held: Held<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pipelines: Vec<Rc<ComputePipeline>>,
    /// Compiled generated kernels, by entry.
    made: HashMap<String, Rc<ComputePipeline>>,
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
            plan: Plan::default(),
            held: Held::default(),
            pipelines: Vec::new(),
            made: HashMap::new(),
        };
        runner.replan(plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &ProtocolObject<dyn MTLBuffer> {
        self.held.get(slot)
    }

    /// Binds each argument at its index, then dispatches.
    fn encode(&self, enc: &Encoder, launch: &Launch) {
        for (index, arg) in launch.args.iter().enumerate() {
            match arg {
                Arg::Buffer(slot, skip) => {
                    let buffer = Some(self.buffer(*slot));
                    unsafe { enc.setBuffer_offset_atIndex(buffer, skip * 4, index) };
                }
                Arg::U32(v) => set_bytes(enc, v, index),
                Arg::F32(v) => set_bytes(enc, v, index),
                Arg::Pair(a, b) => set_bytes(enc, &[*a, *b], index),
                Arg::Bytes(bytes) => {
                    let at = std::ptr::NonNull::from(&bytes[..]).cast();
                    unsafe { enc.setBytes_length_atIndex(at, bytes.len(), index) };
                }
            }
        }
        match launch.shape {
            Shape::Line(count) => {
                let threads = count.clamp(1, 256);
                grid(enc, count.div_ceil(threads), 1, threads);
            }
            Shape::Rows(rows) => grid(enc, 1, rows, 256),
            Shape::Grid {
                across,
                down,
                threads,
            } => grid(enc, across, down, threads),
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
            for launch in self.plan.steps[i].launches() {
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
    fn plan(&self) -> &Plan {
        &self.plan
    }

    fn replan(&mut self, plan: Plan) -> Result<(), Error> {
        if plan.zoned() {
            return Err(Error::Unsupported("a plan over several zones".into()));
        }
        let (ctx, shared) = (&self.ctx, MTLResourceOptions::StorageModeShared);
        self.held.bind(&plan, |len| {
            let buffer = ctx
                .device
                .newBufferWithLength_options((len * 4).max(4), shared);
            buffer.ok_or_else(|| gpu("buffer allocation"))
        })?;
        self.pipelines.clear();
        for step in &plan.steps {
            let text = match step.kernel.source() {
                Source::Fixed { metal, entry, .. } => {
                    let fixed = gpukit::metal::cached_pipeline(ctx, metal, entry)?;
                    self.pipelines.push(fixed);
                    continue;
                }
                Source::Made(text) => text,
            };
            if !self.made.contains_key(&text.entry) {
                let compiled = ctx.compile_kernel(&text.render(Dialect::Metal), &text.entry)?;
                self.made.insert(text.entry.clone(), Rc::new(compiled));
            }
            self.pipelines.push(Rc::clone(&self.made[&text.entry]));
        }
        self.plan = plan;
        Ok(())
    }

    fn write(&self, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        BufferPool::write_bytes(self.buffer(slot), bytes);
        Ok(())
    }

    fn read(&self, slot: Slot) -> Result<Vec<f32>, Error> {
        let mut out = vec![0.0; slot.len];
        BufferPool::read_f32(self.buffer(slot), &mut out);
        Ok(out)
    }

    fn submit(&self, steps: Range<usize>) -> Result<(), Error> {
        self.commit(steps).map(drop)
    }

    fn repeat(&self, step: usize, times: usize) -> Result<f64, Error> {
        let again = std::iter::repeat_n(step, times);
        Ok(self.commit(again)? / times as f64)
    }

    fn device(&self) -> String {
        format!("metal {}", self.ctx.name())
    }
}
