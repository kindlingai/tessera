//! Replays a plan on Metal. Buffers stay on the device.

use crate::Error;
use crate::abi::{Arg, Launch, Shape, Source, launches, source};
use crate::emit::Dialect;
use crate::plan::{Plan, Slot};
use crate::run::Run;
use gpukit::metal::{BufferPool, CacheConfig, ComputePipeline, Context, set_bytes};
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
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pipelines: Vec<Rc<ComputePipeline>>,
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
        let ctx = gpukit::metal::cached_context(CacheConfig::MEMORY)?;
        let shared = MTLResourceOptions::StorageModeShared;
        let mut buffers = Vec::with_capacity(plan.buffers.len());
        for len in &plan.buffers {
            let bytes = (len * 4).max(4);
            let buffer = ctx.device.newBufferWithLength_options(bytes, shared);
            buffers.push(buffer.ok_or_else(|| gpu("buffer allocation"))?);
        }
        let mut made: HashMap<String, Rc<ComputePipeline>> = HashMap::new();
        let mut pipelines = Vec::with_capacity(plan.steps.len());
        for step in &plan.steps {
            let text = match source(&step.kernel) {
                Source::Fixed { metal, entry, .. } => {
                    pipelines.push(gpukit::metal::cached_pipeline(&ctx, metal, entry)?);
                    continue;
                }
                Source::Made(text) => text,
            };
            if !made.contains_key(&text.entry) {
                let compiled = ctx.compile_kernel(&text.render(Dialect::Metal), &text.entry)?;
                made.insert(text.entry.clone(), Rc::new(compiled));
            }
            pipelines.push(Rc::clone(&made[&text.entry]));
        }
        Ok(Self {
            ctx,
            plan,
            buffers,
            pipelines,
        })
    }

    fn buffer(&self, slot: Slot) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffers[slot.buffer as usize]
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

impl Run for Runner {
    fn plan(&self) -> &Plan {
        &self.plan
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

    /// One command buffer holds every launch.
    fn submit(&self, steps: Range<usize>) -> Result<(), Error> {
        let cmd = self.ctx.queue.commandBuffer();
        let cmd = cmd.ok_or_else(|| gpu("command buffer"))?;
        let enc = cmd.computeCommandEncoder();
        let enc = enc.ok_or_else(|| gpu("encoder"))?;
        for i in steps {
            enc.setComputePipelineState(&self.pipelines[i].pipeline);
            for launch in launches(&self.plan.steps[i]) {
                self.encode(&enc, &launch);
            }
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }
}
