//! Replays a plan on Metal. Buffers stay on the device.

mod source;

use crate::Error;
use crate::plan::{Kernel, Plan, Slot, Step};
use gpukit::metal::{BufferPool, CacheConfig, ComputePipeline, Context, div_up, set_bytes};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use ops_math::rms_norm;
use ops_model::{apply_rope_heads, gather_rows, gqa_attention};
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use std::time::Instant;

type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// One entry input's values.
pub enum Feed<'a> {
    Floats(&'a [f32]),
    Ids(&'a [u32]),
}

/// A plan with its buffers allocated and kernels compiled.
pub struct Runner {
    ctx: Rc<Context>,
    plan: Plan,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pipelines: Vec<Rc<ComputePipeline>>,
}

fn gpu(what: &'static str) -> Error {
    Error::Gpu(gpukit::Error::Gpu(what))
}

/// Threadgroups of at most 256 threads covering `count` elements.
fn line(enc: &Encoder, count: usize) {
    let threads = count.clamp(1, 256);
    grid(enc, div_up(count, threads), 1, threads);
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
        let mut made = HashMap::new();
        let mut pipelines = Vec::with_capacity(plan.steps.len());
        for step in &plan.steps {
            pipelines.push(pipeline(&ctx, &mut made, &step.kernel)?);
        }
        Ok(Self {
            ctx,
            plan,
            buffers,
            pipelines,
        })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Uploads one graph weight. Unused weights are ignored.
    pub fn load(&self, weight: u32, data: &[f32]) -> Result<(), Error> {
        for (_, slot) in self.plan.weights.iter().filter(|(w, _)| *w == weight) {
            if slot.len != data.len() {
                return Err(Error::Size(format!("weight {weight}")));
            }
            BufferPool::write_f32(self.buffer(*slot), data);
        }
        Ok(())
    }

    /// Runs every step in one command buffer.
    pub fn run(&self, inputs: &[Feed]) -> Result<Vec<Vec<f32>>, Error> {
        self.feed(inputs)?;
        self.submit(0..self.plan.steps.len())?;
        Ok(self.outputs())
    }

    /// Runs and waits step by step; returns each step's seconds.
    pub fn run_timed(&self, inputs: &[Feed]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let mut seconds = Vec::with_capacity(self.plan.steps.len());
        for i in 0..self.plan.steps.len() {
            let start = Instant::now();
            self.submit(i..i + 1)?;
            seconds.push(start.elapsed().as_secs_f64());
        }
        Ok(seconds)
    }

    fn buffer(&self, slot: Slot) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffers[slot.buffer as usize]
    }

    fn feed(&self, inputs: &[Feed]) -> Result<(), Error> {
        if inputs.len() != self.plan.inputs.len() {
            return Err(Error::Size("the input list".into()));
        }
        for (at, (slot, feed)) in self.plan.inputs.iter().zip(inputs).enumerate() {
            let bytes: Vec<u8> = match feed {
                Feed::Floats(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
                Feed::Ids(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            };
            if bytes.len() != slot.len * 4 {
                return Err(Error::Size(format!("input {at}")));
            }
            BufferPool::write_bytes(self.buffer(*slot), &bytes);
        }
        Ok(())
    }

    fn outputs(&self) -> Vec<Vec<f32>> {
        let read = |o: &crate::Output| {
            let mut out = vec![0.0; o.slot.len];
            BufferPool::read_f32(self.buffer(o.slot), &mut out);
            out
        };
        self.plan.outputs.iter().map(read).collect()
    }

    /// Encodes `steps` into one command buffer and waits for it.
    fn submit(&self, steps: Range<usize>) -> Result<(), Error> {
        let cmd = self.ctx.queue.commandBuffer();
        let cmd = cmd.ok_or_else(|| gpu("command buffer"))?;
        let enc = cmd.computeCommandEncoder();
        let enc = enc.ok_or_else(|| gpu("encoder"))?;
        for i in steps {
            enc.setComputePipelineState(&self.pipelines[i].pipeline);
            self.encode(&enc, &self.plan.steps[i]);
        }
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        Ok(())
    }

    /// Binds a slot, `skip` elements in.
    fn bind(&self, enc: &Encoder, slot: Slot, index: usize, skip: usize) {
        let buffer = Some(self.buffer(slot));
        unsafe { enc.setBuffer_offset_atIndex(buffer, skip * 4, index) };
    }

    /// Binds the reads in order, then the write.
    fn bind_all(&self, enc: &Encoder, step: &Step) -> usize {
        for (index, slot) in step.reads.iter().enumerate() {
            self.bind(enc, *slot, index, 0);
        }
        self.bind(enc, step.write, step.reads.len(), 0);
        step.reads.len() + 1
    }

    fn encode(&self, enc: &Encoder, step: &Step) {
        let pair = |a: usize, b: usize| [a as u32, b as u32];
        match &step.kernel {
            Kernel::Matmul { m, n, k } => {
                let next = self.bind_all(enc, step);
                let params = gemm::abi_params(&gemm::Problem::linear(*m, *n, *k));
                let bytes = std::ptr::NonNull::from(&params[..]).cast();
                unsafe { enc.setBytes_length_atIndex(bytes, params.len(), next) };
                let (across, down) = (div_up(*n, gemm::BN), div_up(*m, gemm::BM));
                grid(enc, across, down, gemm::THREADS as usize);
            }
            Kernel::Thin { m, n, k } => {
                let next = self.bind_all(enc, step);
                set_bytes(enc, &[*m as u32, *n as u32, *k as u32, 0], next);
                line(enc, *n);
            }
            Kernel::RmsNorm { rows, width, eps } => {
                let next = self.bind_all(enc, step);
                set_bytes(enc, &pair(*rows, *width), next);
                set_bytes(enc, eps, next + 1);
                grid(enc, 1, *rows, 256);
            }
            Kernel::Angles { width, theta } => {
                let next = self.bind_all(enc, step);
                let positions = step.reads[0].len;
                set_bytes(enc, &pair(positions, *width), next);
                set_bytes(enc, theta, next + 1);
                line(enc, positions * width / 2);
            }
            Kernel::Rope {
                positions,
                heads,
                width,
            } => {
                // The write is the first read, rotated in place.
                self.bind(enc, step.reads[0], 0, 0);
                self.bind(enc, step.reads[1], 1, 0);
                let shape = apply_rope_heads::Fixture {
                    x: Vec::new(),
                    freqs: Vec::new(),
                    seq_len: *positions,
                    num_heads: *heads,
                    head_dim: *width,
                    rotary_dim: *width,
                    elem_offset: 0,
                };
                set_bytes(enc, &shape.params(), 2);
                set_bytes(enc, &pair(*heads, *positions), 3);
                line(enc, heads * positions);
            }
            Kernel::Attend { .. } => self.attend(enc, step),
            Kernel::Gather { rows, width } => {
                // The kernel takes its output first.
                self.bind(enc, step.write, 0, 0);
                self.bind(enc, step.reads[0], 1, 0);
                self.bind(enc, step.reads[1], 2, 0);
                set_bytes(enc, &pair(*rows, *width), 3);
                grid(enc, 1, *rows, 256);
            }
            Kernel::Map(_) => {
                let next = self.bind_all(enc, step);
                set_bytes(enc, &(step.write.len as u32), next);
                line(enc, step.write.len);
            }
            Kernel::Join { left, right } => {
                let next = self.bind_all(enc, step);
                let rows = step.write.len / (left + right);
                set_bytes(enc, &[*left as u32, *right as u32, rows as u32, 0], next);
                line(enc, step.write.len);
            }
        }
    }

    /// One launch per sequence, each at its own offset.
    fn attend(&self, enc: &Encoder, step: &Step) {
        let Kernel::Attend {
            seqs,
            queries,
            keys,
            heads,
            kv_heads,
            width,
        } = step.kernel
        else {
            return;
        };
        let shape = gqa_attention::Fixture {
            q: Vec::new(),
            kv: Vec::new(),
            seq_len: queries,
            total_kv: keys,
            n_heads: heads,
            n_kv_heads: kv_heads,
            head_dim: width,
            sliding_window: 0,
            kv_cache_len: keys - queries,
        };
        let (q, kv) = (queries * heads * width, keys * kv_heads * width * 2);
        for s in 0..seqs {
            self.bind(enc, step.reads[0], 0, s * q);
            self.bind(enc, step.reads[1], 1, s * kv);
            self.bind(enc, step.write, 2, s * q);
            set_bytes(enc, &shape.params(), 3);
            set_bytes(enc, &[queries as u32, heads as u32], 4);
            grid(enc, 1, queries * heads, 256);
        }
    }
}

type Made = HashMap<String, Rc<ComputePipeline>>;

/// Hand-written kernels come from the ops crates; others are generated.
fn pipeline(ctx: &Context, made: &mut Made, k: &Kernel) -> Result<Rc<ComputePipeline>, Error> {
    let fixed = |source, entry| Ok(gpukit::metal::cached_pipeline(ctx, source, entry)?);
    let generated = match k {
        Kernel::Matmul { .. } => return fixed(gemm::METAL, gemm::ENTRY),
        Kernel::RmsNorm { .. } => return fixed(rms_norm::METAL, rms_norm::ENTRY),
        Kernel::Rope { .. } => return fixed(apply_rope_heads::METAL, apply_rope_heads::ENTRY),
        Kernel::Attend { .. } => return fixed(gqa_attention::METAL, gqa_attention::ENTRY),
        Kernel::Gather { .. } => return fixed(gather_rows::METAL, gather_rows::ENTRY),
        Kernel::Thin { .. } => source::thin(),
        Kernel::Map(stages) => source::map(stages),
        Kernel::Join { .. } => source::join(),
        Kernel::Angles { .. } => source::angles(),
    };
    if let Some(p) = made.get(&generated.entry) {
        return Ok(Rc::clone(p));
    }
    let p = Rc::new(ctx.compile_kernel(&generated.text, &generated.entry)?);
    made.insert(generated.entry, Rc::clone(&p));
    Ok(p)
}
