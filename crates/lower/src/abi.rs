//! Each kernel's arguments and launch shape, stated once.

use crate::emit::{self, Text};
use crate::plan::{Attn, Kernel, Slot, Step};
use ops_model::{apply_rope_heads, gather_rows, gqa_attention};

/// One kernel argument, in the order its source declares.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A slot's buffer, this many elements in.
    Buffer(Slot, usize),
    U32(u32),
    F32(f32),
    /// One `uint2` on Metal; two arguments elsewhere.
    Pair(u32, u32),
    /// A small `repr(C)` struct, passed by value.
    Bytes(Vec<u8>),
}

/// How threads are laid out over the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One thread per element, in groups of 256.
    Line(usize),
    /// One group of 256 threads per row.
    Rows(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub args: Vec<Arg>,
    pub shape: Shape,
}

/// Where a kernel's source comes from.
pub enum Source {
    /// Hand-written, with one text per dialect.
    Fixed {
        metal: &'static str,
        cuda: &'static str,
        entry: &'static str,
    },
    Made(Text),
}

fn fixed(metal: &'static str, cuda: &'static str, entry: &'static str) -> Source {
    Source::Fixed { metal, cuda, entry }
}

pub fn source(kernel: &Kernel) -> Source {
    match kernel {
        Kernel::Matmul { .. } | Kernel::Scores(_) | Kernel::Values(_) => {
            fixed(gemm::METAL, gemm::CUDA, gemm::ENTRY)
        }
        Kernel::Causal { .. } => Source::Made(emit::causal()),
        Kernel::Rope { .. } => {
            use apply_rope_heads::{CUDA, ENTRY, METAL};
            fixed(METAL, CUDA, ENTRY)
        }
        Kernel::Attend(_) => {
            use gqa_attention::{CUDA, ENTRY, METAL};
            fixed(METAL, CUDA, ENTRY)
        }
        Kernel::Gather { .. } => fixed(gather_rows::METAL, gather_rows::CUDA, gather_rows::ENTRY),
        Kernel::RmsNorm { .. } => Source::Made(emit::rms_norm()),
        Kernel::Softmax { .. } => Source::Made(emit::softmax()),
        Kernel::Thin { .. } => Source::Made(emit::thin()),
        Kernel::Map(stages) => Source::Made(emit::map(stages)),
        Kernel::Join { .. } => Source::Made(emit::join()),
        Kernel::Angles { .. } => Source::Made(emit::angles()),
    }
}

/// The bytes of a `repr(C)` value.
fn bytes<T: Copy>(value: &T) -> Arg {
    let at = std::ptr::from_ref(value).cast::<u8>();
    Arg::Bytes(unsafe { std::slice::from_raw_parts(at, size_of::<T>()) }.to_vec())
}

fn pair(a: usize, b: usize) -> Arg {
    Arg::Pair(a as u32, b as u32)
}

/// The launches that run one step, usually one.
pub fn launches(step: &Step) -> Vec<Launch> {
    let read = |i: usize| Arg::Buffer(step.reads[i], 0);
    let write = Arg::Buffer(step.write, 0);
    let len = step.write.len;
    let count = |n: usize| Arg::U32(n as u32);
    let (args, shape) = match step.kernel {
        Kernel::Matmul { m, n, k } => {
            let params = Arg::Bytes(gemm::abi_params(&gemm::Problem::linear(m, n, k)));
            let shape = Shape::Grid {
                across: n.div_ceil(gemm::BN),
                down: m.div_ceil(gemm::BM),
                threads: gemm::THREADS as usize,
            };
            (vec![read(0), read(1), write, params], shape)
        }
        Kernel::Thin { m, n, k } => {
            let args = vec![read(0), read(1), write, count(m), count(n), count(k)];
            (args, Shape::Line(n))
        }
        Kernel::RmsNorm { rows, width, eps } => {
            let sizes = [count(rows), count(width), Arg::F32(eps)];
            let args = [read(0), read(1), write].into_iter().chain(sizes);
            (args.collect(), Shape::Line(rows))
        }
        Kernel::Softmax { rows, width } => {
            let args = vec![read(0), write, count(rows), count(width)];
            (args, Shape::Line(rows))
        }
        Kernel::Angles { width, theta } => {
            let positions = step.reads[0].len;
            let args = vec![
                read(0),
                write,
                count(positions),
                count(width),
                Arg::F32(theta),
            ];
            (args, Shape::Line(positions * width / 2))
        }
        // It rotates its first argument where it lies.
        Kernel::Rope {
            positions,
            heads,
            width,
        } => {
            let shape = apply_rope_heads::Fixture {
                x: Vec::new(),
                freqs: Vec::new(),
                seq_len: positions,
                num_heads: heads,
                head_dim: width,
                rotary_dim: width,
                elem_offset: 0,
            };
            let args = vec![
                read(0),
                read(1),
                bytes(&shape.params()),
                pair(heads, positions),
            ];
            (args, Shape::Line(heads * positions))
        }
        Kernel::Attend(a) => return attend(step, &a),
        Kernel::Scores(a) => return scores(step, &a),
        Kernel::Values(a) => return values(step, &a),
        Kernel::Causal {
            queries,
            keys,
            scale,
        } => {
            let sizes = [count(len / keys), count(queries), count(keys)];
            let args = [read(0)].into_iter().chain(sizes).chain([Arg::F32(scale)]);
            (args.collect(), Shape::Line(len / keys))
        }
        // The kernel takes its output first.
        Kernel::Gather { rows, width } => {
            let args = vec![write, read(0), read(1), pair(rows, width)];
            (args, Shape::Rows(rows))
        }
        Kernel::Map(_) => {
            let mut args: Vec<Arg> = (0..step.reads.len()).map(read).collect();
            args.extend([write, count(len)]);
            (args, Shape::Line(len))
        }
        Kernel::Join { left, right } => {
            let rows = count(len / (left + right));
            let args = vec![read(0), read(1), write, count(left), count(right), rows];
            (args, Shape::Line(len))
        }
    };
    vec![Launch { args, shape }]
}

/// One launch per sequence, each at its own offset.
fn attend(step: &Step, a: &Attn) -> Vec<Launch> {
    let shape = gqa_attention::Fixture {
        q: Vec::new(),
        kv: Vec::new(),
        seq_len: a.queries,
        total_kv: a.keys,
        n_heads: a.heads,
        n_kv_heads: a.kv_heads,
        head_dim: a.width,
        sliding_window: 0,
        kv_cache_len: a.keys - a.queries,
    };
    let (q, kv) = (
        a.queries * a.heads * a.width,
        a.keys * a.kv_heads * a.width * 2,
    );
    let launch = |s: usize| Launch {
        args: vec![
            Arg::Buffer(step.reads[0], s * q),
            Arg::Buffer(step.reads[1], s * kv),
            Arg::Buffer(step.write, s * q),
            bytes(&shape.params()),
            pair(a.queries, a.heads),
        ],
        shape: Shape::Rows(a.queries * a.heads),
    };
    (0..a.seqs).map(launch).collect()
}

/// One tiled matmul per sequence and head, over strided rows.
fn per_head(a: &Attn, p: gemm::Problem, at: impl Fn(usize, usize) -> [Arg; 3]) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: p.n.div_ceil(gemm::BN),
        down: p.m.div_ceil(gemm::BM),
        threads: gemm::THREADS as usize,
    };
    let pairs = (0..a.seqs).flat_map(|s| (0..a.heads).map(move |h| (s, h)));
    let launch = |(s, h)| Launch {
        args: Vec::from(at(s, h))
            .into_iter()
            .chain([Arg::Bytes(gemm::abi_params(&p))])
            .collect(),
        shape,
    };
    pairs.map(launch).collect()
}

/// Where head `h` of sequence `s` starts in each tensor.
struct Starts {
    /// In `[Seqs, Queries, Heads, Width]`.
    query: usize,
    /// In `[Seqs, Keys, KvHeads, Width]`.
    key: usize,
    /// In `[Seqs, Heads, Queries, Keys]`.
    score: usize,
}

fn starts(a: &Attn, s: usize, h: usize) -> Starts {
    let kv = h / (a.heads / a.kv_heads);
    Starts {
        query: (s * a.queries * a.heads + h) * a.width,
        key: (s * a.keys * a.kv_heads + kv) * a.width,
        score: (s * a.heads + h) * a.queries * a.keys,
    }
}

/// Queries times keys, each head's rows a stride apart.
fn scores(step: &Step, a: &Attn) -> Vec<Launch> {
    let rows = gemm::Problem::linear(a.queries, a.keys, a.width);
    let p = gemm::Problem {
        lda: a.heads * a.width,
        ldb: a.kv_heads * a.width,
        ..rows
    };
    per_head(a, p, |s, h| {
        let at = starts(a, s, h);
        let (q, k) = (step.reads[0], step.reads[1]);
        [
            Arg::Buffer(q, at.query),
            Arg::Buffer(k, at.key),
            Arg::Buffer(step.write, at.score),
        ]
    })
}

/// Probabilities times values, written into each head's columns.
fn values(step: &Step, a: &Attn) -> Vec<Launch> {
    let rows = gemm::Problem::matmul(a.queries, a.width, a.keys);
    let p = gemm::Problem {
        ldb: a.kv_heads * a.width,
        ldc: a.heads * a.width,
        ..rows
    };
    per_head(a, p, |s, h| {
        let at = starts(a, s, h);
        let (probs, v) = (step.reads[0], step.reads[1]);
        [
            Arg::Buffer(probs, at.score),
            Arg::Buffer(v, at.key),
            Arg::Buffer(step.write, at.query),
        ]
    })
}
