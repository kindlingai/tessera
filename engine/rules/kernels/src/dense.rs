//! Attention as two matmuls per head around a softmax.

use codegen::prelude::*;
use graph::{Mask, Order};
use lower::sites;
use lower::{
    Arg, Attn, DType, Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree, UNPROVEN,
};

/// Rows are queries. The last query sees every key.
#[kernel]
fn causal(mut x: BufMut<F32>, rows: u32, queries: u32, keys: u32, scale: f32) {
    if gid() >= rows {
        return;
    }
    let seen = keys - queries + gid() % queries + 1;
    let at = gid() as u64 * keys as u64;
    let mut top = x[at] * scale;
    for i in 1..seen {
        top = max(top, x[at + i as u64] * scale);
    }
    let mut sum = 0.0f32;
    for i in 0..seen {
        sum += exp(x[at + i as u64] * scale - top);
    }
    for i in 0..keys {
        x[at + i as u64] = if i < seen {
            exp(x[at + i as u64] * scale - top) / sum
        } else {
            0.0
        };
    }
}

/// The most score elements this form may hold.
pub const DENSE_SCORES: usize = 1 << 28;

/// The hand-written gemm, which takes either operand transposed.
fn source(form: DType) -> Source {
    Source::Fixed {
        metal: gemm::METAL,
        cuda: gemm::CUDA,
        entry: match form {
            DType::Bf16 => gemm::ENTRY_BF16,
            _ => gemm::ENTRY,
        },
    }
}

/// A count rounded up to whole 64-wide tiles.
fn tiles(n: usize) -> usize {
    n.div_ceil(64) * 64
}

/// One gemm launch on buffers `a`, `b` and `c`.
fn launch(p: gemm::Problem, buffers: [Arg; 3]) -> Launch {
    let shape = Shape::Grid {
        across: p.n.div_ceil(gemm::BN),
        down: p.m.div_ceil(gemm::BM),
        threads: gemm::THREADS as usize,
    };
    let params = Arg::Value(gemm::abi_params(&p));
    let args = Vec::from(buffers).into_iter().chain([params]);
    Launch {
        args: args.collect(),
        shape,
    }
}

/// Which of the two products a launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Every query-key product, as `[Seqs, Heads, Queries, Keys]`.
    Scores,
    /// Each head's probabilities times its values.
    Values,
}

/// One tiled matmul per head.
#[derive(Clone, Copy, Debug)]
pub struct PerHead(pub Part, pub Attn);

/// Scaled causal softmax over each score row, in place.
#[derive(Clone, Copy, Debug)]
pub struct Causal {
    pub queries: usize,
    pub keys: usize,
    pub scale: f32,
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

/// One launch per sequence and head, at that head's starts.
fn per_head(a: &Attn, each: impl Fn(Starts) -> Launch) -> Vec<Launch> {
    let pairs = (0..a.seqs).flat_map(|s| (0..a.heads).map(move |h| (s, h)));
    let at = |(s, h): (usize, usize)| Starts {
        query: (s * a.queries * a.heads + h) * a.width,
        key: (s * a.keys.most() * a.kv_heads + h / (a.heads / a.kv_heads)) * a.width,
        score: (s * a.heads + h) * a.queries * a.keys.most(),
    };
    pairs.map(|pair| each(at(pair))).collect()
}

impl Kernel for PerHead {
    fn needs(&self) -> Needs {
        Needs::text()
    }

    fn name(&self) -> &'static str {
        match self.0 {
            Part::Scores => "scores",
            Part::Values => "values",
        }
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn sizes(&self) -> Vec<usize> {
        self.1.sizes()
    }

    fn source(&self) -> Source {
        source(DType::F32)
    }

    /// Each head's rows sit a stride apart, with no copy.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.1;
        let scores = gemm::Problem {
            lda: a.heads * a.width,
            ldb: a.kv_heads * a.width,
            ..gemm::Problem::linear(a.queries, a.keys.most(), a.width)
        };
        let values = gemm::Problem {
            ldb: a.kv_heads * a.width,
            ldc: a.heads * a.width,
            ..gemm::Problem::matmul(a.queries, a.width, a.keys.most())
        };
        let slots = [step.reads[0], step.reads[1], step.write];
        per_head(a, |at| {
            let (p, skips) = match self.0 {
                Part::Scores => (scores, [at.query, at.key, at.score]),
                Part::Values => (values, [at.score, at.key, at.query]),
            };
            launch(p, [0, 1, 2].map(|i| Arg::at(slots[i], skips[i])))
        })
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.1;
        let per_head = match self.0 {
            Part::Scores => tiles(a.queries) * tiles(a.keys.most()) * a.width,
            Part::Values => tiles(a.queries) * tiles(a.width) * a.keys.most(),
        };
        (a.seqs * a.heads * per_head) as f64
    }

    fn guess(&self) -> f64 {
        UNPROVEN
    }
}

impl Kernel for Causal {
    fn name(&self) -> &'static str {
        "causal"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.queries, self.keys]
    }

    fn source(&self) -> Source {
        Source::named(causal_ir(), "plan_causal")
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let rows = step.write.len / self.keys;
        let sizes = [rows, self.queries, self.keys].map(Arg::count);
        let args = [step.read(0)].into_iter().chain(sizes);
        vec![Launch {
            args: args.chain([Arg::value(&self.scale)]).collect(),
            shape: Shape::Line(rows),
        }]
    }

    fn overwrites(&self) -> bool {
        true
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.sizes.mask != Mask::Causal {
        return None;
    }
    // Whole sequences only: the scores take a rectangular launch.
    let sites::Attend {
        scale,
        q,
        k,
        v,
        sizes,
        rows: None,
    } = site
    else {
        return None;
    };
    // The keys shape the scores, so they must be known.
    if sizes.keys.runtime() {
        return None;
    }
    let pairs = sizes.seqs * sizes.heads * sizes.queries * sizes.keys.most();
    let out = sizes.seqs * sizes.queries * sizes.heads * sizes.width;
    let causal = Causal {
        queries: sizes.queries,
        keys: sizes.keys.most(),
        scale,
    };
    (pairs <= DENSE_SCORES).then(|| {
        vec![
            Draft::new(PerHead(Part::Scores, sizes), [q, k], pairs),
            Draft::new(causal, [Read::Draft(0)], pairs),
            Draft::new(
                PerHead(Part::Values, sizes),
                [Read::Draft(1), Read::Node(v)],
                out,
            ),
        ]
    })
}

lower::rule!("dense", propose);
