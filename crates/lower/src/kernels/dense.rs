//! Attention as two matmuls per head around a softmax.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::kernels::matmul::{launch, source, tiles};
use crate::plan::Step;
use crate::rule::{Attn, Draft, Read, Site};

/// The most score elements this form may hold.
pub const DENSE_SCORES: usize = 1 << 28;

/// Every query-key product, as `[Seqs, Heads, Queries, Keys]`.
#[derive(Clone, Copy, Debug)]
pub struct Scores(pub Attn);

/// Scaled causal softmax over each score row, in place.
#[derive(Clone, Copy, Debug)]
pub struct Causal {
    pub queries: usize,
    pub keys: usize,
    pub scale: f32,
}

/// Each head's probabilities times its values.
#[derive(Clone, Copy, Debug)]
pub struct Values(pub Attn);

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
        key: (s * a.keys * a.kv_heads + h / (a.heads / a.kv_heads)) * a.width,
        score: (s * a.heads + h) * a.queries * a.keys,
    };
    pairs.map(|pair| each(at(pair))).collect()
}

impl Kernel for Scores {
    fn name(&self) -> &'static str {
        "scores"
    }

    fn source(&self) -> Source {
        source()
    }

    /// Each head's rows sit a stride apart, with no copy.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let p = gemm::Problem {
            lda: a.heads * a.width,
            ldb: a.kv_heads * a.width,
            ..gemm::Problem::linear(a.queries, a.keys, a.width)
        };
        let (q, k, out) = (step.reads[0], step.reads[1], step.write);
        per_head(a, |at| {
            let buffers = [(q, at.query), (k, at.key), (out, at.score)];
            launch(p, buffers.map(|(slot, skip)| Arg::Buffer(slot, skip)))
        })
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * tiles(a.queries) * tiles(a.keys) * a.width) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }
}

// Rows are queries; the last query sees every key.
const BODY: &str = "    if (gid >= rows) {
        return;
    }
    u32 seen = keys - queries + gid % queries + 1;
    u64 at = (u64)gid * keys;
    float top = x[at] * scale;
    for (u32 i = 1; i < seen; i++) {
        top = f_max(top, x[at + i] * scale);
    }
    float sum = 0.0f;
    for (u32 i = 0; i < seen; i++) {
        sum += f_exp(x[at + i] * scale - top);
    }
    for (u32 i = 0; i < keys; i++) {
        x[at + i] = i < seen ? f_exp(x[at + i] * scale - top) / sum : 0.0f;
    }
";

impl Kernel for Causal {
    fn name(&self) -> &'static str {
        "causal"
    }

    fn source(&self) -> Source {
        let sizes = [("rows", "u32"), ("queries", "u32"), ("keys", "u32")];
        Source::Made(Text {
            entry: "plan_causal".into(),
            buffers: vec![float("x", true)],
            scalars: sizes.into_iter().chain([("scale", "float")]).collect(),
            body: BODY.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let rows = step.write.len / self.keys;
        let sizes = [rows, self.queries, self.keys].map(Arg::count);
        let args = [step.read(0)].into_iter().chain(sizes);
        vec![Launch {
            args: args.chain([Arg::F32(self.scale)]).collect(),
            shape: Shape::Line(rows),
        }]
    }

    fn overwrites(&self) -> bool {
        true
    }
}

impl Kernel for Values {
    fn name(&self) -> &'static str {
        "values"
    }

    fn source(&self) -> Source {
        source()
    }

    /// Each head writes its own columns of the output.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let p = gemm::Problem {
            ldb: a.kv_heads * a.width,
            ldc: a.heads * a.width,
            ..gemm::Problem::matmul(a.queries, a.width, a.keys)
        };
        let (probs, v, out) = (step.reads[0], step.reads[1], step.write);
        per_head(a, |at| {
            let buffers = [(probs, at.score), (v, at.key), (out, at.query)];
            launch(p, buffers.map(|(slot, skip)| Arg::Buffer(slot, skip)))
        })
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * tiles(a.queries) * tiles(a.width) * a.keys) as f64
    }

    fn guess(&self) -> f64 {
        1e-9
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Attend {
        scale,
        q,
        k,
        v,
        sizes,
    } = *site
    else {
        return None;
    };
    let pairs = sizes.seqs * sizes.heads * sizes.queries * sizes.keys;
    let out = sizes.seqs * sizes.queries * sizes.heads * sizes.width;
    let causal = Causal {
        queries: sizes.queries,
        keys: sizes.keys,
        scale,
    };
    (pairs <= DENSE_SCORES).then(|| {
        vec![
            Draft::new(Scores(sizes), [q, k], pairs),
            Draft::new(causal, [Read::Draft(0)], pairs),
            Draft::new(Values(sizes), [Read::Draft(1), Read::Value(v)], out),
        ]
    })
}

crate::rule!("dense", propose);
