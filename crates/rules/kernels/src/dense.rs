//! Attention as two matmuls per head around a softmax.

use crate::matmul::{launch, source, tiles};
use graph::Order;
use lower::{
    Arg, Attn, Draft, Kernel, Launch, Read, Shape, Site, Source, Step, T, Text, Tree, float,
};

/// The most score elements this form may hold.
pub const DENSE_SCORES: usize = 1 << 28;

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
        key: (s * a.keys * a.kv_heads + h / (a.heads / a.kv_heads)) * a.width,
        score: (s * a.heads + h) * a.queries * a.keys,
    };
    pairs.map(|pair| each(at(pair))).collect()
}

impl Kernel for PerHead {
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
        source(lower::Form::F32)
    }

    /// Each head's rows sit a stride apart, with no copy.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.1;
        let scores = gemm::Problem {
            lda: a.heads * a.width,
            ldb: a.kv_heads * a.width,
            ..gemm::Problem::linear(a.queries, a.keys, a.width)
        };
        let values = gemm::Problem {
            ldb: a.kv_heads * a.width,
            ldc: a.heads * a.width,
            ..gemm::Problem::matmul(a.queries, a.width, a.keys)
        };
        let slots = [step.reads[0], step.reads[1], step.write];
        per_head(a, |at| {
            let (p, skips) = match self.0 {
                Part::Scores => (scores, [at.query, at.key, at.score]),
                Part::Values => (values, [at.score, at.key, at.query]),
            };
            launch(p, [0, 1, 2].map(|i| Arg::Buffer(slots[i], skips[i])))
        })
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.1;
        let per_head = match self.0 {
            Part::Scores => tiles(a.queries) * tiles(a.keys) * a.width,
            Part::Values => tiles(a.queries) * tiles(a.width) * a.keys,
        };
        (a.seqs * a.heads * per_head) as f64
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

    fn sizes(&self) -> Vec<usize> {
        vec![self.queries, self.keys]
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

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Attend {
        scale,
        q,
        k,
        v,
        sizes,
    } = tree.site(at)?
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
