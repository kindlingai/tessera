// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's tok2vec and tagger: hash embeddings, maxout window layers, tags.

use graph::{DType, Dim, DimExpr, EntryKind, Graph, PathSeg, Port, Role, TensorType};
use trace::DimName::{Context, Seqs};
use trace::{Act, Embed, Linear, Model, Tensor, Vector, Weights};

/// Rows hashed per feature, and pieces per maxout unit.
pub const KEYS: u64 = 4;
pub const PIECES: u64 = 3;
/// Thinc's zero rows on each side of a text.
pub const PAD: u64 = 4;
/// Layer norm's variance floor in Thinc.
const EPS: f32 = 1e-8;

pub struct Config {
    /// Each feature's hash table rows, in feature order.
    pub rows: Vec<u64>,
    pub width: u64,
    /// Residual window layers after the embedding.
    pub depth: u64,
    pub tags: u64,
    pub max_context: u64,
}

/// The weight path for `name`, under layer `index` if given.
pub fn path(name: &str, index: Option<u32>, part: &str) -> Vec<PathSeg> {
    let mut p = vec![PathSeg::Field(name.into())];
    p.extend(index.map(PathSeg::Index));
    p.push(PathSeg::Field(part.into()));
    p
}

fn declare<W: Weights>(m: &Model, path: Vec<PathSeg>, shape: &[u64]) -> W {
    let shape: Vec<DimExpr> = shape.iter().map(|&n| n.into()).collect();
    W::declare(m, &mut path.clone(), &shape)
}

/// A maxout layer, stored by piece, then its layer norm.
struct Layer {
    w: Linear,
    b: Vector,
    gain: Vector,
    bias: Vector,
}

fn layer(m: &Model, c: &Config, i: u32, inputs: u64) -> Layer {
    let units = PIECES * c.width;
    Layer {
        w: declare(m, path("maxout", Some(i), "w"), &[inputs, units]),
        b: declare(m, path("maxout", Some(i), "b"), &[units]),
        gain: declare(m, path("norm", Some(i), "g"), &[c.width]),
        bias: declare(m, path("norm", Some(i), "b"), &[c.width]),
    }
}

/// Largest piece per unit, as `b + relu(a - b)`, then norm.
fn maxout_norm(x: &Tensor, l: &Layer, width: u64) -> Tensor {
    let h = x.matmul(&l.w).add_vector(&l.b);
    let mut best = h.slice(2, 0u64, width);
    for p in 1..PIECES {
        let piece = h.slice(2, p * width, width);
        best = best.add(&piece.add(&best.scale(-1.0)).act(Act::Relu));
    }
    best.center()
        .rms(EPS)
        .mul_vector(&l.gain)
        .add_vector(&l.bias)
}

/// `x`'s rows, flattened, then one zero row.
fn with_zero_row(x: &Tensor, len: DimExpr, width: u64) -> Tensor {
    let t = x.reshape(vec![Seqs * len, width.into()]).permute(&[1, 0]);
    // A mean over rows, zeroed: a column of zeros.
    let zero = t
        .mean(1)
        .reshape(vec![width.into(), 1u64.into()])
        .scale(0.0);
    t.concat(&zero).permute(&[1, 0])
}

/// Each padded row beside its previous and next rows.
fn window(x: &Tensor, around: &Tensor, width: u64) -> Tensor {
    let padded = Context + 2 * PAD;
    let rows = with_zero_row(x, padded.clone(), width).gather(around);
    rows.reshape(vec![Seqs.into(), padded, (3 * width).into()])
}

/// Logits for every padded row, `[Seqs, Context + 2 * PAD, tags]`.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 1),
        Dim::bounded(Context, c.max_context),
    ]);
    let ids = |shape: Vec<DimExpr>| TensorType::new(shape, DType::U32);
    let features = c.rows.len() as u64;
    let padded = Context + 2 * PAD;
    let ports = [
        // Each token's table rows: `KEYS` per feature.
        Port {
            role: Role::Tokens,
            ty: ids(vec![Seqs.into(), Context.into(), (features * KEYS).into()]),
        },
        // Each padded row's token, or the zero row.
        Port {
            role: Role::Positions,
            ty: ids(vec![Seqs.into(), padded.clone()]),
        },
        // Each padded row's previous, own and next row.
        Port {
            role: Role::Slots,
            ty: ids(vec![Seqs.into(), padded, 3u64.into()]),
        },
    ];
    let [keys, pad, around] = m.entry(EntryKind::Forward, ports);
    let total: u64 = c.rows.iter().sum();
    let table: Embed = declare(&m, vec![PathSeg::Field("embed".into())], &[total, c.width]);
    let first = layer(&m, c, 0, features * c.width);
    let rest: Vec<Layer> = (1..=c.depth as u32)
        .map(|i| layer(&m, c, i, 3 * c.width))
        .collect();
    let tag_w: Linear = declare(&m, path("tagger", None, "w"), &[c.width, c.tags]);
    let tag_b: Vector = declare(&m, path("tagger", None, "b"), &[c.tags]);

    // Each feature sums its rows. The sums are joined.
    let split = vec![
        Seqs.into(),
        Context.into(),
        features.into(),
        KEYS.into(),
        c.width.into(),
    ];
    let summed = m
        .embed(&keys, &table)
        .reshape(split)
        .mean(3)
        .scale(KEYS as f32);
    let joined = summed.reshape(vec![
        Seqs.into(),
        Context.into(),
        (features * c.width).into(),
    ]);
    let embedded = maxout_norm(&joined, &first, c.width);
    let mut x = with_zero_row(&embedded, Context.into(), c.width).gather(&pad);
    for l in &rest {
        x = x.add(&maxout_norm(&window(&x, &around, c.width), l, c.width));
    }
    let logits = x.matmul(&tag_w).add_vector(&tag_b);
    m.finish([(Role::Logits, &logits)])
}
