// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's tok2vec and tagger: hash embeddings, maxout window layers, tags.

use tessera_graph::{DType, Dim, DimExpr, EntryKind, Graph, PathSeg, Port, Role, TensorType};
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::{Act, Embed, Linear, Model, Tensor, Vector, Weights};

/// Rows hashed per feature, and pieces per maxout unit.
pub const KEYS: u64 = 4;
pub const PIECES: u64 = 3;
/// Thinc's zero rows on each side of a text.
pub const PAD: u64 = 4;
/// The fewest rows in one launch.
const LEAST: usize = 8;
/// An id past every row. It gathers zeros.
const NONE: u32 = u32::MAX;
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

/// `path`, then maxout piece `p`.
pub fn piece(name: &str, index: u32, part: &str, p: u32) -> Vec<PathSeg> {
    let mut q = path(name, Some(index), part);
    q.push(PathSeg::Index(p));
    q
}

/// A maxout layer, a matrix per piece, then layer norm.
struct Layer {
    w: Vec<Linear>,
    b: Vec<Vector>,
    gain: Vector,
    bias: Vector,
}

fn layer(m: &Model, c: &Config, i: u32, inputs: u64) -> Layer {
    let pieces = 0..PIECES as u32;
    Layer {
        w: (pieces.clone())
            .map(|p| declare(m, piece("maxout", i, "w", p), &[inputs, c.width]))
            .collect(),
        b: pieces
            .map(|p| declare(m, piece("maxout", i, "b", p), &[c.width]))
            .collect(),
        gain: declare(m, path("norm", Some(i), "g"), &[c.width]),
        bias: declare(m, path("norm", Some(i), "b"), &[c.width]),
    }
}

/// Largest piece per unit, as `b + relu(a - b)`, then norm.
fn maxout_norm(x: &Tensor, l: &Layer) -> Tensor {
    let piece = |p: usize| x.matmul(&l.w[p]).add_vector(&l.b[p]);
    let mut best = piece(0);
    for p in 1..PIECES as usize {
        let gap = piece(p).add(&best.scale(-1.0));
        best = best.add(&gap.act(Act::Relu));
    }
    best.center()
        .rms(EPS)
        .mul_vector(&l.gain)
        .add_vector(&l.bias)
}

/// Rows of `x` at `ids`. Ids past them read zeros.
fn rows(x: &Tensor, len: DimExpr, width: u64, ids: &Tensor) -> Tensor {
    x.reshape(vec![Seqs * len, width.into()]).gather(ids)
}

/// Two features of eight rows, one window layer.
pub fn small() -> Config {
    Config {
        rows: vec![8, 8],
        width: 32,
        depth: 1,
        tags: 32,
        max_context: 16,
    }
}

crate::preset::preset!("tagger", small: || program(&small()));

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
        // Each padded row's token, or any id past them.
        Port {
            role: Role::Positions,
            ty: ids(vec![Seqs.into(), padded.clone()]),
        },
        // Each padded row's previous, own and next row.
        Port {
            role: Role::Slots,
            ty: ids(vec![Seqs.into(), padded.clone(), 3u64.into()]),
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
    let embedded = maxout_norm(&joined, &first);
    let mut x = rows(&embedded, Context.into(), c.width, &pad);
    for l in &rest {
        let window = rows(&x, padded.clone(), c.width, &around);
        let window = window.reshape(vec![Seqs.into(), padded.clone(), (3 * c.width).into()]);
        x = x.add(&maxout_norm(&window, l));
    }
    let logits = x.matmul(&tag_w).add_vector(&tag_b);
    m.finish([(Role::Logits, &logits)])
}

/// The three graph inputs for one text, as flat ids.
pub struct Input {
    /// Token rows, rounded up to share plans.
    pub rows: usize,
    /// Ids per token: `KEYS` per feature.
    pub width: usize,
    /// `[rows, width]`: each token's table rows. Extra rows hold zeros.
    pub keys: Vec<u32>,
    /// `[padded]`: each padded row's token, or an id past them.
    pub sources: Vec<u32>,
    /// `[padded, 3]`: each padded row's previous, own and next row.
    pub around: Vec<u32>,
}

/// Inputs for `tokens` tokens with `width` row ids each.
pub fn input(ids: &[u32], tokens: usize, width: usize) -> Input {
    let rows = tokens.next_power_of_two().max(LEAST);
    let mut keys = vec![0; rows * width];
    keys[..ids.len()].copy_from_slice(ids);
    let pad = PAD as usize;
    let padded = rows + 2 * pad;
    let mut sources = Vec::with_capacity(padded);
    let mut around = Vec::with_capacity(3 * padded);
    for j in 0..padded {
        let inside = (pad..pad + tokens).contains(&j);
        sources.push(if inside { (j - pad) as u32 } else { NONE });
        let prev = if j == 0 { NONE } else { (j - 1) as u32 };
        let next = if j + 1 == padded {
            NONE
        } else {
            (j + 1) as u32
        };
        around.extend([prev, j as u32, next]);
    }
    Input {
        rows,
        width,
        keys,
        sources,
        around,
    }
}
