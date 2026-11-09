//! GLM-5.3: MLA with a sparse indexer, and MoE. Sketch.

use crate::core::*;
use crate::library::{Mlp, Moe, mlp, moe};

pub struct Config {
    pub layers: usize,
    pub dense_layers: usize,
    pub top_k: usize,
    pub attn: AttnShape,
}

pub struct AttnShape {
    pub latent: usize,
    pub rope: usize,
    /// Keys the indexer keeps per query at long context.
    pub index_keep: usize,
}

#[derive(Weights)]
pub struct Mla {
    pub norm: Norm,
    pub q_a: Linear,
    pub q_norm: Norm,
    #[split(heads)]
    pub q_b: Linear,
    pub kv_a: Linear,
    pub kv_norm: Norm,
    #[split(rows)]
    pub o_proj: Linear,
}

/// The lightning indexer: scores every key, keeps the best.
#[derive(Weights)]
pub struct Indexer {
    pub wq: Linear,
    pub wk: Linear,
    pub weights: Linear,
}

#[derive(Weights)]
pub struct Block {
    pub attn: Mla,
    pub indexer: Option<Indexer>,
    pub mlp_norm: Norm,
    pub ffn: Ffn,
}

pub enum Ffn {
    Dense(Mlp<SwiGlu>),
    Moe(Moe<SwiGlu>),
}

#[derive(Weights)]
pub struct Glm53 {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    #[split(vocab)]
    pub head: Linear,
}

/// Exported for the drafters: last hidden state and chosen layers.
pub struct Taps {
    pub last: Tensor,
    pub layers: Vec<Tensor>,
}

pub fn forward(m: &mut Model, c: &Config, w: &Glm53, tokens: &Tensor) -> (Tensor, Taps) {
    let mut x = m.embed(tokens, &w.embed).reduce();
    let mut layers = Vec::new();
    for (l, b) in w.blocks.iter().enumerate() {
        let a = attention(m, &c.attn, &b.attn, b.indexer.as_ref(), Layer(l), &x);
        x = x.add(&a.reduce());
        let h = x.rms_norm(&b.mlp_norm);
        let f = match &b.ffn {
            Ffn::Dense(d) => mlp(d, &h),
            Ffn::Moe(e) => moe(m, e, &h, c.top_k),
        };
        x = x.add(&f.reduce());
        layers.push(m.tap(&x, Layer(l)));
    }
    let h = x.rms_norm(&w.norm);
    let taps = Taps {
        last: h.clone(),
        layers,
    };
    (h.matmul(&w.head), taps)
}

/// MLA, sparse when an indexer is given. GLM-5.3-Flash reuses it.
pub fn attention(
    m: &mut Model,
    c: &AttnShape,
    w: &Mla,
    indexer: Option<&Indexer>,
    l: Layer,
    x: &Tensor,
) -> Tensor {
    let h = x.rms_norm(&w.norm);
    let q = h.matmul(&w.q_a).rms_norm(&w.q_norm).matmul(&w.q_b);
    let kv = h.matmul(&w.kv_a);
    let latent = kv.rms_norm(&w.kv_norm);
    let kind = AttnKind::Mla {
        latent: c.latent,
        rope: c.rope,
    };
    let cache = m.paged_kv(l, kind, Sync::Eager);
    let keys = indexer.map(|i| select_keys(m, c, i, &h, l));
    let q = q.rope(c.rope);
    m.attend(kind, Mask::Causal, &q, &latent, &cache, keys)
        .matmul(&w.o_proj)
}

/// Scores every cached key, keeps the best `index_keep`.
fn select_keys(m: &mut Model, c: &AttnShape, i: &Indexer, h: &Tensor, l: Layer) -> KeySet {
    let q = h.matmul(&i.wq);
    let k = m.index_keys(l, &h.matmul(&i.wk));
    let scores = q.dot_all(&k).act(Act::Silu).weighted(&h.matmul(&i.weights));
    m.top_k(&scores, c.index_keep).keys()
}
