//! GLM-5.3-Flash: KDA linear attention beside full attention. Sketch.

use crate::core::*;
use crate::glm53::{AttnShape, Indexer, Mla, attention};
use crate::library::{Moe, moe};

pub struct Config {
    pub layers: usize,
    /// Which layers use full attention; the rest use KDA.
    pub full: LayerSet,
    pub top_k: usize,
    pub attn: AttnShape,
    pub heads: usize,
    pub head_dim: usize,
    pub conv: usize,
}

/// Kimi delta attention: a gated delta rule per head.
#[derive(Weights)]
pub struct Kda {
    pub norm: Norm,
    #[split(heads)]
    pub qkv: Linear,
    #[split(heads)]
    pub conv: Linear,
    #[split(heads)]
    pub gates: Linear,
    #[split(heads)]
    pub out_gate: Linear,
    pub out_norm: Norm,
    #[split(rows)]
    pub out: Linear,
}

pub enum Mixer {
    Full(Mla, Indexer),
    Kda(Kda),
}

#[derive(Weights)]
pub struct Block {
    pub mixer: Mixer,
    pub mlp_norm: Norm,
    pub moe: Moe<SwiGlu>,
}

#[derive(Weights)]
pub struct Flash {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    #[split(vocab)]
    pub head: Linear,
}

pub fn forward(m: &mut Model, c: &Config, w: &Flash, tokens: &Tensor) -> Tensor {
    let mut x = m.embed(tokens, &w.embed).reduce();
    for (i, b) in w.blocks.iter().enumerate() {
        let l = Layer(i);
        let a = match &b.mixer {
            Mixer::Full(f, ix) => attention(m, &c.attn, f, Some(ix), l, &x),
            Mixer::Kda(k) => kda(m, c, k, l, &x),
        };
        x = x.add(&m.site(Site::Writer(l, Sublayer::Attn), a.reduce()));
        let f = moe(m, &b.moe, &x.rms_norm(&b.mlp_norm), c.top_k, l);
        x = x.add(&m.site(Site::Writer(l, Sublayer::Mlp), f.reduce()));
        x = m.site(Site::Residual(l), x);
    }
    let h = m.site(Site::PreHead, x.rms_norm(&w.norm));
    h.matmul(&w.head)
}

/// Short conv, then the delta-rule recurrence over per-head state.
fn kda(m: &mut Model, c: &Config, w: &Kda, l: Layer, x: &Tensor) -> Tensor {
    let h = x.rms_norm(&w.norm);
    let history = m.conv_history(l, c.conv);
    let qkv = m.recur(
        RecurRule::Conv { width: c.conv },
        &h.matmul(&w.qkv),
        &history,
    );
    let state = m.recur_state(l, RecurRule::Kda, c.heads, c.head_dim, c.head_dim);
    let o = m
        .recur(RecurRule::Kda, &qkv, &state)
        .with_gates(&h.matmul(&w.gates));
    o.rms_norm(&w.out_norm)
        .gate(Act::Sigmoid, &h.matmul(&w.out_gate))
        .matmul(&w.out)
}
