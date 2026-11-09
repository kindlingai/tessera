//! Qwen 3.8 Flash Next: Gated DeltaNet beside gated attention. Sketch.

use crate::core::*;
use crate::library::{Moe, moe};

pub struct Config {
    pub layers: usize,
    /// Every n-th layer uses attention; the rest use DeltaNet.
    pub attn_every: usize,
    pub top_k: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// RoPE covers only part of each head.
    pub rope_dims: usize,
    pub conv: usize,
}

#[derive(Weights)]
pub struct DeltaNet {
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

#[derive(Weights)]
pub struct GatedAttn {
    pub norm: Norm,
    #[split(heads)]
    pub q: Linear,
    #[split(heads)]
    pub kv: Linear,
    pub q_norm: Norm,
    pub k_norm: Norm,
    #[split(heads)]
    pub gate: Linear,
    #[split(rows)]
    pub o: Linear,
}

pub enum Mixer {
    Attn(GatedAttn),
    Delta(DeltaNet),
}

#[derive(Weights)]
pub struct Block {
    pub mixer: Mixer,
    pub mlp_norm: Norm,
    pub moe: Moe<SwiGlu>,
}

#[derive(Weights)]
pub struct Qwen {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    #[split(vocab)]
    pub head: Linear,
}

pub fn forward(m: &mut Model, c: &Config, w: &Qwen, tokens: &Tensor) -> Tensor {
    let mut x = m.embed(tokens, &w.embed).reduce();
    for (i, b) in w.blocks.iter().enumerate() {
        let l = Layer(i);
        let a = match &b.mixer {
            Mixer::Attn(g) => gated_attention(m, c, g, l, &x),
            Mixer::Delta(d) => deltanet(m, c, d, l, &x),
        };
        x = x.add(&m.site(Site::Writer(l, Sublayer::Attn), a.reduce()));
        let f = moe(m, &b.moe, &x.rms_norm_zc(&b.mlp_norm), c.top_k, l);
        x = x.add(&m.site(Site::Writer(l, Sublayer::Mlp), f.reduce()));
        x = m.site(Site::Residual(l), x);
    }
    let h = m.site(Site::PreHead, x.rms_norm_zc(&w.norm));
    h.matmul(&w.head)
}

/// GQA with partial RoPE and a sigmoid output gate.
fn gated_attention(m: &mut Model, c: &Config, w: &GatedAttn, l: Layer, x: &Tensor) -> Tensor {
    let h = x.rms_norm_zc(&w.norm);
    let q = h.matmul(&w.q).rms_norm(&w.q_norm).rope(c.rope_dims);
    let kv = h.matmul(&w.kv).norm_keys(&w.k_norm).rope_keys(c.rope_dims);
    let kind = AttnKind::Gqa {
        kv_heads: c.kv_heads,
    };
    let cache = m.paged_kv(l, kind, Sync::Eager);
    m.attend(kind, Mask::Causal, &q, &kv, &cache, None)
        .gate(Act::Sigmoid, &h.matmul(&w.gate))
        .matmul(&w.o)
}

fn deltanet(m: &mut Model, c: &Config, w: &DeltaNet, l: Layer, x: &Tensor) -> Tensor {
    let h = x.rms_norm_zc(&w.norm);
    let history = m.conv_history(l, c.conv);
    let qkv = m.recur(
        RecurRule::Conv { width: c.conv },
        &h.matmul(&w.qkv),
        &history,
    );
    let state = m.recur_state(l, RecurRule::GatedDelta, c.heads, c.head_dim, c.head_dim);
    let o = m
        .recur(RecurRule::GatedDelta, &qkv, &state)
        .with_gates(&h.matmul(&w.gates));
    o.rms_norm(&w.out_norm)
        .gate(Act::Silu, &h.matmul(&w.out_gate))
        .matmul(&w.out)
}
