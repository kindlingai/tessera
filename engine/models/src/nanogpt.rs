//! A tiny GPT: learned positions, GELU MLPs, causal attention.

use crate::shared::{Attend, Stack};
use tessera_trace::DimName::*;
use tessera_trace::tokens;
use tessera_trace::{
    Act, Composite, Dim, Embed, Graph, Linear, Mask, Model, Norm, Role, Tensor, Weights,
};

pub struct Config {
    /// Its context bound is also the position table's length.
    pub stack: Stack,
    pub eps: f32,
}

impl Config {
    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[(Positions, self.stack.max_context)])
    }
}

#[derive(Weights)]
pub struct Layer {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Hidden)]
    pub wq: Linear,
    #[shape(Hidden, Hidden)]
    pub wk: Linear,
    #[shape(Hidden, Hidden)]
    pub wv: Linear,
    #[shape(Hidden, Hidden)]
    pub wproj: Linear,
    #[shape(Hidden)]
    pub ln2: Norm,
    #[shape(Hidden, Ffn)]
    pub w1: Linear,
    #[shape(Ffn, Hidden)]
    pub w2: Linear,
}

#[derive(Weights)]
pub struct Gpt {
    #[shape(Vocab, Hidden)]
    pub tok_emb: Embed,
    #[shape(Positions, Hidden)]
    pub pos_emb: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer>,
    #[shape(Hidden)]
    pub ln_f: Norm,
    #[shape(Hidden, Vocab)]
    pub lm_head: Linear,
}

pub fn forward(m: &Model, c: &Config, w: &Gpt, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let mut x = m
        .embed(tokens, &w.tok_emb)
        .add(&m.embed(positions, &w.pos_emb));
    for l in &w.layers {
        let h = x.rms_norm(&l.ln1, c.eps);
        let [attn] = m.composite(Composite::Attention, || [attention(m, c, l, &h)]);
        x = x.add(&attn);
        let h = x.rms_norm(&l.ln2, c.eps);
        let [mlp] = m.composite(Composite::Mlp, || {
            [h.matmul(&l.w1).act(Act::GeluTanh).matmul(&l.w2)]
        });
        x = x.add(&mlp);
    }
    x.rms_norm(&w.ln_f, c.eps).matmul(&w.lm_head)
}

/// Causal multi-head attention and its output projection.
fn attention(m: &Model, c: &Config, l: &Layer, h: &Tensor) -> Tensor {
    let lead = vec![Seqs.into(), Context.into()];
    let shape = [Heads.into(), Heads.into(), HeadDim.into()];
    let heads = Attend::new(&l.wproj, lead, shape);
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    heads.apply(
        [&l.wq, &l.wk, &l.wv],
        h,
        |t, _| t,
        |qkv| m.attend(Mask::Causal, scale, qkv, None),
    )
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Gpt = tessera_trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}
