//! A tiny GPT: learned positions, GELU MLPs, causal attention.

use crate::shared::{Stack, tokens};
use trace::DimName::*;
use trace::{
    Act, Composite, Dim, Embed, Graph, Linear, Mask, Model, Norm, Role, Shape, Tensor, Weights,
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
    let heads = Shape::from([Seqs, Context, Heads, HeadDim]);
    let project = |w: &Linear| h.matmul(w).reshape(heads.clone());
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    let qkv = [&l.wq, &l.wk, &l.wv].map(project);
    let [q, k, v] = &qkv;
    let out = m.attend(Mask::Causal, scale, [q, k, v], None);
    out.reshape([Seqs, Context, Hidden]).matmul(&l.wproj)
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Gpt = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}
