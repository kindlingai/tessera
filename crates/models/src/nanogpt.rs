//! The tiny GPT from `crates/nanogpt`, as a description.

use trace::DimName::*;
use trace::{
    Act, Attend, Composite, DType, Dim, Embed, EntryKind, Graph, Linear, Mask, Model, Norm, Port,
    Role, Shape, Size, Tensor, TensorType, Weights,
};

pub struct Config {
    pub vocab: u64,
    /// Longest context, and the position table's length.
    pub block: u64,
    pub layers: u64,
    pub heads: u64,
    pub hidden: u64,
    pub eps: f32,
    /// Most sequences one launch may carry.
    pub max_seqs: u64,
}

impl Config {
    pub fn dims(&self) -> Vec<Dim> {
        let fixed = |name, n| Dim {
            name,
            size: Size::Fixed(n),
        };
        let bounded = |name, n| Dim {
            name,
            size: Size::Bounded(n),
        };
        vec![
            bounded(Seqs, self.max_seqs),
            bounded(Context, self.block),
            fixed(Positions, self.block),
            fixed(Vocab, self.vocab),
            fixed(Hidden, self.hidden),
            fixed(Heads, self.heads),
            fixed(HeadDim, self.hidden / self.heads),
            fixed(Ffn, 4 * self.hidden),
            fixed(Layers, self.layers),
        ]
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
    let scale = 1.0 / ((c.hidden / c.heads) as f32).sqrt();
    let mask = Mask::Causal;
    let out = m.attend(
        Attend { mask, scale },
        &project(&l.wq),
        &project(&l.wk),
        &project(&l.wv),
    );
    out.reshape([Seqs, Context, Hidden]).matmul(&l.wproj)
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Gpt = trace::declare(&m);
    let ids = TensorType::new([Seqs, Context], DType::U32);
    let [tokens, positions] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Tokens,
                ty: ids.clone(),
            },
            Port {
                role: Role::Positions,
                ty: ids,
            },
        ],
    );
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}
