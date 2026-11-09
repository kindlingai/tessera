//! Qwen3: grouped-query attention with RoPE, and SwiGLU MLPs.

use crate::shared::{Stack, rows, swiglu, tokens};
use trace::DimName::*;
use trace::{
    Composite, Dim, DimName, Embed, Graph, Linear, Mask, Model, Norm, Pairs, PathSeg, Role, Tensor,
    Weights,
};

pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Config {
    /// The 0.6B checkpoint's sizes.
    pub fn small(max_seqs: u64, max_context: u64) -> Self {
        let stack = Stack {
            vocab: 151_936,
            hidden: 1024,
            layers: 28,
            heads: 16,
            head_dim: 128,
            ffn: 3072,
            max_seqs,
            max_context,
        };
        Self {
            stack,
            kv_heads: 8,
            rope_theta: 1e6,
            eps: 1e-6,
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[(KvHeads, self.kv_heads)])
    }
}

#[derive(Weights)]
pub struct Layer {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    pub q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: Norm,
    #[shape(HeadDim)]
    pub k_norm: Norm,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
    #[shape(Hidden)]
    pub ln2: Norm,
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

/// The embedding table is also the output head.
#[derive(Weights)]
pub struct Qwen3 {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer>,
    #[shape(Hidden)]
    pub norm: Norm,
}

pub fn forward(m: &Model, c: &Config, w: &Qwen3, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let mut x = m.embed(tokens, &w.embed);
    for l in &w.layers {
        let h = x.rms_norm(&l.ln1, c.eps);
        let attend = || [attention(m, c, l, &h, positions)];
        let [attn] = m.composite(Composite::Attention, attend);
        x = x.add(&attn);
        let h = x.rms_norm(&l.ln2, c.eps);
        let [mlp] = m.composite(Composite::Mlp, || [swiglu(&h, [&l.gate, &l.up, &l.down])]);
        x = x.add(&mlp);
    }
    x.rms_norm(&w.norm, c.eps).unembed(&w.embed)
}

/// Queries and keys are normed per head, then rotated.
fn attention(m: &Model, c: &Config, l: &Layer, h: &Tensor, positions: &Tensor) -> Tensor {
    let split = |w: &Linear, heads: DimName| h.matmul(w).reshape([Seqs, Context, heads, HeadDim]);
    let turn = |t: Tensor, norm: &Norm| {
        let normed = t.rms_norm(norm, c.eps);
        normed.rope(positions, c.rope_theta, Pairs::Halves)
    };
    let q = turn(split(&l.q, Heads), &l.q_norm);
    let k = turn(split(&l.k, KvHeads), &l.k_norm);
    let v = split(&l.v, KvHeads);
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    let out = m.attend(Mask::Causal, scale, [&q, &k, &v], None);
    out.reshape(rows([Heads * HeadDim])).matmul(&l.o)
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Qwen3 = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// The released checkpoint's tensor name for a weight path.
pub fn tensor_name(path: &[PathSeg]) -> Option<String> {
    use PathSeg::{Field, Index};
    let name = match path {
        [Field(f)] if f == "embed" => "embed_tokens".into(),
        [Field(f)] if f == "norm" => "norm".into(),
        [Field(layers), Index(i), Field(f)] if layers == "layers" => {
            let part = match f.as_str() {
                "ln1" => "input_layernorm",
                "ln2" => "post_attention_layernorm",
                "q_norm" | "k_norm" => &format!("self_attn.{f}"),
                "q" | "k" | "v" | "o" => &format!("self_attn.{f}_proj"),
                "gate" | "up" | "down" => &format!("mlp.{f}_proj"),
                _ => return None,
            };
            format!("layers.{i}.{part}")
        }
        _ => return None,
    };
    Some(format!("model.{name}.weight"))
}
