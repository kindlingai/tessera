//! CLIP's text tower: learned positions, causal pre-norm blocks.

use crate::keys::Keys;
use crate::shared::{Attend, Dense, Stack};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::tokens;
use tessera_trace::{
    Act, Dim, Embed, Graph, Linear, Mask, Model, Norm, Role, Tensor, Vector, Weights,
};

pub struct Config {
    /// Its context bound is also the position table's length.
    pub stack: Stack,
    pub eps: f32,
}

impl Config {
    /// The text tower `k` describes, cut to `context` tokens.
    pub fn read(k: &mut Keys, max_seqs: u64, context: u64) -> Self {
        let mut text = |key: &str| -> u64 { k.need(&[&format!("text_config.{key}"), key]) };
        let [vocab, hidden, layers, heads, ffn, positions] = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
        ]
        .map(&mut text);
        if heads == 0 || !hidden.is_multiple_of(heads) {
            k.misfit(
                "text_config.num_attention_heads",
                "heads must split hidden_size",
            );
        }
        if context > positions {
            let why = format!("{context} tokens exceed the {positions} positions");
            k.misfit("text_config.max_position_embeddings", why);
        }
        for path in ["text_config.hidden_act", "hidden_act"] {
            k.fixed(path, json!("quick_gelu"));
        }
        let head_dim = hidden.checked_div(heads).unwrap_or(0);
        let sizes = [vocab, hidden, layers, heads, head_dim, ffn];
        Self {
            stack: Stack::sized(sizes, max_seqs, context),
            eps: k.need(&["text_config.layer_norm_eps", "layer_norm_eps"]),
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[(Positions, self.stack.max_context)])
    }
}

/// A layer norm's gain and bias.
#[derive(Weights)]
pub struct LayerNorm {
    #[shape(S0)]
    pub weight: Norm,
    #[shape(S0)]
    pub bias: Vector,
}

/// A matrix and its bias.
#[derive(Weights)]
pub struct Affine {
    #[shape(S0, S1)]
    pub weight: Linear,
    #[shape(S1)]
    pub bias: Vector,
}

impl Dense for Affine {
    fn apply(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add_vector(&self.bias)
    }
}

#[derive(Weights)]
pub struct Layer {
    #[shape(Hidden)]
    pub ln1: LayerNorm,
    #[shape(Hidden, Hidden)]
    pub q: Affine,
    #[shape(Hidden, Hidden)]
    pub k: Affine,
    #[shape(Hidden, Hidden)]
    pub v: Affine,
    #[shape(Hidden, Hidden)]
    pub o: Affine,
    #[shape(Hidden)]
    pub ln2: LayerNorm,
    #[shape(Hidden, Ffn)]
    pub fc1: Affine,
    #[shape(Ffn, Hidden)]
    pub fc2: Affine,
}

#[derive(Weights)]
pub struct Clip {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Positions, Hidden)]
    pub pos: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer>,
    #[shape(Hidden)]
    pub norm: LayerNorm,
}

fn layer_norm(x: &Tensor, n: &LayerNorm, eps: f32) -> Tensor {
    x.center().rms_norm(&n.weight, eps).add_vector(&n.bias)
}

/// `x * sigmoid(1.702 x)`.
fn quick_gelu(x: &Tensor) -> Tensor {
    x.mul(&x.scale(1.702).act(Act::Sigmoid))
}

fn attention(m: &Model, c: &Config, l: &Layer, h: &Tensor) -> Tensor {
    let lead = vec![Seqs.into(), Context.into()];
    let heads = Attend::new(&l.o, lead, [Heads.into(), Heads.into(), HeadDim.into()]);
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    heads.apply(
        [&l.q, &l.k, &l.v],
        h,
        |t, _| t,
        |qkv| m.attend(Mask::Causal, scale, qkv, None),
    )
}

/// The final normed states, `[Seqs, Context, Hidden]`.
pub fn forward(m: &Model, c: &Config, w: &Clip, [tokens, positions]: [&Tensor; 2]) -> Tensor {
    let mut x = m.embed(tokens, &w.embed).add(&m.embed(positions, &w.pos));
    for l in &w.layers {
        let h = layer_norm(&x, &l.ln1, c.eps);
        x = x.add(&attention(m, c, l, &h));
        let h = layer_norm(&x, &l.ln2, c.eps);
        x = x.add(&l.fc2.apply(&quick_gelu(&l.fc1.apply(&h))));
    }
    layer_norm(&x, &w.norm, c.eps)
}

/// One forward entry: tokens and positions in, states out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Clip = tessera_trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let states = forward(&m, c, &w, [&tokens, &positions]);
    m.finish([(Role::Logits, &states)])
}
