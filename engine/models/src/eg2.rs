//! EmbeddingGemma 2: a bidirectional encoder with per-layer embeddings.

use crate::shared::{Stack, tokens};
use trace::DimName::*;
use trace::{
    Act, Composite, Dim, DimExpr, Embed, Graph, Linear, Mask, Model, Norm, Pairs, PathSeg, Role,
    Tensor, Vector, Weights,
};

pub struct Config {
    /// Its `head_dim` is a sliding layer's.
    pub stack: Stack,
    pub kv_heads: u64,
    /// Every n-th layer attends to the whole text.
    pub full_every: u64,
    /// A full layer's head width. It has one key head.
    pub full_head_dim: u64,
    pub per_layer: u64,
    pub embedding: u64,
    /// Keys a sliding layer sees either side of the query.
    pub window: u64,
    pub theta: f32,
    pub full_theta: f32,
    pub eps: f32,
}

impl Config {
    /// The released checkpoint's sizes.
    pub fn release(max_seqs: u64, max_context: u64) -> Self {
        let (vocab, hidden, layers) = (262_144, 512, 24);
        let (heads, head_dim, ffn) = (4, 256, 2048);
        let stack = Stack {
            vocab,
            hidden,
            layers,
            heads,
            head_dim,
            ffn,
            max_seqs,
            max_context,
        };
        Self {
            stack,
            kv_heads: 2,
            full_every: 6,
            full_head_dim: 512,
            per_layer: 512,
            embedding: 768,
            window: 512,
            theta: 1e4,
            full_theta: 1e6,
            eps: 1e-6,
        }
    }

    /// Whether layer `l` attends to the whole text.
    pub fn full(&self, l: u64) -> bool {
        (l + 1).is_multiple_of(self.full_every)
    }

    fn full_layers(&self) -> u64 {
        (0..self.stack.layers).filter(|&l| self.full(l)).count() as u64
    }

    pub fn dims(&self) -> Vec<Dim> {
        let full = self.full_layers();
        self.stack.dims(&[
            (KvHeads, self.kv_heads),
            (FullHeadDim, self.full_head_dim),
            (PerLayer, self.per_layer),
            (Embedding, self.embedding),
            (SlidingLayers, self.stack.layers - full),
            (FullLayers, full),
        ])
    }

    /// The layer the n-th sliding or full attention serves.
    pub fn layer_of(&self, full: bool, n: u64) -> u64 {
        let kind = |l: &u64| self.full(*l) == full;
        (0..self.stack.layers).filter(kind).nth(n as usize).unwrap()
    }
}

/// An attention's weights, as the layer kinds share them.
struct Parts<'a> {
    ln: &'a Norm,
    q: &'a Linear,
    k: &'a Linear,
    v: &'a Linear,
    q_norm: &'a Norm,
    k_norm: &'a Norm,
    v_norm: &'a Norm,
    o: &'a Linear,
    post: &'a Norm,
    head: DimExpr,
    kv_heads: DimExpr,
    theta: f32,
    mask: Mask,
}

macro_rules! attention {
    ($name:ident, $head:ident, $kv:expr) => {
        #[derive(Weights)]
        pub struct $name {
            #[shape(Hidden)]
            pub ln: Norm,
            #[shape(Hidden, Heads * $head)]
            pub q: Linear,
            #[shape(Hidden, $kv)]
            pub k: Linear,
            #[shape(Hidden, $kv)]
            pub v: Linear,
            #[shape($head)]
            pub q_norm: Norm,
            #[shape($head)]
            pub k_norm: Norm,
            /// All ones: the checkpoint has no weight.
            #[shape($head)]
            pub v_norm: Norm,
            #[shape(Heads * $head, Hidden)]
            pub o: Linear,
            #[shape(Hidden)]
            pub post: Norm,
        }

        impl $name {
            fn parts(&self, kv_heads: DimExpr, theta: f32, mask: Mask) -> Parts<'_> {
                Parts {
                    ln: &self.ln,
                    q: &self.q,
                    k: &self.k,
                    v: &self.v,
                    q_norm: &self.q_norm,
                    k_norm: &self.k_norm,
                    v_norm: &self.v_norm,
                    o: &self.o,
                    post: &self.post,
                    head: $head.into(),
                    kv_heads,
                    theta,
                    mask,
                }
            }
        }
    };
}

attention!(Sliding, HeadDim, KvHeads * HeadDim);
attention!(Full, FullHeadDim, FullHeadDim);

#[derive(Weights)]
pub struct Layer {
    #[shape(Hidden)]
    pub pre_ff: Norm,
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
    #[shape(Hidden)]
    pub post_ff: Norm,
    /// This layer's rows of the shared per-layer projection.
    #[shape(Hidden, PerLayer)]
    pub ple_in: Linear,
    #[shape(Hidden, PerLayer)]
    pub ple_gate: Linear,
    #[shape(PerLayer, Hidden)]
    pub ple_proj: Linear,
    #[shape(Hidden)]
    pub ple_norm: Norm,
    /// The checkpoint's one scalar, repeated across the row.
    #[shape(Hidden)]
    pub scalar: Vector,
}

#[derive(Weights)]
pub struct Eg2 {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(SlidingLayers)]
    pub sliding: Vec<Sliding>,
    #[shape(FullLayers)]
    pub full: Vec<Full>,
    #[shape(Layers)]
    pub layers: Vec<Layer>,
    #[shape(PerLayer)]
    pub ple_norm: Norm,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Embedding)]
    pub proj: Linear,
}

/// The residual stream after every layer, `[Seqs, Context, Hidden]`.
pub fn trunk(m: &Model, c: &Config, w: &Eg2, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let (eps, hidden) = (c.eps, c.stack.hidden as f32);
    let inputs = m.embed(tokens, &w.embed).scale(hidden.sqrt());
    let mut x = inputs.clone();
    let (mut sliding, mut full) = (w.sliding.iter(), w.full.iter());
    for (l, layer) in w.layers.iter().enumerate() {
        let ple = inputs.matmul(&layer.ple_in).scale(hidden.powf(-0.5));
        let ple = ple.rms_norm(&w.ple_norm, eps);
        let parts = match c.full(l as u64) {
            true => full
                .next()
                .unwrap()
                .parts(1u64.into(), c.full_theta, Mask::Bidirectional),
            false => {
                let kv = KvHeads.into();
                sliding
                    .next()
                    .unwrap()
                    .parts(kv, c.theta, Mask::Within(c.window))
            }
        };
        let attend = || [attention(m, c, &parts, &x, positions)];
        let [attn] = m.composite(Composite::Attention, attend);
        x = x.add(&attn);
        let h = x.rms_norm(&layer.pre_ff, eps);
        let [mlp] = m.composite(Composite::Mlp, || {
            [geglu(&h, layer).rms_norm(&layer.post_ff, eps)]
        });
        x = x.add(&mlp);
        let gated = x.matmul(&layer.ple_gate).act(Act::GeluTanh).mul(&ple);
        let o = gated.matmul(&layer.ple_proj).rms_norm(&layer.ple_norm, eps);
        x = x.add(&o).mul_vector(&layer.scalar);
    }
    x
}

/// Each token projected to the embedding width. Mean these.
pub fn forward(m: &Model, c: &Config, w: &Eg2, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let x = trunk(m, c, w, tokens, positions);
    x.rms_norm(&w.norm, c.eps).matmul(&w.proj)
}

fn geglu(h: &Tensor, l: &Layer) -> Tensor {
    let inner = h.matmul(&l.gate).act(Act::GeluTanh).mul(&h.matmul(&l.up));
    inner.matmul(&l.down)
}

/// Queries, keys and values are normed per head, scores unscaled.
fn attention(m: &Model, c: &Config, p: &Parts, x: &Tensor, positions: &Tensor) -> Tensor {
    let h = x.rms_norm(p.ln, c.eps);
    let split = |w: &Linear, heads: DimExpr| {
        let shaped = vec![Seqs.into(), Context.into(), heads, p.head.clone()];
        h.matmul(w).reshape(shaped)
    };
    let turn = |t: Tensor, norm: &Norm| {
        let normed = t.rms_norm(norm, c.eps);
        normed.rope(positions, p.theta, Pairs::Halves)
    };
    let q = turn(split(p.q, Heads.into()), p.q_norm);
    let k = turn(split(p.k, p.kv_heads.clone()), p.k_norm);
    let v = split(p.v, p.kv_heads.clone()).rms_norm(p.v_norm, c.eps);
    let out = m.attend(p.mask, 1.0, [&q, &k, &v], None);
    let flat = vec![Seqs.into(), Context.into(), Heads * p.head.clone()];
    out.reshape(flat).matmul(p.o).rms_norm(p.post, c.eps)
}

/// One forward entry: tokens and positions in, projected tokens out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Eg2 = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let out = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &out)])
}

/// The residual stream after the last layer, for parity checks.
pub fn probe(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Eg2 = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let out = trunk(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &out)])
}

/// The released checkpoint's tensor name for a weight path.
pub fn tensor_name(c: &Config, path: &[PathSeg]) -> Option<String> {
    use PathSeg::{Field, Index};
    let name = match path {
        [Field(f)] => match f.as_str() {
            "embed" => "embed_tokens.weight".into(),
            "norm" => "norm.weight".into(),
            "proj" => "embedding_projection.weight".into(),
            "ple_norm" => "ple.per_layer_projection_norm.weight".into(),
            _ => return None,
        },
        [Field(kind), Index(n), Field(f)] if kind == "sliding" || kind == "full" => {
            let part = match f.as_str() {
                "ln" => "input_layernorm",
                "post" => "post_attention_layernorm",
                "q_norm" | "k_norm" => &format!("self_attn.{f}"),
                "q" | "k" | "v" | "o" => &format!("self_attn.{f}_proj"),
                _ => return None,
            };
            let l = c.layer_of(kind == "full", u64::from(*n));
            format!("layers.{l}.{part}.weight")
        }
        [Field(layers), Index(i), Field(f)] if layers == "layers" => {
            let part = match f.as_str() {
                "pre_ff" => "pre_feedforward_layernorm.weight",
                "post_ff" => "post_feedforward_layernorm.weight",
                "gate" | "up" | "down" => &format!("mlp.{f}_proj.weight"),
                "ple_gate" => "ple_block.per_layer_input_gate.weight",
                "ple_proj" => "ple_block.per_layer_projection.weight",
                "ple_norm" => "ple_block.post_per_layer_input_norm.weight",
                "scalar" => "layer_scalar",
                _ => return None,
            };
            format!("layers.{i}.{part}")
        }
        _ => return None,
    };
    Some(format!("language_model.{name}"))
}

/// A weight's values from the checkpoint, through `read` by name.
pub fn weight(
    c: &Config,
    path: &[PathSeg],
    read: &dyn Fn(&str) -> Option<Vec<f32>>,
) -> Option<Vec<f32>> {
    use PathSeg::{Field, Index};
    match path {
        [Field(kind), _, Field(f)] if f == "v_norm" => {
            let width = if kind == "full" {
                c.full_head_dim
            } else {
                c.stack.head_dim
            };
            Some(vec![1.0; width as usize])
        }
        [Field(_), Index(i), Field(f)] if f == "ple_in" => {
            let all = read("language_model.ple.per_layer_model_projection.weight")?;
            let rows = (c.per_layer * c.stack.hidden) as usize;
            let at = *i as usize * rows;
            Some(all[at..at + rows].to_vec())
        }
        [Field(_), Index(_), Field(f)] if f == "scalar" => {
            let one = read(&tensor_name(c, path)?)?;
            Some(vec![one[0]; c.stack.hidden as usize])
        }
        _ => read(&tensor_name(c, path)?),
    }
}
