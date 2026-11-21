//! EmbeddingGemma 2: a bidirectional encoder with per-layer embeddings.

use crate::keys::Keys;
use crate::shared::{Attend, Dense, SIZES, Side, Stack, listed as listed_ports, tokens};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, Dim, DimExpr, Embed, Graph, Mask, Model, Norm, Pairs, Role, Scalar, Tensor,
    Weights, each,
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
        let sizes = [262_144, 512, 24, 4, 256, 2048];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
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

    /// The config `k` holds, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        k.fixed("text_config.model_type", json!("embedding_gemma2_text"));
        k.fixed("text_config.hidden_activation", json!("gelu_pytorch_tanh"));
        k.fixed("text_config.attention_bias", json!(false));
        for kind in ["sliding_attention", "full_attention"] {
            let path = format!("text_config.rope_parameters.{kind}.rope_type");
            k.fixed(&path, json!("default"));
        }
        let sizes = SIZES.map(|key| k.need(&[&format!("text_config.{key}")]));
        let layers = sizes[2];
        let (full_every, full_head_dim) = (full_layers(k, layers), full_heads(k, layers));
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: k.need(&["text_config.num_key_value_heads"]),
            full_every,
            full_head_dim,
            per_layer: k.need(&["text_config.hidden_size_per_layer_input"]),
            embedding: k.need(&["text_config.embedding_dim"]),
            window: k.need(&["text_config.sliding_window"]),
            theta: k.need(&[
                "text_config.rope_parameters.sliding_attention.rope_theta",
                "text_config.rope_local_base_freq",
            ]),
            full_theta: k.need(&[
                "text_config.rope_parameters.full_attention.rope_theta",
                "text_config.rope_theta",
            ]),
            eps: k.need(&["text_config.rms_norm_eps"]),
        }
    }

    /// Whether layer `l` attends to the whole text.
    pub fn full(&self, l: u64) -> bool {
        (l + 1).is_multiple_of(self.full_every)
    }

    /// Layer `l`'s head width and key head count.
    pub fn heads(&self, l: u64) -> [DimExpr; 2] {
        match self.full(l) {
            true => [FullHeadDim.into(), 1u64.into()],
            false => [HeadDim.into(), KvHeads.into()],
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[
            (KvHeads, self.kv_heads),
            (FullHeadDim, self.full_head_dim),
            (PerLayer, self.per_layer),
            (Embedding, self.embedding),
        ])
    }
}

/// One layer's attention: `S0` head width, `S1` key heads.
#[derive(Weights)]
pub struct Attention<D: Dense> {
    #[shape(Hidden)]
    pub ln: Norm,
    #[shape(Hidden, Heads * S0)]
    pub q: D,
    #[shape(Hidden, S1 * S0)]
    pub k: D,
    #[shape(Hidden, S1 * S0)]
    pub v: D,
    #[shape(S0)]
    pub q_norm: Norm,
    #[shape(S0)]
    pub k_norm: Norm,
    #[shape(Heads * S0, Hidden)]
    pub o: D,
    #[shape(Hidden)]
    pub post: Norm,
}

#[derive(Weights)]
pub struct Layer<D: Dense> {
    #[shape(Hidden)]
    pub pre_ff: Norm,
    #[shape(Hidden, Ffn)]
    pub gate: D,
    #[shape(Hidden, Ffn)]
    pub up: D,
    #[shape(Ffn, Hidden)]
    pub down: D,
    #[shape(Hidden)]
    pub post_ff: Norm,
    /// This layer's rows of the shared per-layer projection.
    #[shape(Hidden, PerLayer)]
    pub ple_in: D,
    #[shape(Hidden, PerLayer)]
    pub ple_gate: D,
    #[shape(PerLayer, Hidden)]
    pub ple_proj: D,
    #[shape(Hidden)]
    pub ple_norm: Norm,
    #[shape()]
    pub scalar: Scalar,
}

/// The weights outside attention. `D` is how matrices are stored.
#[derive(Weights)]
pub struct Eg2<D: Dense> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer<D>>,
    #[shape(PerLayer)]
    pub ple_norm: Norm,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Embedding)]
    pub proj: D,
}

/// Each layer's attention, shaped for its kind.
pub fn attention<D: Dense>(m: &Model, c: &Config) -> Vec<Attention<D>> {
    each(m, "attention", Layers, |l| c.heads(u64::from(l)).to_vec())
}

/// The residual stream after every layer, `[Seqs, Context, Hidden]`.
fn trunk<D: Dense>(
    m: &Model,
    c: &Config,
    w: &Eg2<D>,
    attention: &[Attention<D>],
    [tokens, positions]: [&Tensor; 2],
) -> Tensor {
    listed(m, c, w, attention, [tokens, positions], None)
}

/// The same, each query reading only `keys` when given.
fn listed<D: Dense>(
    m: &Model,
    c: &Config,
    w: &Eg2<D>,
    attention: &[Attention<D>],
    [tokens, positions]: [&Tensor; 2],
    keys: Option<&Tensor>,
) -> Tensor {
    let (eps, hidden) = (c.eps, c.stack.hidden as f32);
    let inputs = m.embed(tokens, &w.embed).scale(hidden.sqrt());
    let mut x = inputs.clone();
    for (l, layer) in w.layers.iter().enumerate() {
        let ple = context(layer, &inputs, &w.ple_norm, [hidden, eps]);
        let attend = || {
            [attend(
                m,
                c,
                (l as u64, &attention[l]),
                &x,
                [positions],
                keys,
            )]
        };
        let [attn] = m.composite(Composite::Attention, attend);
        x = after(m, eps, layer, x.add(&attn), &ple);
    }
    x
}

/// The layer's per-layer input from the scaled embeddings, normed.
pub(crate) fn context<D: Dense>(
    layer: &Layer<D>,
    inputs: &Tensor,
    norm: &Norm,
    [hidden, eps]: [f32; 2],
) -> Tensor {
    let ple = layer.ple_in.apply(inputs).scale(hidden.powf(-0.5));
    ple.rms_norm(norm, eps)
}

/// The MLP, then the per-layer input, then the layer's scalar.
pub(crate) fn after<D: Dense>(
    m: &Model,
    eps: f32,
    layer: &Layer<D>,
    x: Tensor,
    ple: &Tensor,
) -> Tensor {
    let h = x.rms_norm(&layer.pre_ff, eps);
    let [mlp] = m.composite(Composite::Mlp, || {
        [geglu(&h, layer).rms_norm(&layer.post_ff, eps)]
    });
    let x = x.add(&mlp);
    let gated = layer.ple_gate.apply(&x).act(Act::GeluTanh).mul(ple);
    let o = layer.ple_proj.apply(&gated).rms_norm(&layer.ple_norm, eps);
    x.add(&o).mul_scalar(&layer.scalar)
}

/// One embedding per text, `[Seqs, Embedding]`. Callers normalize.
pub fn forward<D: Dense>(
    m: &Model,
    c: &Config,
    w: &Eg2<D>,
    attention: &[Attention<D>],
    ports: [&Tensor; 2],
) -> Tensor {
    let x = trunk(m, c, w, attention, ports);
    w.proj.apply(&x.rms_norm(&w.norm, c.eps)).mean(1)
}

fn geglu<D: Dense>(h: &Tensor, l: &Layer<D>) -> Tensor {
    let inner = l.gate.apply(h).act(Act::GeluTanh).mul(&l.up.apply(h));
    l.down.apply(&inner)
}

/// Queries, keys and values are normed per head, scores unscaled.
fn attend<D: Dense>(
    m: &Model,
    c: &Config,
    (l, a): (u64, &Attention<D>),
    x: &Tensor,
    [positions]: [&Tensor; 1],
    keys: Option<&Tensor>,
) -> Tensor {
    let [head, kv_heads] = c.heads(l);
    let (theta, mask) = match c.full(l) {
        true => (c.full_theta, Mask::Bidirectional),
        false => (c.theta, Mask::Within(c.window)),
    };
    let h = x.rms_norm(&a.ln, c.eps);
    let lead = vec![Seqs.into(), Context.into()];
    let heads = Attend::new(&a.o, lead, [Heads.into(), kv_heads, head]);
    let heads = heads.rotary(positions, theta, Pairs::Halves);
    let turn = |t: Tensor, side: Side| match side.of([&a.q_norm, &a.k_norm]) {
        Some(norm) => t.rms_norm(norm, c.eps),
        None => t.rms(c.eps),
    };
    let out = heads.apply([&a.q, &a.k, &a.v], &h, turn, |qkv| {
        m.attend(mask, 1.0, qkv, keys)
    });
    out.rms_norm(&a.post, c.eps)
}

/// The graph of one stage, from tokens and positions.
fn staged<D: Dense>(c: &Config, stage: Stage<D>) -> Graph {
    let m = Model::new(c.dims());
    let w: Eg2<D> = tessera_trace::declare(&m);
    let attention = attention(&m, c);
    let [tokens, positions] = tokens(&m);
    let out = stage(&m, c, &w, &attention, [&tokens, &positions]);
    m.finish([(Role::Logits, &out)])
}

type Stage<D> = fn(&Model, &Config, &Eg2<D>, &[Attention<D>], [&Tensor; 2]) -> Tensor;

/// One forward entry: tokens and positions in, embeddings out.
pub fn program<D: Dense>(c: &Config) -> Graph {
    staged(c, forward::<D>)
}

/// Padded texts reading listed `Keys`: a row per token.
pub fn batched<D: Dense>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Eg2<D> = tessera_trace::declare(&m);
    let attention = attention(&m, c);
    let [tokens, positions, keys] = listed_ports(&m);
    let x = listed(&m, c, &w, &attention, [&tokens, &positions], Some(&keys));
    let out = w.proj.apply(&x.rms_norm(&w.norm, c.eps));
    m.finish([(Role::Logits, &out)])
}

/// The residual stream after the last layer, for parity checks.
pub fn probe<D: Dense>(c: &Config) -> Graph {
    staged(c, trunk::<D>)
}

const TYPES: &str = "text_config.layer_types";

/// The full layers' spacing. The graph takes one every n.
pub(crate) fn full_layers(k: &mut Keys, layers: u64) -> u64 {
    let types: Vec<String> = k.need(&[TYPES]);
    let full: Vec<bool> = types.iter().map(|t| t == "full_attention").collect();
    let every = full.iter().position(|&f| f).map_or(0, |p| p as u64 + 1);
    let spaced =
        (0..layers).all(|l| full.get(l as usize) == Some(&(every > 0 && (l + 1) % every == 0)));
    let known = types
        .iter()
        .all(|t| t == "full_attention" || t == "sliding_attention");
    if !(spaced && known && types.len() as u64 == layers) {
        k.misfit(TYPES, "the graph takes one full layer every n");
    }
    every
}

/// One layer's own sizes in `per_layer_config`.
#[derive(Default, Deserialize, PartialEq)]
struct Own {
    head_dim: u64,
    num_key_value_heads: u64,
}

/// A full layer's head width. Each has one key head.
fn full_heads(k: &mut Keys, layers: u64) -> u64 {
    const OWN: &str = "text_config.per_layer_config";
    let own: BTreeMap<String, Own> = k.need(&[OWN]);
    let types: Vec<String> = k.or(&[TYPES], Vec::new());
    let width = own.values().next().map_or(0, |o| o.head_dim);
    for l in 0..layers as usize {
        let entry = own.get(&format!("{l:02}"));
        let full = types.get(l).is_some_and(|t| t == "full_attention");
        let fits = match entry {
            Some(o) => full && o.head_dim == width && o.num_key_value_heads == 1,
            None => !full,
        };
        if !fits {
            k.misfit(OWN, format!("layer {l} does not fit the graph"));
        }
    }
    width
}
