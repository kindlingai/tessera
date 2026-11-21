//! Qwen3.5: gated delta rule layers and gated full attention.

use crate::decoder::{Mlp, Swiglu};
use crate::keys::Keys;
use crate::shared::{Attend, Stack, part, tokens};
use serde::de::DeserializeOwned;
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, Dim, DimExpr, Embed, Graph, Linear, Mask, Model, Norm, Pairs, PathSeg, Role,
    Taps, Tensor, Vector, Weights,
};

pub struct Config {
    /// `heads` and `head_dim` are the full-attention layers'.
    pub stack: Stack,
    pub kv_heads: u64,
    /// The leading width of each full-attention head that rotates.
    pub rotary: u64,
    pub rope_theta: f32,
    pub eps: f32,
    pub linear: Recurrent,
    /// Each layer's token mixer, in order.
    pub layers: Vec<Mixing>,
    /// The output head is the embedding table.
    pub tied: bool,
}

/// How a layer mixes its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixing {
    /// The gated delta rule.
    Linear,
    Full,
}

/// A linear-attention layer's sizes.
#[derive(Clone, Copy, Debug)]
pub struct Recurrent {
    pub key_heads: u64,
    pub value_heads: u64,
    pub key_dim: u64,
    pub value_dim: u64,
    /// Rows the causal filter reads.
    pub window: u64,
}

impl Config {
    /// The model `k` describes, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let fixed = [
            ("hidden_act", json!("silu")),
            ("attention_bias", json!(false)),
            ("attn_output_gate", json!(true)),
            ("output_gate_type", json!("swish")),
            ("mtp_num_hidden_layers", json!(0)),
            ("rope_parameters.rope_type", json!("default")),
        ];
        for (key, want) in fixed {
            k.fixed(&format!("text_config.{key}"), want.clone());
            k.fixed(key, want);
        }
        let sizes = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "intermediate_size",
        ]
        .map(|key| text::<u64>(k, key));
        let layers = mixing(k, sizes[2]);
        let factor: f32 = text(k, "rope_parameters.partial_rotary_factor");
        let linear = Recurrent {
            key_heads: text(k, "linear_num_key_heads"),
            value_heads: text(k, "linear_num_value_heads"),
            key_dim: text(k, "linear_key_head_dim"),
            value_dim: text(k, "linear_value_head_dim"),
            window: text(k, "linear_conv_kernel_dim"),
        };
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: text(k, "num_key_value_heads"),
            rotary: (sizes[4] as f32 * factor) as u64,
            rope_theta: text(k, "rope_parameters.rope_theta"),
            eps: text(k, "rms_norm_eps"),
            linear,
            layers,
            tied: k.or(
                &["text_config.tie_word_embeddings", "tie_word_embeddings"],
                false,
            ),
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        let (width, l) = (self.stack.head_dim, &self.linear);
        assert!(self.rotary <= width && self.rotary % 2 == 0, "rotary width");
        let mut own = vec![
            (KvHeads, self.kv_heads),
            (KeyHeads, l.key_heads),
            (ValueHeads, l.value_heads),
            (KeyDim, l.key_dim),
            (ValueDim, l.value_dim),
            (Window, l.window),
        ];
        if self.partial() {
            own.extend([(Rope, self.rotary), (Nope, width - self.rotary)]);
        }
        self.stack.dims(&own)
    }

    /// Whether only part of each full-attention head rotates.
    fn partial(&self) -> bool {
        self.rotary < self.stack.head_dim
    }
}

/// Rows over their root mean square, times one plus `gain`.
fn norm(x: &Tensor, gain: &Vector, eps: f32) -> Tensor {
    let r = x.rms(eps);
    r.add(&r.mul_vector(gain))
}

/// Gated causal attention.
#[derive(Weights)]
pub struct Full {
    /// Each head's query, then its gate.
    #[shape(Hidden, Heads * HeadDim * 2)]
    pub q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: Vector,
    #[shape(HeadDim)]
    pub k_norm: Vector,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
}

impl Full {
    /// Normed rows `h` `[Seqs, Context, Hidden]`, at `positions`.
    pub fn apply(&self, m: &Model, c: &Config, h: &Tensor, positions: &Tensor) -> Tensor {
        let lead = vec![Seqs.into(), Context.into()];
        let shape = [Heads.into(), KvHeads.into(), HeadDim.into()];
        let mut heads =
            Attend::new(&self.o, lead, shape).rotary(positions, c.rope_theta, Pairs::Halves);
        if c.partial() {
            heads = heads.partial([Rope.into(), Nope.into()]);
        }
        let both = heads.split(h.matmul(&self.q), Heads, HeadDim * 2u64);
        let q = norm(&both.slice(3, 0u64, HeadDim), &self.q_norm, c.eps);
        let gate = both.slice(3, HeadDim, HeadDim).act(Act::Sigmoid);
        let k = norm(
            &heads.split(h.matmul(&self.k), KvHeads, HeadDim),
            &self.k_norm,
            c.eps,
        );
        let v = heads.split(h.matmul(&self.v), KvHeads, HeadDim);
        let (q, k) = (heads.rotate(q), heads.rotate(k));
        let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
        let out = m.attend(Mask::Causal, scale, [&q, &k, &v], None);
        heads.merge(out.mul(&gate))
    }
}

/// The gated delta rule over a causal filter.
#[derive(Weights)]
pub struct DeltaNet {
    /// Queries, keys, then values, all through the filter.
    #[shape(Hidden, KeyHeads * KeyDim * 2 + ValueHeads * ValueDim)]
    pub qkv: Linear,
    /// The output gate.
    #[shape(Hidden, ValueHeads * ValueDim)]
    pub z: Linear,
    /// Write rates, before the sigmoid.
    #[shape(Hidden, ValueHeads)]
    pub b: Linear,
    /// Decays, before the bias and softplus.
    #[shape(Hidden, ValueHeads)]
    pub a: Linear,
    #[shape(KeyHeads * KeyDim * 2 + ValueHeads * ValueDim, Window)]
    pub conv: Taps,
    #[shape(ValueHeads)]
    pub dt_bias: Vector,
    /// The log of each value head's decay speed.
    #[shape(ValueHeads)]
    pub a_log: Vector,
    #[shape(ValueDim)]
    pub norm: Norm,
    #[shape(ValueHeads * ValueDim, Hidden)]
    pub out: Linear,
}

impl DeltaNet {
    /// Normed rows `h` `[Seqs, Context, Hidden]`.
    pub fn apply(&self, m: &Model, c: &Config, h: &Tensor) -> Tensor {
        let rows = |a: DimExpr, b: DimExpr| vec![Seqs.into(), Context.into(), a, b];
        let keys = KeyHeads * KeyDim;
        let values = ValueHeads * ValueDim;
        let mixed = h.matmul(&self.qkv).conv(&self.conv).act(Act::Silu);
        let split = |start: DimExpr, len: DimExpr, [heads, width]: [DimExpr; 2]| {
            mixed.slice(2, start, len).reshape(rows(heads, width))
        };
        let key = [KeyHeads.into(), KeyDim.into()];
        let (q, k) = (
            split(0u64.into(), keys.clone(), key.clone()),
            split(keys.clone(), keys.clone(), key),
        );
        let v = split(
            keys * 2u64,
            values.clone(),
            [ValueHeads.into(), ValueDim.into()],
        );
        // Unit rows: rms over the width, eps per element.
        let width = c.linear.key_dim as f32;
        let (q, k) = (
            q.rms(L2_EPS / width),
            k.rms(L2_EPS / width).scale(width.sqrt().recip()),
        );
        let decay = h
            .matmul(&self.a)
            .add_vector(&self.dt_bias)
            .act(Act::Softplus);
        let decay = decay.mul(&self.a_log.stored().act(Act::Exp)).scale(-1.0);
        let rate = h.matmul(&self.b).act(Act::Sigmoid);
        // The query's unit scale and the rule's root fold together.
        let out = m.delta(width.recip(), [&q, &k, &v], [&decay, &rate]);
        let z = h
            .matmul(&self.z)
            .reshape(rows(ValueHeads.into(), ValueDim.into()));
        let gated = out.rms_norm(&self.norm, c.eps).mul(&z.act(Act::Silu));
        gated
            .reshape(vec![Seqs.into(), Context.into(), values])
            .matmul(&self.out)
    }
}

/// The epsilon under the unit norm of queries and keys.
const L2_EPS: f32 = 1e-6;

/// `key` under `text_config`, else at the top.
fn text<T: DeserializeOwned + Default>(k: &mut Keys, key: &str) -> T {
    k.need(&[&format!("text_config.{key}"), key])
}

/// Each layer's mixer, from `layer_types`.
fn mixing(k: &mut Keys, layers: u64) -> Vec<Mixing> {
    let types: Vec<String> = text(k, "layer_types");
    let mut out = Vec::new();
    for t in &types {
        match t.as_str() {
            "linear_attention" => out.push(Mixing::Linear),
            "full_attention" => out.push(Mixing::Full),
            _ => k.misfit("layer_types", format!("no layer type {t}")),
        }
    }
    if out.len() as u64 != layers {
        k.misfit("layer_types", "one type per layer");
    }
    out
}

/// Weights every layer holds, whatever its mixer.
#[derive(Weights)]
pub struct Common {
    #[shape(Hidden)]
    pub ln1: Vector,
    #[shape(Hidden)]
    pub ln2: Vector,
    pub mlp: Swiglu,
}

/// A layer's token mixer.
pub enum Mixer {
    Linear(DeltaNet),
    Full(Full),
}

pub struct Layer {
    pub common: Common,
    pub mixer: Mixer,
}

#[derive(Weights)]
pub struct Ends {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Hidden)]
    pub norm: Vector,
}

pub struct Qwen35 {
    pub ends: Ends,
    pub layers: Vec<Layer>,
    /// The output head, when not tied to the embedding.
    pub head: Option<Embed>,
}

impl Qwen35 {
    /// Every weight `c` names, each layer as its mixer needs.
    pub fn declare(m: &Model, c: &Config) -> Self {
        let ends: Ends = tessera_trace::declare(m);
        let mut layers = Vec::new();
        for (i, kind) in c.layers.iter().enumerate() {
            let mut path = vec![PathSeg::Field("layers".into()), PathSeg::Index(i as u32)];
            let common = Common::declare(m, &mut path, &[]);
            let mixer = match kind {
                Mixing::Linear => Mixer::Linear(part(m, &mut path, "linear", &[])),
                Mixing::Full => Mixer::Full(part(m, &mut path, "attn", &[])),
            };
            layers.push(Layer { common, mixer });
        }
        let shape = [Vocab.into(), Hidden.into()];
        let head = (!c.tied).then(|| part(m, &mut Vec::new(), "head", &shape));
        Self { ends, layers, head }
    }
}

/// Each row's logits, from tokens and positions `[Seqs, Context]`.
pub fn forward(m: &Model, c: &Config, w: &Qwen35, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let [logits] = m.composite(Composite::Model, || {
        let mut x = m.embed(tokens, &w.ends.embed);
        for l in &w.layers {
            let [out] = m.composite(Composite::Layer, || {
                let h = norm(&x, &l.common.ln1, c.eps);
                let mixed = match &l.mixer {
                    Mixer::Linear(net) => net.apply(m, c, &h),
                    Mixer::Full(attn) => attn.apply(m, c, &h, positions),
                };
                let x = x.add(&mixed);
                let h = norm(&x, &l.common.ln2, c.eps);
                let [mlp] = m.composite(Swiglu::KIND, || [l.common.mlp.apply(m, &h)]);
                [x.add(&mlp)]
            });
            x = out;
        }
        let h = norm(&x, &w.ends.norm, c.eps);
        [h.unembed(w.head.as_ref().unwrap_or(&w.ends.embed))]
    });
    logits
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w = Qwen35::declare(&m, c);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}
