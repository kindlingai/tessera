//! Qwen3.5: gated delta rule layers and gated full attention.

use crate::shared::{Attend, Stack};
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Dim, DimExpr, Linear, Mask, Model, Norm, Pairs, Taps, Tensor, Vector, Weights,
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
    pub fn dims(&self) -> Vec<Dim> {
        let (rest, l) = (self.stack.head_dim - self.rotary, &self.linear);
        let own = [
            (KvHeads, self.kv_heads),
            (Rope, self.rotary),
            (Nope, rest),
            (KeyHeads, l.key_heads),
            (ValueHeads, l.value_heads),
            (KeyDim, l.key_dim),
            (ValueDim, l.value_dim),
            (Window, l.window),
        ];
        self.stack.dims(&own)
    }
}

/// Rows over their root mean square, times one plus `gain`.
fn norm(x: &Tensor, gain: &Vector, eps: f32) -> Tensor {
    let r = x.rms(eps);
    r.add(&r.mul_vector(gain))
}

/// Gated attention over every earlier row.
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
        if c.rotary < c.stack.head_dim {
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
