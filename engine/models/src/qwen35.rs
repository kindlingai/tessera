//! Qwen3.5: gated delta rule layers and gated full attention.

use crate::shared::{Attend, Stack};
use tessera_trace::DimName::*;
use tessera_trace::{Act, Dim, Linear, Mask, Model, Pairs, Tensor, Vector, Weights};

pub struct Config {
    /// `heads` and `head_dim` are the full-attention layers'.
    pub stack: Stack,
    pub kv_heads: u64,
    /// The leading width of each full-attention head that rotates.
    pub rotary: u64,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Config {
    pub fn dims(&self) -> Vec<Dim> {
        let rest = self.stack.head_dim - self.rotary;
        let own = [(KvHeads, self.kv_heads), (Rope, self.rotary), (Nope, rest)];
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
