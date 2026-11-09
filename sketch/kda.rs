//! Kimi delta attention: short conv, then gated delta rule. Sketch.

use crate::core::*;

pub struct KdaShape {
    pub heads: usize,
    pub head_dim: usize,
    pub conv: usize,
}

#[derive(Weights)]
pub struct Kda {
    pub norm: Norm,
    #[split(heads)]
    pub qkv: Linear,
    #[split(heads)]
    pub conv: Linear,
    /// Per-channel decay gates, finer than Gated DeltaNet's per head.
    #[split(heads)]
    pub gates: Linear,
    #[split(heads)]
    pub out_gate: Linear,
    pub out_norm: Norm,
    #[split(rows)]
    pub out: Linear,
}

/// Reads and writes its conv history and head state.
pub fn kda(m: &mut Model, s: &KdaShape, w: &Kda, l: Layer, x: &Tensor) -> Tensor {
    let h = x.rms_norm(&w.norm);
    let history = m.conv_history(l, s.conv);
    let conv = RecurRule::Conv { width: s.conv };
    let qkv = m.recur(conv, &h.matmul(&w.qkv), &history);
    let state = m.recur_state(l, RecurRule::Kda, s.heads, s.head_dim, s.head_dim);
    let o = m
        .recur(RecurRule::Kda, &qkv, &state)
        .with_gates(&h.matmul(&w.gates));
    o.rms_norm(&w.out_norm)
        .gate(Act::Sigmoid, &h.matmul(&w.out_gate))
        .matmul(&w.out)
}
