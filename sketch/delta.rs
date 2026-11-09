//! Delta-rule mixers: KDA and Gated DeltaNet share one shape. Sketch.

use crate::core::*;

/// What separates KDA from Gated DeltaNet.
pub enum DeltaKind {
    /// Per-channel decay gates, sigmoid output gate.
    Kda,
    /// Per-head decay gate, SiLU output gate.
    GatedDelta,
}

pub struct DeltaShape {
    pub kind: DeltaKind,
    pub heads: usize,
    pub head_dim: usize,
    pub conv: usize,
    pub norm: Norm,
}

#[derive(Weights)]
pub struct Delta {
    pub norm: Norm,
    #[shape(Hidden, Heads, HeadDim)]
    pub qkv: Linear,
    #[shape(Heads, HeadDim, Conv)]
    pub conv: Linear,
    #[shape(Hidden, Heads, HeadDim)]
    pub gates: Linear,
    #[shape(Hidden, Heads, HeadDim)]
    pub out_gate: Linear,
    pub out_norm: Norm,
    #[shape(Heads, HeadDim, Hidden)]
    pub out: Linear,
}

/// Short conv, then the delta rule over per-head state.
pub fn delta(m: &mut Model, s: &DeltaShape, w: &Delta, l: Layer, x: &Tensor) -> Tensor {
    let (rule, gate) = match s.kind {
        DeltaKind::Kda => (RecurRule::Kda, Act::Sigmoid),
        DeltaKind::GatedDelta => (RecurRule::GatedDelta, Act::Silu),
    };
    let h = x.norm_as(&s.norm, &w.norm);
    let history = m.conv_history(l, s.conv);
    let conv = RecurRule::Conv { width: s.conv };
    let qkv = m.recur(conv, &h.matmul(&w.qkv), &history);
    let state = m.recur_state(l, rule, s.heads, s.head_dim, s.head_dim);
    let o = m.recur(rule, &qkv, &state).with_gates(&h.matmul(&w.gates));
    o.rms_norm(&w.out_norm)
        .gate(gate, &h.matmul(&w.out_gate))
        .matmul(&w.out)
}
