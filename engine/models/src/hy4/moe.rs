//! Latent MoE blocks for Hy4.

use super::{Config, ROUTED_HIDDEN};
use crate::shared::CalibTaps;
use crate::shared::{Dense, ROTATION, Rotated, Swiglu, cat, part, swiglu_tapped};
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, CalibTap, DimExpr, Embed, Linear, Linears, Model, Norm, PathSeg, Tensor, Vector, Weights,
};

/// Clamps `gate <= limit` and `-limit <= up <= limit`.
fn clamp_swiglu(gate: Tensor, up: Tensor, limit: Option<f32>) -> [Tensor; 2] {
    match limit {
        Some(lim) => [gate.min(lim), up.min(lim).max(-lim)],
        None => [gate, up],
    }
}

/// Routed latent experts in `ROUTED_HIDDEN` space.
pub trait LatentExperts: Weights {
    type Shared: Dense;

    fn apply_routed(
        &self,
        m: &Model,
        x_lat: &Tensor,
        picks: [&Tensor; 2],
        limit: Option<f32>,
    ) -> Tensor;
}

/// Plain routed latent experts.
#[derive(Weights)]
pub struct PlainLatent {
    #[shape(Experts, ROUTED_HIDDEN, Ffn)]
    pub ex_gate: Linears,
    #[shape(Experts, ROUTED_HIDDEN, Ffn)]
    pub ex_up: Linears,
    #[shape(Experts, Ffn, ROUTED_HIDDEN)]
    pub ex_down: Linears,
}

impl LatentExperts for PlainLatent {
    type Shared = Linear;

    fn apply_routed(
        &self,
        m: &Model,
        x_lat: &Tensor,
        [ids, gates]: [&Tensor; 2],
        limit: Option<f32>,
    ) -> Tensor {
        let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, ids));
        let [g, u] = clamp_swiglu(on(&self.ex_gate, x_lat), on(&self.ex_up, x_lat), limit);
        let inner = g.act(Act::Silu).mul(&u);
        let picks = x_lat.ty().rank() as u32 - 1;
        on(&self.ex_down, &inner).mix(picks, gates)
    }
}

/// Exlr8 rotated routed latent experts.
#[derive(Weights)]
pub struct RotatedLatent {
    #[shape(ROUTED_HIDDEN)]
    pub signs_in: Vector,
    #[shape(ROUTED_HIDDEN)]
    pub signs_out: Vector,
    #[shape(Experts, ROUTED_HIDDEN, Ffn)]
    pub ex_gate: Linears,
    #[shape(Experts, ROUTED_HIDDEN)]
    pub ex_gate_su: Embed,
    #[shape(Experts, Ffn)]
    pub ex_gate_sv: Embed,
    #[shape(Experts, ROUTED_HIDDEN, Ffn)]
    pub ex_up: Linears,
    #[shape(Experts, ROUTED_HIDDEN)]
    pub ex_up_su: Embed,
    #[shape(Experts, Ffn)]
    pub ex_up_sv: Embed,
    #[shape(Experts, Ffn, ROUTED_HIDDEN)]
    pub ex_down: Linears,
    #[shape(Experts, Ffn)]
    pub ex_down_su: Embed,
    #[shape(Experts, ROUTED_HIDDEN)]
    pub ex_down_sv: Embed,
}

impl LatentExperts for RotatedLatent {
    type Shared = Rotated;

    fn apply_routed(
        &self,
        m: &Model,
        x_lat: &Tensor,
        [ids, gates]: [&Tensor; 2],
        limit: Option<f32>,
    ) -> Tensor {
        let picks = x_lat.ty().rank() as u32 - 1;
        let lead = &x_lat.ty().shape.0[..picks as usize];
        let x_rot = x_lat
            .mul_vector(&self.signs_in)
            .reshape(cat(lead, [DimExpr::Const(1), ROUTED_HIDDEN.into()]))
            .expand(picks, TopK);
        let turn = |x: &Tensor, set: &Linears, su: &Embed, sv: &Embed| {
            let turned = x
                .mul(&m.embed(ids, su))
                .hadamard(ROTATION)
                .matmul_rows(&m.pick(set, ids));
            turned.hadamard(ROTATION).mul(&m.embed(ids, sv))
        };
        let g = turn(&x_rot, &self.ex_gate, &self.ex_gate_su, &self.ex_gate_sv);
        let u = turn(&x_rot, &self.ex_up, &self.ex_up_su, &self.ex_up_sv);
        let [g, u] = clamp_swiglu(g, u, limit);
        let inner = g.act(Act::Silu).mul(&u);
        let out = turn(&inner, &self.ex_down, &self.ex_down_su, &self.ex_down_sv);
        out.mix(picks, gates).mul_vector(&self.signs_out)
    }
}

/// LatentMoE block with shared expert and combine norm.
pub(super) struct LatentMoe<E: LatentExperts> {
    post_attn_norm: Norm,
    router: Linear,
    router_bias: Vector,
    routed_down: Linear,
    routed: E,
    routed_up: Linear,
    shared: Swiglu<E::Shared>,
    combine_norm: Norm,
}

impl<E: LatentExperts> Weights for LatentMoe<E> {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, _shape: &[DimExpr]) -> Self {
        let hid: DimExpr = Hidden.into();
        let r_hid: DimExpr = ROUTED_HIDDEN.into();
        let exp: DimExpr = Experts.into();
        Self {
            post_attn_norm: part(m, path, "post_attn_norm", std::slice::from_ref(&hid)),
            router: part(m, path, "router", &[hid.clone(), exp.clone()]),
            router_bias: part(m, path, "router_bias", &[exp]),
            routed_down: part(m, path, "routed_down", &[hid.clone(), r_hid.clone()]),
            routed: part(m, path, "routed", &[]),
            routed_up: part(m, path, "routed_up", &[r_hid, hid.clone()]),
            shared: part(m, path, "shared", &[hid.clone(), Ffn.into()]),
            combine_norm: part(m, path, "combine_norm", &[hid]),
        }
    }
}

impl<E: LatentExperts> LatentMoe<E> {
    pub(super) fn apply(&self, m: &Model, c: &Config, h_in: &Tensor, taps: &CalibTaps) -> Tensor {
        let h = h_in.rms_norm(&self.post_attn_norm, c.eps);
        let probs = h.matmul(&self.router).act(Act::Sigmoid);
        let ids = probs.add_vector(&self.router_bias).top_k(TopK);
        let gates = probs.take(&ids).unit_sum().scale(c.route_scale);
        taps.note(CalibTap::RouterIds, &ids);
        taps.note(CalibTap::RouterGates, &gates);
        let x_lat = h.matmul(&self.routed_down);
        let y_lat = self
            .routed
            .apply_routed(m, &x_lat, [&ids, &gates], c.swiglu_limit);
        let y_routed = y_lat.matmul(&self.routed_up);
        let [y_shared, sh_in] =
            swiglu_tapped(&h, [&self.shared.gate, &self.shared.up, &self.shared.down]);
        taps.note(CalibTap::SharedDownIn, &sh_in);
        y_routed.add(&y_shared).rms_norm(&self.combine_norm, c.eps)
    }
}
