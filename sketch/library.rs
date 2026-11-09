//! Functions built from the primitives. Sketch; it does not compile.

use crate::core::*;

#[derive(Weights)]
pub struct Mlp<A> {
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
    pub act: A,
}

/// Up projection, gated activation, down projection: a Partial.
pub fn mlp<A: Activation>(w: &Mlp<A>, h: &Tensor) -> Tensor {
    h.matmul(&w.up).act(A::ACT).matmul(&w.down)
}

#[derive(Weights)]
pub struct Moe<A> {
    pub router: Linear,
    pub router_bias: Vector,
    #[shape(Experts)]
    pub experts: Experts<Mlp<A>>,
    pub shared: Option<Mlp<A>>,
}

/// Router top-k, then grouped expert matmuls, then weighted combine.
pub fn moe<A: Activation>(m: &mut Model, w: &Moe<A>, h: &Tensor, k: usize, l: Layer) -> Tensor {
    let logits = h.matmul(&w.router).f32();
    let picked = m.top_k(&logits.add(&w.router_bias), k);
    let parts = m.partition(&picked.ids, w.experts.len());
    let up = h.matmul_grouped(&w.experts.up, &parts).act(A::ACT);
    let out = up.matmul_grouped(&w.experts.down, &parts);
    let routed = m.scatter(&out.mul(&picked.weights), &parts);
    match &w.shared {
        Some(s) => routed.add(&m.site(Site::SharedExpert(l), mlp(s, h))),
        None => routed,
    }
}

/// Greedy speculation: the run of drafts the target agrees with.
pub fn greedy_accept(m: &mut Model, target: &Tensor, drafts: &Tensor) -> Counts {
    let best = m.sample(target, Sampler::Argmax);
    let same = best.compare(Cmp::Eq, drafts);
    m.prefix(&same, PrefixOp::And).count()
}

/// Rejection sampling: accept while u < p / q.
pub fn rejection_accept(m: &mut Model, p: &Tensor, q: &Tensor, drafts: &Tensor) -> Counts {
    let ratio = p.gather_ids(drafts).div(&q.gather_ids(drafts)).min_one();
    let u = m.sample(&ratio, Sampler::Uniform);
    let keep = u.compare(Cmp::Lt, &ratio);
    m.prefix(&keep, PrefixOp::And).count()
}

/// Rows each sequence asks for, clamped to a shared budget.
pub fn fit_budget(m: &mut Model, wanted: &Tensor, budget: Rows) -> Tensor {
    let total = m.prefix(wanted, PrefixOp::Sum);
    wanted.clamp_to(&total, budget)
}

/// Drafting continues only for rows confident enough.
pub fn keep_confident(m: &mut Model, confidence: &Tensor, floor: f32) -> Plan {
    m.filter(&confidence.compare_scalar(Cmp::Gt, floor))
}
