//! Multi-Token Prediction (MTP) one-layer MLA drafter.

use crate::dflash_shard::{adopt_split, insert_stage, shard_topk, split_weight};
use crate::mla::{self, Attention, Config, INDEX_LAYERS, Index, Indexer, Routed};
use crate::shared::Steps;
use tessera_graph::{
    Graph, NodeId, Op, Parts, PathSeg, Placement, Sharded, Shares, Stage, VerifyError, shard_mode,
    verify,
};
use tessera_trace::DimName::*;
use tessera_trace::{
    Composite, DType, Dim, DimExpr, Embed, EntryKind, Linear, Model, Norm, Port, Role, Size,
    TensorType, Weights,
};

/// Single-layer MTP config derived from a target MLA config.
pub fn config(target: &Config, indexed: bool) -> Config {
    let mut c = target.clone();
    c.stack.layers = 1;
    c.stack.steps = Steps::Plain;
    c.stack.taps.clear();
    c.dense_layers = 0;
    c.index = target
        .index
        .filter(|_| indexed)
        .map(|i| Index { every: 1, ..i });
    c
}

/// 1-based tap index capturing the post-final-norm hidden state.
pub fn tap_layer(target: &Config) -> u64 {
    target.stack.layers + 1
}

/// Weights for one MLA MTP draft layer.
#[derive(Weights)]
pub struct Mtp<R: Routed> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Hidden)]
    pub enorm: Norm,
    #[shape(Hidden)]
    pub hnorm: Norm,
    #[shape(Hidden * 2, Hidden)]
    pub eh_proj: Linear,
    pub attn: Attention,
    #[shape(INDEX_LAYERS)]
    pub indexers: Vec<Indexer>,
    #[shape(Hidden)]
    pub mlp_norm: Norm,
    pub moe: R,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Vocab)]
    pub head: Linear,
}

/// MTP step entry: tokens and hidden states in, drafts out.
pub fn stepper<R: Routed>(c: &Config) -> Graph {
    assert_eq!(c.stack.layers, 1, "mtp graph expects one layer");
    let mut dims = c.dims();
    dims.push(Dim {
        name: Rows,
        size: Size::Bounded(c.stack.max_context),
    });
    let m = Model::new(dims);
    let w: Mtp<R> = tessera_trace::declare(&m);
    let port = |role, shape: Vec<DimExpr>, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let [tokens, positions, slots, hidden, scale] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, vec![Rows.into()], DType::U32),
            port(Role::Positions, vec![Rows.into()], DType::U32),
            port(Role::Slots, vec![Rows.into()], DType::U32),
            port(Role::Taps, vec![Rows.into(), Hidden.into()], DType::F32),
            port(Role::Taps, vec![Rows.into(), DimExpr::Const(1)], DType::F32),
        ],
    );
    let [logits, out_hidden] = m.composite(Composite::Model, || {
        let emb = m
            .embed(&tokens, &w.embed)
            .rms_norm(&w.enorm, c.eps)
            .mul(&scale.expand(1, Hidden));
        let hid = hidden.rms_norm(&w.hnorm, c.eps);
        let x = emb.concat(&hid).matmul(&w.eh_proj);
        let x = mla::step_layer(
            &m,
            c,
            &w.attn,
            w.indexers.first(),
            &w.mlp_norm,
            &w.moe,
            &x,
            &positions,
            &slots,
        );
        let h = x.rms_norm(&w.norm, c.eps);
        [h.matmul(&w.head), h]
    });
    let ids = logits.top_k(1u64).reshape([Rows]);
    m.finish([
        (Role::Tokens, &ids),
        (Role::Logits, &logits),
        (Role::Taps, &out_hidden),
    ])
}

/// Shards an MTP graph, splitting `eh_proj` and local argmax.
pub fn shard(
    g: &Graph,
    parts: &Parts,
    shares: &Shares,
    moe_tp: bool,
) -> Result<Sharded, VerifyError> {
    let mut sharded = shard_mode(g, parts, shares, moe_tp)?;
    split_eh_proj(&mut sharded.graph, parts, shares);
    shard_topk(&mut sharded.graph, parts, shares);
    sharded
        .whole
        .retain(|w| sharded.graph.weights[w.weight as usize].ty.placement == Placement::Replicated);
    verify(&sharded.graph)?;
    Ok(sharded)
}

fn split_eh_proj(g: &mut Graph, parts: &Parts, shares: &Shares) {
    let target = (0..g.nodes.len()).find_map(|i| {
        let n = &g.nodes[i];
        if n.op != Op::Matmul {
            return None;
        }
        let Op::Weight(w) = g.node(n.args[1]).op else {
            return None;
        };
        let wt = &g.weights[w as usize];
        let is_eh = matches!(wt.path.as_slice(), [PathSeg::Field(f)] if f == "eh_proj");
        (is_eh && wt.ty.placement == Placement::Replicated).then_some((NodeId(i as u32), n.args[1]))
    });
    let Some((at, w_node)) = target else {
        return;
    };
    let mut tried = g.clone();
    if split_weight(&mut tried, w_node, 0) {
        insert_stage(&mut tried, at, Stage::AllGather);
        adopt_split(g, tried, parts, shares);
    }
}
