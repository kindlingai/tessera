//! Multi-Token Prediction (MTP) one-layer MLA drafter.

use crate::mla::{self, Attention, Config, Index, Indexer, Routed};
use crate::shared::Steps;
use tessera_graph::{
    Graph, Node, NodeId, Numerics, Op, Parts, PathSeg, Placement, Row as GraphRow, Sharded, Shares,
    Stage, VerifyError, infer, shard_mode, verify,
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
    #[shape(IndexLayers)]
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

/// Shards an MTP graph, splitting `eh_proj` across ranks.
pub fn shard(
    g: &Graph,
    parts: &Parts,
    shares: &Shares,
    moe_tp: bool,
) -> Result<Sharded, VerifyError> {
    let mut sharded = shard_mode(g, parts, shares, moe_tp)?;
    split_eh_proj(&mut sharded.graph, parts, shares);
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
        (is_eh && wt.ty.placement == Placement::Replicated).then_some((NodeId(i as u32), w))
    });
    let Some((at, w)) = target else {
        return;
    };
    let val = |d: &DimExpr| d.value(&g.dims, &|_| None).map(|v| v as usize);
    let Some(sizes) = g.weights[w as usize]
        .ty
        .shape
        .0
        .iter()
        .map(val)
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let Some(counts) = parts.counts(sizes[0]) else {
        return;
    };
    if counts.iter().any(|&c| !shares(w, &sizes, 0, c)) {
        return;
    }
    let mut tried = g.clone();
    tried.weights[w as usize].ty.placement = Placement::Split { axis: 0 };
    let new = NodeId(at.0 + 1);
    let shift = |n: &mut NodeId| {
        if *n >= new {
            n.0 += 1;
        } else if *n == at {
            *n = new;
        }
    };
    for node in &mut tried.nodes[new.0 as usize..] {
        node.args.iter_mut().for_each(shift);
    }
    let gather = Node {
        op: Op::Row(GraphRow(vec![Stage::AllGather])),
        args: vec![at],
        ty: tried.node(at).ty.clone(),
        numerics: Some(Numerics::default()),
    };
    tried.nodes.insert(new.0 as usize, gather);
    for e in &mut tried.entries {
        for o in &mut e.outputs {
            shift(&mut o.node);
        }
    }
    for r in &mut tried.regions {
        if r.start >= new {
            r.start.0 += 1;
        }
        if r.end >= new {
            r.end.0 += 1;
        }
        r.inputs.iter_mut().chain(&mut r.outputs).for_each(shift);
    }
    for i in 0..tried.nodes.len() {
        let args: Vec<_> = tried.nodes[i]
            .args
            .iter()
            .map(|a| &tried.node(*a).ty)
            .collect();
        let Ok(ty) = infer(&tried, &tried.nodes[i].op, &args) else {
            return;
        };
        tried.nodes[i].ty = ty;
    }
    if verify(&tried).is_ok() {
        *g = tried;
    }
}
