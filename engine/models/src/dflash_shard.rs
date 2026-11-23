//! Tensor-parallel sharding for DFlash, DFlash2 and DSpine drafters.

use tessera_graph::{
    DType, DimExpr, Graph, Node, NodeId, Numerics, Op, Parts, PathSeg, Placement, Row as GraphRow,
    Select, Shape, Sharded, Shares, Stage, TensorType, VerifyError, View, infer, shard_mode,
    verify,
};

pub(crate) fn insert_stage(g: &mut Graph, at: NodeId, stage: Stage) {
    let new = NodeId(at.0 + 1);
    let remap = |n: &mut NodeId| {
        n.0 += u32::from(*n >= new);
        if *n == at {
            *n = new;
        }
    };
    g.nodes[new.0 as usize..]
        .iter_mut()
        .for_each(|n| n.args.iter_mut().for_each(remap));
    let ty = g.node(at).ty.clone();
    let op = Op::Row(GraphRow(vec![stage]));
    let node = Node {
        op,
        args: vec![at],
        ty,
        numerics: Some(Numerics::default()),
    };
    g.nodes.insert(new.0 as usize, node);
    g.entries
        .iter_mut()
        .flat_map(|e| &mut e.outputs)
        .for_each(|o| remap(&mut o.node));
    for r in &mut g.regions {
        r.start.0 += u32::from(r.start >= new);
        r.end.0 += u32::from(r.end >= new);
        r.inputs.iter_mut().chain(&mut r.outputs).for_each(remap);
    }
}

pub(crate) fn split_weight(g: &mut Graph, node: NodeId, axis: u32) -> bool {
    let Op::Weight(w) = g.node(node).op else {
        return false;
    };
    let ty = &mut g.weights[w as usize].ty;
    if ty.rank() != 2 || ty.placement != Placement::Replicated {
        return false;
    }
    ty.placement = Placement::Split { axis };
    true
}

pub(crate) fn adopt_split(out: &mut Graph, mut g: Graph, parts: &Parts, shares: &Shares) -> bool {
    for i in 0..g.nodes.len() {
        if let Op::Append(k) = g.nodes[i].op {
            let p = g.node(g.nodes[i].args[0]).ty.placement;
            g.kept[k as usize].placement = match p {
                Placement::Split { axis } => Placement::Split { axis: axis + 1 },
                other => other,
            };
        }
        let args: Vec<_> = g.nodes[i].args.iter().map(|a| &g.node(*a).ty).collect();
        let Ok(ty) = infer(&g, &g.nodes[i].op, &args) else {
            return false;
        };
        g.nodes[i].ty = ty;
    }
    let val = |d: &DimExpr| d.value(&g.dims, &|_| None).map(|v| v as usize);
    for n in &g.nodes {
        let Placement::Split { axis } = n.ty.placement else {
            continue;
        };
        let a = axis as usize;
        let Some(counts) = val(&n.ty.shape.0[a]).and_then(|v| parts.counts(v)) else {
            return false;
        };
        if let Op::Weight(w) = n.op {
            let sizes: Option<Vec<_>> = n.ty.shape.0.iter().map(val).collect();
            if sizes.is_none_or(|s| counts.iter().any(|&c| !shares(w, &s, a, c))) {
                return false;
            }
        }
    }
    if verify(&g).is_err() {
        return false;
    }
    *out = g;
    true
}

/// Replaces full-logits top-k with per-rank top-k and small gather.
pub(crate) fn shard_topk(out: &mut Graph, parts: &Parts, shares: &Shares) {
    let nr = parts.ranks() as u64;
    if nr <= 1 || parts.0.windows(2).any(|w| w[0] != w[1]) {
        return;
    }
    let gather_op = Op::Row(GraphRow(vec![Stage::AllGather]));
    let mut found = None;
    for (i, n) in out.nodes.iter().enumerate() {
        let Op::Select(Select::TopK { k }) = &n.op else {
            continue;
        };
        let [gather_id] = n.args[..] else {
            continue;
        };
        let gnode = out.node(gather_id);
        if gnode.op != gather_op {
            continue;
        }
        let local_id = gnode.args[0];
        let lnode = out.node(local_id);
        if lnode.op != Op::Matmul
            || lnode.ty.rank() != 2
            || lnode.ty.placement != (Placement::Split { axis: 1 })
        {
            continue;
        }
        let cands_id = NodeId(i as u32);
        if out.regions.iter().any(|r| r.end > cands_id) {
            continue;
        }
        let val = |d: &DimExpr| d.value(&out.dims, &|_| None);
        let n_expr = lnode.ty.shape.0[0].clone();
        let Some(vocab) = val(&lnode.ty.shape.0[1]) else {
            continue;
        };
        let Some(k_val) = val(k) else {
            continue;
        };
        let even = parts.counts((nr * k_val) as usize) == Some(vec![k_val as usize; nr as usize]);
        if !even
            || vocab > (1u64 << 24)
            || k_val == 0
            || !vocab.is_multiple_of(nr)
            || vocab / nr < k_val
        {
            continue;
        }
        let unary_id = out.nodes.iter().enumerate().find_map(|(j, u)| {
            (u.op == Op::Select(Select::Take) && u.args == [gather_id, cands_id])
                .then_some(NodeId(j as u32))
        });
        found = Some((local_id, gather_id, cands_id, unary_id, n_expr, k_val));
        break;
    }
    let Some((local_id, gather_id, cands_id, unary_id, n_expr, k_val)) = found else {
        return;
    };
    let mut tried = out.clone();
    let pos = cands_id.0 as usize;
    let mut built: Vec<Node> = Vec::with_capacity(16);
    let dummy = TensorType::new(Vec::<DimExpr>::new(), DType::F32);
    let mut push = |op: Op, args: Vec<NodeId>, reduces: bool| -> NodeId {
        let id = NodeId((pos + built.len()) as u32);
        built.push(Node {
            op,
            args,
            ty: dummy.clone(),
            numerics: reduces.then(Numerics::default),
        });
        id
    };
    let rk = DimExpr::Const(nr * k_val);
    let pair_in = Shape(vec![n_expr.clone(), rk.clone(), DimExpr::Const(1)]);
    let flat = Shape(vec![n_expr, rk.clone()]);
    let local_idx = push(Op::Select(Select::TopK { k: rk }), vec![local_id], false);
    let local_val = push(Op::Select(Select::Take), vec![local_id, local_idx], false);
    let local_idx_f = push(Op::Cast(DType::F32), vec![local_idx], false);
    let val_3 = push(
        Op::View(View::Reshape(pair_in.clone())),
        vec![local_val],
        false,
    );
    let idx_3 = push(Op::View(View::Reshape(pair_in)), vec![local_idx_f], false);
    let both_3 = push(Op::View(View::Concat), vec![val_3, idx_3], false);
    let both = push(Op::View(View::Permute(vec![0, 2, 1])), vec![both_3], false);
    let gathered = push(Op::Row(GraphRow(vec![Stage::AllGather])), vec![both], true);
    let all_val_3 = push(
        Op::View(View::Slice {
            axis: 1,
            start: 0u64.into(),
            len: 1u64.into(),
        }),
        vec![gathered],
        false,
    );
    let all_val = push(
        Op::View(View::Reshape(flat.clone())),
        vec![all_val_3],
        false,
    );
    let all_idx_3 = push(
        Op::View(View::Slice {
            axis: 1,
            start: 1u64.into(),
            len: 1u64.into(),
        }),
        vec![gathered],
        false,
    );
    let all_idx_f = push(Op::View(View::Reshape(flat)), vec![all_idx_3], false);
    let pick = push(
        Op::Select(Select::TopK {
            k: DimExpr::Const(k_val),
        }),
        vec![all_val],
        false,
    );
    let global_idx_f = push(Op::Select(Select::Take), vec![all_idx_f, pick], false);
    let global_cands = push(Op::Cast(DType::U32), vec![global_idx_f], false);
    let new_unary = unary_id.map(|_| push(Op::Select(Select::Take), vec![all_val, pick], false));
    let shift = built.len() as u32;
    let old_cands = NodeId(cands_id.0 + shift);
    let old_unary = unary_id.map(|u| NodeId(u.0 + shift));
    let remap = |n: &mut NodeId| {
        if n.0 >= pos as u32 {
            n.0 += shift;
        }
        if *n == old_cands {
            *n = global_cands;
        } else if let (Some(ou), Some(nu)) = (old_unary, new_unary)
            && *n == ou
        {
            *n = nu;
        }
    };
    tried.nodes.splice(pos..pos, built);
    for n in &mut tried.nodes[pos + shift as usize..] {
        n.args.iter_mut().for_each(remap);
    }
    for o in tried.entries.iter_mut().flat_map(|e| &mut e.outputs) {
        remap(&mut o.node);
    }
    for r in &mut tried.regions {
        if r.end == NodeId(gather_id.0 + 1) && r.outputs.contains(&gather_id) {
            r.end = gather_id;
            for o in &mut r.outputs {
                if *o == gather_id {
                    *o = local_id;
                }
            }
        }
        if r.start.0 >= pos as u32 {
            r.start.0 += shift;
        }
        if r.end.0 >= pos as u32 {
            r.end.0 += shift;
        }
        r.inputs.iter_mut().chain(&mut r.outputs).for_each(remap);
    }
    adopt_split(out, tried, parts, shares);
}

/// Shards a DFlash/DFlash2/DSpine graph across ranks in `parts`.
pub fn shard(g: &Graph, parts: &Parts, shares: &Shares) -> Result<Sharded, VerifyError> {
    let mut sharded = shard_mode(g, parts, shares, false)?;
    let out = &mut sharded.graph;
    // Shard `fc` (AllGather) and conv-wrapped `o`/`down` (AllReduce).
    let mut i = 0;
    while i < out.nodes.len() {
        let id = NodeId(i as u32);
        i += 1;
        let node = out.node(id);
        let Some(&w_node) = (node.op == Op::Matmul).then(|| &node.args[1]) else {
            continue;
        };
        let Op::Weight(w) = out.node(w_node).op else {
            continue;
        };
        let wt = &out.weights[w as usize];
        if wt.ty.placement != Placement::Replicated {
            continue;
        }
        let mut tried = out.clone();
        match wt.path.last() {
            Some(PathSeg::Field(f)) if f == "fc" && wt.path.len() == 1 => {
                let lone = (0..out.nodes.len()).all(|k| {
                    k + 1 == i
                        || !out.nodes[k]
                            .args
                            .iter()
                            .any(|a| out.node(*a).op == Op::Weight(w))
                });
                if !lone || !split_weight(&mut tried, w_node, 0) {
                    continue;
                }
                insert_stage(&mut tried, id, Stage::AllGather);
            }
            Some(PathSeg::Field(f)) if f == "o" || f == "down" => {
                let (mut stack, mut seen) = (vec![node.args[0]], vec![false; out.nodes.len()]);
                while let Some(cur) = stack.pop() {
                    if std::mem::replace(&mut seen[cur.0 as usize], true) {
                        continue;
                    }
                    let n = out.node(cur);
                    if (n.op == Op::Matmul && split_weight(&mut tried, n.args[1], 0))
                        || split_weight(&mut tried, cur, 0)
                    {
                        continue;
                    }
                    stack.extend_from_slice(&n.args);
                }
                split_weight(&mut tried, w_node, 1);
                insert_stage(&mut tried, id, Stage::AllReduce);
            }
            _ => continue,
        }
        if adopt_split(out, tried, parts, shares) {
            i += 1;
        }
    }
    shard_topk(out, parts, shares);
    verify(out)?;
    Ok(sharded)
}
