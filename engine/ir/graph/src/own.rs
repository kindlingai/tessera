//! Row ownership: each rank keeps its share of summed rows.

use crate::dims::{DimExpr, DimName};
use crate::infer::TypeError;
use crate::node::{Composite, Graph, Node, NodeId, Op, Region, Row};
use crate::numerics::Numerics;
use crate::shard::{open, retype_node};
use crate::stages::Stage;
use crate::types::{DType, Placement};
use crate::verify::{VerifyError, verify};

fn gather_op() -> Op {
    Op::Row(Row(vec![Stage::GatherRows]))
}

fn own_op() -> Op {
    Op::Row(Row(vec![Stage::OwnRows]))
}

fn owned(g: &Graph, id: NodeId) -> bool {
    g.node(id).ty.placement == Placement::Owned { axis: 0 }
}

/// A gather of `from`: one before `before`, else new.
fn gathered(
    g: &mut Graph,
    from: NodeId,
    at: usize,
    before: usize,
    wire: Option<DType>,
) -> (NodeId, bool) {
    let op = gather_op();
    let found = g.nodes[..before]
        .iter()
        .position(|n| n.op == op && n.args == [from]);
    if let Some(found) = found {
        return (NodeId(found as u32), false);
    }
    let node = Node {
        op,
        args: vec![from],
        ty: g.node(from).ty.clone(),
        numerics: Some(Numerics {
            wire,
            ..Numerics::default()
        }),
    };
    let new = NodeId(at as u32);
    open(g, new, node);
    // A gather at a region's old end lies past it.
    for r in &mut g.regions {
        if r.end.0 == new.0 + 1 && r.start < new {
            r.end = new;
        }
    }
    (new, true)
}

/// Whether `r` can run ragged attention on owned rows.
fn can_own_attn(g: &Graph, r: &Region) -> bool {
    if r.kind != Composite::Attention {
        return false;
    }
    let mut has_attend = false;
    for i in r.start.0 as usize..r.end.0 as usize {
        match &g.nodes[i].op {
            Op::Attend(a) => {
                if !a.ragged {
                    return false;
                }
                has_attend = true;
            }
            Op::Append(k) => {
                if g.kept[*k as usize].placement != Placement::Replicated {
                    return false;
                }
            }
            Op::Conv(c) if c.state.is_some() => {
                let k = c.state.expect("checked") as usize;
                if g.kept[k].placement != Placement::Replicated {
                    return false;
                }
            }
            Op::Delta(d) if d.state.is_some() => {
                let k = d.state.expect("checked") as usize;
                if g.kept[k].placement != Placement::Replicated {
                    return false;
                }
            }
            _ => {}
        }
    }
    has_attend
}

/// Removes dropped pass-through nodes and remaps ids.
fn compact(g: &mut Graph, drop: &[bool]) {
    let mut map = vec![NodeId(0); g.nodes.len()];
    let mut kept = Vec::with_capacity(g.nodes.len());
    for (i, node) in g.nodes.iter().enumerate() {
        if drop[i] {
            map[i] = map[node.args[0].0 as usize];
            continue;
        }
        let mut copy = node.clone();
        for a in &mut copy.args {
            *a = map[a.0 as usize];
        }
        map[i] = NodeId(kept.len() as u32);
        kept.push(copy);
    }
    g.nodes = kept;
    for e in &mut g.entries {
        for o in &mut e.outputs {
            o.node = map[o.node.0 as usize];
        }
    }
    for r in 0..g.regions.len() {
        let span = g.regions[r].start.0 as usize..g.regions[r].end.0 as usize;
        let first = span.clone().find(|&i| !drop[i]);
        let last = span.rev().find(|&i| !drop[i]);
        if let (Some(first), Some(last)) = (first, last) {
            g.regions[r].start = map[first];
            g.regions[r].end = NodeId(map[last].0 + 1);
        }
        for o in &mut g.regions[r].outputs {
            *o = map[o.0 as usize];
        }
        let (start, end) = (g.regions[r].start, g.regions[r].end);
        g.regions[r].inputs = g.reads(start, end);
    }
}

/// Weight index under `id` through any views.
fn weight_of(g: &Graph, mut id: NodeId) -> Option<u32> {
    loop {
        match g.node(id).op {
            Op::Weight(w) => return Some(w),
            Op::View(crate::node::View::Concat) => return None,
            Op::View(_) => id = g.node(id).args[0],
            _ => return None,
        }
    }
}

/// Replicates weights in row-ownable attention regions and drops their sums.
fn unshard_attn(g: &mut Graph) -> Vec<bool> {
    let mut eligible: Vec<bool> = g.regions.iter().map(|r| can_own_attn(g, r)).collect();
    if !eligible.iter().any(|&ok| ok) {
        return eligible;
    }
    let mut in_attn = vec![false; g.nodes.len()];
    for (ri, r) in g.regions.iter().enumerate() {
        if eligible[ri] {
            for flag in &mut in_attn[r.start.0 as usize..r.end.0 as usize] {
                *flag = true;
            }
        }
    }
    let mut outside = vec![false; g.weights.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if in_attn[i] {
            continue;
        }
        for &a in &n.args {
            if let Some(w) = weight_of(g, a) {
                outside[w as usize] = true;
            }
        }
    }
    for (ri, r) in g.regions.iter().enumerate() {
        if !eligible[ri] {
            continue;
        }
        'region: for i in r.start.0 as usize..r.end.0 as usize {
            for &a in &g.nodes[i].args {
                if let Some(w) = weight_of(g, a)
                    && g.weights[w as usize].ty.placement != Placement::Replicated
                    && outside[w as usize]
                {
                    eligible[ri] = false;
                    break 'region;
                }
            }
        }
    }
    let sum = Op::Row(Row(vec![Stage::AllReduce]));
    let mut drop = vec![false; g.nodes.len()];
    let mut dropped = false;
    for (ri, r) in g.regions.iter().enumerate() {
        if !eligible[ri] {
            continue;
        }
        let span = r.start.0 as usize..r.end.0 as usize;
        for (node, flag) in g.nodes[span.clone()].iter().zip(&mut drop[span]) {
            if node.op == sum {
                *flag = true;
                dropped = true;
            }
            for &a in &node.args {
                if let Some(w) = weight_of(g, a) {
                    g.weights[w as usize].ty.placement = Placement::Replicated;
                }
            }
        }
    }
    if dropped {
        compact(g, &drop);
    }
    eligible
}

/// Whether a region's ops split their rows among ranks.
fn splits(ri: usize, r: &Region, owned_attn: &[bool]) -> bool {
    matches!(r.kind, Composite::Moe | Composite::Mlp)
        || (r.kind == Composite::Attention && !owned_attn[ri])
}

/// Gathers owned inputs of split regions starting at `at`.
fn enter_regions(g: &mut Graph, at: usize, wire: Option<DType>, owned_attn: &[bool]) -> bool {
    let mut found = None;
    for (ri, r) in g.regions.iter().enumerate() {
        if r.start.0 as usize == at
            && splits(ri, r, owned_attn)
            && let Some(&i) = r.inputs.iter().find(|i| owned(g, **i))
        {
            found = Some(i);
            break;
        }
    }
    let Some(from) = found else {
        return false;
    };
    let (whole, new) = gathered(g, from, at, at, wire);
    let first = at + usize::from(new);
    for ri in 0..g.regions.len() {
        if g.regions[ri].start.0 as usize != first || !splits(ri, &g.regions[ri], owned_attn) {
            continue;
        }
        let end = g.regions[ri].end.0 as usize;
        for n in &mut g.nodes[first..end] {
            n.args
                .iter_mut()
                .filter(|a| **a == from)
                .for_each(|a| *a = whole);
        }
        g.regions[ri].inputs = g.reads(NodeId(first as u32), NodeId(end as u32));
    }
    true
}

/// Slices a replicated row input before owned attention.
fn enter_owned_attn(g: &mut Graph, at: usize, owned_attn: &[bool]) -> bool {
    let mut found = None;
    for (ri, r) in g.regions.iter().enumerate() {
        if r.start.0 as usize != at || !owned_attn[ri] {
            continue;
        }
        if r.inputs.iter().any(|&i| owned(g, i)) {
            continue;
        }
        if let Some(&from) = r.inputs.iter().find(|&&i| {
            let ty = &g.node(i).ty;
            ty.placement == Placement::Replicated
                && ty.dtype.is_float()
                && ty.rank() == 2
                && weight_of(g, i).is_none()
                && !matches!(
                    ty.shape.0[0],
                    DimExpr::Const(_) | DimExpr::Dim(DimName::Picks)
                )
        }) {
            found = Some(from);
            break;
        }
    }
    let Some(from) = found else {
        return false;
    };
    let node = Node {
        op: own_op(),
        args: vec![from],
        ty: g.node(from).ty.clone(),
        numerics: None,
    };
    let new = NodeId(at as u32);
    open(g, new, node);
    for r in &mut g.regions {
        if r.end.0 == new.0 + 1 && r.start < new {
            r.end = new;
        }
    }
    let first = at + 1;
    for n in &mut g.nodes[first..] {
        n.args
            .iter_mut()
            .filter(|a| **a == from)
            .for_each(|a| *a = new);
    }
    for ri in 0..g.regions.len() {
        let (start, end) = (g.regions[ri].start, g.regions[ri].end);
        if end.0 as usize > first {
            g.regions[ri].inputs = g.reads(start, end);
        }
    }
    true
}

/// Sums over rows scatter. Readers of all rows gather.
pub fn own_rows(g: &Graph) -> Result<Graph, VerifyError> {
    own_rows_with(g, false)
}

/// Row ownership with replicated-cache attention run on owned rows.
pub fn own_rows_attn(g: &Graph) -> Result<Graph, VerifyError> {
    own_rows_with(g, true)
}

/// Whether node `at` sits inside a row-owned attention region.
fn in_owned_attn(g: &Graph, at: usize, owned_attn: &[bool]) -> bool {
    let at = at as u32;
    g.regions
        .iter()
        .enumerate()
        .any(|(ri, r)| owned_attn[ri] && r.start.0 <= at && at < r.end.0)
}

fn own_rows_with(g: &Graph, attn: bool) -> Result<Graph, VerifyError> {
    let mut out = g.clone();
    let mut wire = None;
    for n in &out.nodes {
        if let Op::Row(r) = &n.op
            && r.0.contains(&Stage::AllReduce)
        {
            wire = wire.or(n.numerics.and_then(|num| num.wire));
        }
    }
    let owned_attn = if attn {
        unshard_attn(&mut out)
    } else {
        vec![false; out.regions.len()]
    };
    for n in &mut out.nodes {
        if let Op::Row(r) = &mut n.op
            && n.ty.rank() == 2
            && n.ty.shape.0[0] != DimExpr::Dim(DimName::Picks)
        {
            for s in r.0.iter_mut().filter(|s| **s == Stage::AllReduce) {
                *s = Stage::ReduceScatter;
            }
        }
    }
    let mut at = 0;
    while at < out.nodes.len() {
        if enter_regions(&mut out, at, wire, &owned_attn)
            || enter_owned_attn(&mut out, at, &owned_attn)
        {
            continue;
        }
        let retyped = retype_node(&mut out, at);
        let bad_int = retyped.is_ok()
            && owned(&out, NodeId(at as u32))
            && !out.nodes[at].ty.dtype.is_float()
            && !in_owned_attn(&out, at, &owned_attn);
        if retyped.is_ok() && !bad_int {
            at += 1;
            continue;
        }
        let args = out.nodes[at].args.clone();
        let Some(from) = args.into_iter().find(|a| owned(&out, *a)) else {
            return Err(failed(
                at,
                retyped.expect_err("non-float owned needs owned arg"),
            ));
        };
        let (whole, new) = gathered(&mut out, from, at, at, wire);
        let reader = at + usize::from(new);
        let args = &mut out.nodes[reader].args;
        args.iter_mut()
            .filter(|a| **a == from)
            .for_each(|a| *a = whole);
    }
    for e in 0..out.entries.len() {
        for o in 0..out.entries[e].outputs.len() {
            let from = out.entries[e].outputs[o].node;
            if !owned(&out, from) {
                continue;
            }
            let all = out.nodes.len();
            let (whole, new) = gathered(&mut out, from, from.0 as usize + 1, all, wire);
            if new {
                retype_node(&mut out, whole.0 as usize).map_err(|e| failed(whole.0 as usize, e))?;
            }
            out.entries[e].outputs[o].node = whole;
        }
    }
    for r in 0..out.regions.len() {
        let (start, end) = (out.regions[r].start, out.regions[r].end);
        out.regions[r].inputs = out.reads(start, end);
    }
    verify(&out)?;
    Ok(out)
}

fn failed(at: usize, error: TypeError) -> VerifyError {
    VerifyError {
        node: Some(at as u32),
        error,
    }
}
