//! Tensor parallel splits matmuls. Expert parallel splits sets.

use crate::infer::{infer, poly};
use crate::node::{Graph, Node, NodeId, Op, Row, Select};
use crate::numerics::Numerics;
use crate::parts::Parts;
use crate::stages::Stage;
use crate::types::Placement;
use crate::verify::{VerifyError, verify};

/// A split graph, and the splits it left out.
#[derive(Clone, Debug)]
pub struct Sharded {
    pub graph: Graph,
    pub whole: Vec<Whole>,
}

/// A weight a split would claim, kept whole, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Whole {
    pub weight: u32,
    pub why: String,
}

/// Splits residual-bound matmuls and gathered sets that `parts` divides.
pub fn shard(g: &Graph, parts: &Parts) -> Result<Sharded, VerifyError> {
    shard_aligned(g, parts, &|_, _, _, _| true)
}

/// Whether weight `w` shares `(w, shape, axis, count)` as bytes.
pub type Shares<'a> = dyn Fn(u32, &[usize], usize, usize) -> bool + 'a;

/// As `shard`. A split weight's form `shares` each part.
pub fn shard_aligned(g: &Graph, parts: &Parts, shares: &Shares) -> Result<Sharded, VerifyError> {
    shard_mode(g, parts, shares, false)
}

/// As `shard_aligned`, splitting wide experts along inner width.
pub fn shard_mode(
    g: &Graph,
    parts: &Parts,
    shares: &Shares,
    moe_tp: bool,
) -> Result<Sharded, VerifyError> {
    let fit = &Fit { parts, shares };
    let mut out = g.clone();
    let mut whole = Vec::new();
    let mut at = 0;
    while at < out.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        if !returns(&out, id) {
            continue;
        }
        let tried = settle(split(&out, id), &out);
        // The sum is the new node behind this one.
        if adopt(&mut out, tried, fit, Some(&mut whole)) {
            at += 1;
        }
    }
    if moe_tp {
        let mut at = 0;
        while at < out.nodes.len() {
            let id = NodeId(at as u32);
            at += 1;
            let Some(tried) = split_moe(&out, id, fit.parts) else {
                continue;
            };
            let tried = settle(tried, &out);
            if adopt(&mut out, tried, fit, None) {
                at += 1;
            }
        }
    }
    let routed = route(&out);
    adopt(&mut out, routed, fit, Some(&mut whole));
    let gathered = gather(&out);
    adopt(&mut out, gathered, fit, Some(&mut whole));
    if let Ok(merged) = retype(&merge(&out)) {
        out = merged;
    }
    verify(&out)?;
    Ok(Sharded { graph: out, whole })
}

/// Drops each split a failed op reads, until it types.
fn settle(tried: Graph, was: &Graph) -> Graph {
    let first = tried.clone();
    let mut tried = tried;
    loop {
        let Err(at) = retype_at(&tried) else {
            return tried;
        };
        let mut reverted = false;
        for w in feeders(&tried, at) {
            let placement = &mut tried.weights[w as usize].ty.placement;
            let whole = was.weights[w as usize].ty.placement;
            reverted |= std::mem::replace(placement, whole) != whole;
        }
        // Nothing left to drop: the first try keeps its reason.
        if !reverted {
            return first;
        }
    }
}

/// The weights `at` reads, through any ops.
fn feeders(g: &Graph, at: usize) -> Vec<u32> {
    let mut found = Vec::new();
    let mut pending = vec![NodeId(at as u32)];
    let mut seen = vec![false; g.nodes.len()];
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut seen[n.0 as usize], true) {
            continue;
        }
        match g.node(n).op {
            Op::Weight(w) => found.push(w),
            _ => pending.extend(g.node(n).args.iter().copied()),
        }
    }
    found
}

/// Takes `tried` if it types and fits `fit`.
fn adopt(out: &mut Graph, tried: Graph, fit: &Fit, whole: Option<&mut Vec<Whole>>) -> bool {
    let why = match retype(&tried) {
        Ok(typed) => match misfit(&typed, fit) {
            None => {
                *out = typed;
                return true;
            }
            Some(why) => why,
        },
        Err(e) => e.to_string(),
    };
    let Some(whole) = whole else {
        return false;
    };
    let claimed = (out.weights.iter().zip(&tried.weights))
        .enumerate()
        .filter(|(_, (was, now))| was.ty.placement != now.ty.placement);
    for (w, _) in claimed {
        let why = why.clone();
        whole.push(Whole {
            weight: w as u32,
            why,
        });
    }
    false
}

/// The ranks' parts, and whether a weight's form shares them.
struct Fit<'a> {
    parts: &'a Parts,
    shares: &'a Shares<'a>,
}

/// The first split `fit` cannot divide or share.
fn misfit(g: &Graph, fit: &Fit) -> Option<String> {
    for n in &g.nodes {
        let Placement::Split { axis } = n.ty.placement else {
            continue;
        };
        let (shape, parts) = (&n.ty.shape, &fit.parts.0);
        let at = axis as usize;
        let whole = poly(&g.dims, &shape.0[at]).ok()?.as_const();
        let Some(counts) = whole.and_then(|s| fit.parts.counts(s as usize)) else {
            return Some(format!(
                "axis {axis} of {shape} does not divide by {parts:?}"
            ));
        };
        if at + 1 == shape.0.len()
            && let Op::Row(r) = &n.op
        {
            for s in &r.0 {
                if let Stage::Hadamard { block } = *s
                    && counts.iter().any(|c| !c.is_multiple_of(block as usize))
                {
                    return Some(format!("axis {axis} of {shape} cuts hadamard {block}"));
                }
            }
        }
        let Op::Weight(w) = n.op else {
            continue;
        };
        let size = |d| poly(&g.dims, d).ok()?.as_const().map(|s| s as usize);
        let Some(sizes) = shape.0.iter().map(size).collect::<Option<Vec<usize>>>() else {
            continue;
        };
        if counts.iter().any(|c| !(fit.shares)(w, &sizes, at, *c)) {
            return Some(format!(
                "axis {axis} of {shape} over {parts:?} cuts its form's blocks"
            ));
        }
    }
    None
}

/// A matmul of a whole weight, read by an add.
fn returns(g: &Graph, id: NodeId) -> bool {
    let node = g.node(id);
    node.op == Op::Matmul && whole_weight(g, node.args[1]) && summed(g, id)
}

/// Whether an add reads `id`, through scales or gathers.
fn summed(g: &Graph, id: NodeId) -> bool {
    let readers = (0..g.nodes.len())
        .map(|i| NodeId(i as u32))
        .filter(|n| g.node(*n).args.contains(&id));
    for reader in readers {
        let node = g.node(reader);
        match &node.op {
            Op::Select(Select::Gather) if node.args[0] == id && summed(g, reader) => return true,
            Op::Row(r) if r.0.contains(&Stage::Add) => return true,
            Op::Row(r) if r.0.iter().all(|s| matches!(s, Stage::Scale(_))) && summed(g, reader) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Replicated weight of rank `rank` gathered by `id`.
fn gathered(g: &Graph, id: NodeId, rank: usize) -> Option<u32> {
    if g.node(id).op != Op::Select(Select::Gather) {
        return None;
    }
    let Op::Weight(w) = g.node(set_of(g, g.node(id).args[0])).op else {
        return None;
    };
    let ty = &g.weights[w as usize].ty;
    (ty.rank() == rank && ty.placement == Placement::Replicated).then_some(w)
}

/// The routed down matmul feeding `id`, through linear stages.
fn down_matmul(g: &Graph, mut id: NodeId) -> Option<(NodeId, u32)> {
    loop {
        let node = g.node(id);
        match &node.op {
            Op::Matmul => return gathered(g, node.args[1], 3).map(|w| (id, w)),
            Op::Row(r)
                if r.0.iter().all(|s| {
                    matches!(s, Stage::Hadamard { .. } | Stage::Mul | Stage::Scale(_))
                }) =>
            {
                id = node.args[0];
            }
            _ => return None,
        }
    }
}

/// Splits wide routed experts along their inner width.
fn split_moe(g: &Graph, id: NodeId, parts: &Parts) -> Option<Graph> {
    if !matches!(g.node(id).op, Op::Mix(_)) || g.node(id).ty.placement != Placement::Replicated {
        return None;
    }
    let (down, down_w) = down_matmul(g, g.node(id).args[0])?;
    let whole = poly(&g.dims, &g.weights[down_w as usize].ty.shape.0[2])
        .ok()?
        .as_const()? as usize;
    let counts = parts.counts(whole)?;
    if counts.iter().any(|&c| c < 256) {
        return None;
    }
    let mut out = g.clone();
    let mut pending = vec![g.node(down).args[0]];
    let mut seen = vec![false; g.nodes.len()];
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut seen[n.0 as usize], true) {
            continue;
        }
        let node = g.node(n);
        if node.op == Op::Matmul
            && let Some(w) = gathered(g, node.args[1], 3)
        {
            out.weights[w as usize].ty.placement = Placement::Split { axis: 1 };
            continue;
        }
        if let Some(w) = gathered(g, n, 2) {
            out.weights[w as usize].ty.placement = Placement::Split { axis: 1 };
            continue;
        }
        pending.extend(node.args.iter().copied());
    }
    out.weights[down_w as usize].ty.placement = Placement::Split { axis: 2 };
    let sum = Node {
        op: Op::Row(Row(vec![Stage::AllReduce])),
        args: vec![id],
        ty: g.node(id).ty.clone(),
        numerics: Some(Numerics::default()),
    };
    insert(&mut out, id, sum);
    Some(out)
}

/// Every gathered set split by matrix, each mix summed back.
fn route(g: &Graph) -> Graph {
    let mut out = g.clone();
    let mut changed = false;
    for n in &g.nodes {
        if n.op != Op::Select(Select::Gather) {
            continue;
        }
        let Op::Weight(w) = g.node(set_of(g, n.args[0])).op else {
            continue;
        };
        let ty = &g.weights[w as usize].ty;
        if ty.rank() == 3 && ty.placement == Placement::Replicated {
            out.weights[w as usize].ty.placement = Placement::Split { axis: 0 };
            changed = true;
        }
    }
    if !changed {
        return out;
    }
    let mixes: Vec<NodeId> = (0..g.nodes.len())
        .map(|i| NodeId(i as u32))
        .filter(|&id| {
            matches!(g.node(id).op, Op::Mix(_))
                && g.node(id).ty.placement != Placement::Partial
                && down_matmul(g, g.node(id).args[0]).is_some()
        })
        .collect();
    for id in mixes.into_iter().rev() {
        let sum = Node {
            op: Op::Row(Row(vec![Stage::AllReduce])),
            args: vec![id],
            ty: out.node(id).ty.clone(),
            numerics: Some(Numerics::default()),
        };
        insert(&mut out, id, sum);
    }
    out
}

/// Returned matmuls of lone weights split by rows, then gather.
fn gather(g: &Graph) -> Graph {
    let mut out = g.clone();
    let mut at = 0;
    while at < out.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        let node = out.node(id);
        let returned = (out.entries.iter().flat_map(|e| &e.outputs)).any(|o| o.node == id);
        let weight = node.args.get(1).copied();
        let Some(weight) = weight.filter(|_| returned && node.op == Op::Matmul) else {
            continue;
        };
        if !whole_weight(&out, weight) || !lone(&out, weight, id) {
            continue;
        }
        place(&mut out, weight, 0);
        let joined = Node {
            op: Op::Row(Row(vec![Stage::AllGather])),
            args: vec![id],
            ty: out.node(id).ty.clone(),
            numerics: Some(Numerics::default()),
        };
        insert(&mut out, id, joined);
        leave_region(&mut out, id);
        at += 1;
    }
    out
}

/// Whether only `reader` reads the weight at `weight`.
fn lone(g: &Graph, weight: NodeId, reader: NodeId) -> bool {
    let w = &g.node(weight).op;
    let others = (g.nodes.iter().enumerate()).filter(|(i, _)| *i != reader.0 as usize);
    !others
        .flat_map(|(_, n)| &n.args)
        .any(|a| g.node(*a).op == *w)
}

/// Whether `id` is a sum that only `reader` reads.
fn lone_sum(g: &Graph, id: NodeId, reader: NodeId) -> bool {
    let sum = Op::Row(Row(vec![Stage::AllReduce]));
    let others = (g.nodes.iter().enumerate())
        .filter(|(i, n)| *i != reader.0 as usize && n.args.contains(&id));
    g.node(id).op == sum && others.count() == 0
}

/// Two sums an add reads become one, after any region.
fn merge(g: &Graph) -> Graph {
    let mut g = g.clone();
    let add = Op::Row(Row(vec![Stage::Add]));
    let mut at = 0;
    while at < g.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        let node = g.node(id);
        let [a, b] = node.args[..] else {
            continue;
        };
        if node.op != add || !lone_sum(&g, a, id) || !lone_sum(&g, b, id) || a == b {
            continue;
        }
        let parts = [g.node(a).args[0], g.node(b).args[0]];
        g.nodes[at - 1].args = parts.to_vec();
        let sum = Node {
            op: Op::Row(Row(vec![Stage::AllReduce])),
            args: vec![id],
            ty: g.node(id).ty.clone(),
            numerics: Some(Numerics::default()),
        };
        insert(&mut g, id, sum);
        leave_region(&mut g, id);
        at += 1;
    }
    g
}

/// Keeps the node after `id` outside regions ending there.
fn leave_region(g: &mut Graph, id: NodeId) {
    let new = NodeId(id.0 + 1);
    for r in &mut g.regions {
        if r.outputs == [new] && r.end == NodeId(new.0 + 1) {
            r.outputs = vec![id];
            r.end = new;
        }
    }
}

/// The node under any slices of `id`.
fn set_of(g: &Graph, mut id: NodeId) -> NodeId {
    while let Op::View(crate::node::View::Slice { .. }) = &g.node(id).op {
        id = g.node(id).args[0];
    }
    id
}

/// A rank-two weight that earlier splits left unclaimed.
fn whole_weight(g: &Graph, id: NodeId) -> bool {
    let Op::Weight(w) = g.node(id).op else {
        return false;
    };
    let ty = &g.weights[w as usize].ty;
    ty.rank() == 2 && ty.placement == Placement::Replicated
}

/// `g` with `id` split over rows, its feeders over columns.
fn split(g: &Graph, id: NodeId) -> Graph {
    let mut out = g.clone();
    let mut pending = vec![g.node(id).args[0]];
    let mut seen = vec![false; g.nodes.len()];
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut seen[n.0 as usize], true) {
            continue;
        }
        let node = g.node(n);
        if node.op == Op::Matmul && whole_weight(g, node.args[1]) {
            place(&mut out, node.args[1], 0);
            continue;
        }
        // A weight read through views splits with its readers.
        if whole_weight(g, n) {
            place(&mut out, n, 0);
            continue;
        }
        pending.extend(node.args.iter().copied());
    }
    place(&mut out, g.node(id).args[1], 1);
    let sum = Node {
        op: Op::Row(Row(vec![Stage::AllReduce])),
        args: vec![id],
        ty: g.node(id).ty.clone(),
        numerics: Some(Numerics::default()),
    };
    insert(&mut out, id, sum);
    out
}

fn place(g: &mut Graph, weight: NodeId, axis: u32) {
    let Op::Weight(w) = g.node(weight).op else {
        unreachable!("checked by whole_weight")
    };
    g.weights[w as usize].ty.placement = Placement::Split { axis };
}

/// Puts `node` at `new`. Later nodes move up by one.
pub(crate) fn open(g: &mut Graph, new: NodeId, node: Node) {
    let shift = |n: &mut NodeId| {
        if *n >= new {
            n.0 += 1;
        }
    };
    for n in &mut g.nodes[new.0 as usize..] {
        n.args.iter_mut().for_each(shift);
    }
    g.nodes.insert(new.0 as usize, node);
    for e in &mut g.entries {
        e.outputs.iter_mut().for_each(|o| shift(&mut o.node));
    }
    for r in &mut g.regions {
        shift(&mut r.start);
        // The end is past the last, so it moves.
        shift(&mut r.end);
        r.inputs.iter_mut().chain(&mut r.outputs).for_each(shift);
    }
}

/// Puts `node` after `at`. Reads of `at` move to it.
fn insert(g: &mut Graph, at: NodeId, node: Node) {
    let new = NodeId(at.0 + 1);
    open(g, new, node);
    let redirect = |n: &mut NodeId| {
        if *n == at {
            *n = new;
        }
    };
    for n in &mut g.nodes[new.0 as usize + 1..] {
        n.args.iter_mut().for_each(redirect);
    }
    for e in &mut g.entries {
        e.outputs.iter_mut().for_each(|o| redirect(&mut o.node));
    }
    for r in &mut g.regions {
        r.inputs.iter_mut().chain(&mut r.outputs).for_each(redirect);
    }
}

/// The first op that fails to retype, if any.
fn retype_at(g: &Graph) -> Result<(), usize> {
    typed(g).map(|_| ()).map_err(|(at, _)| at)
}

/// Retypes every op in order. Kept rows follow their appends.
fn retype(g: &Graph) -> Result<Graph, crate::infer::TypeError> {
    typed(g).map_err(|(_, e)| e)
}

/// `retype`, with the index of the op that failed.
fn typed(g: &Graph) -> Result<Graph, (usize, crate::infer::TypeError)> {
    let mut out = g.clone();
    for i in 0..out.nodes.len() {
        retype_node(&mut out, i).map_err(|e| (i, e))?;
    }
    Ok(out)
}

/// Retypes op `i` from its arguments' types.
pub(crate) fn retype_node(out: &mut Graph, i: usize) -> Result<(), crate::infer::TypeError> {
    if let Op::Append(k) = out.nodes[i].op {
        let rows = out.node(out.nodes[i].args[0]).ty.placement;
        out.kept[k as usize].placement = match rows {
            Placement::Split { axis } => Placement::Split { axis: axis + 1 },
            // Kept rows never follow a share of the rows.
            Placement::Owned { .. } => out.kept[k as usize].placement,
            p => p,
        };
    } else if let Op::Delta(d) = out.nodes[i].op
        && let Some(k) = d.state
    {
        out.kept[k as usize].placement = out.node(out.nodes[i].args[2]).ty.placement;
    }
    let args: Vec<_> = out.nodes[i]
        .args
        .iter()
        .map(|a| &out.nodes[a.0 as usize].ty)
        .collect();
    out.nodes[i].ty = infer(out, &out.nodes[i].op, &args)?;
    Ok(())
}
