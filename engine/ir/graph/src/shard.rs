//! Tensor parallel splits matmuls. Expert parallel splits sets.

use crate::infer::{infer, poly};
use crate::node::{Graph, Node, NodeId, Op, Row, Select, Stage};
use crate::numerics::Numerics;
use crate::parts::Parts;
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
    let mut out = g.clone();
    let mut whole = Vec::new();
    let mut at = 0;
    while at < out.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        if !returns(&out, id) {
            continue;
        }
        let tried = split(&out, id);
        // The sum is the new node behind this one.
        if adopt(&mut out, tried, parts, &mut whole) {
            at += 1;
        }
    }
    let routed = route(&out);
    adopt(&mut out, routed, parts, &mut whole);
    if let Ok(merged) = retype(&merge(&out)) {
        out = merged;
    }
    verify(&out)?;
    Ok(Sharded { graph: out, whole })
}

/// Takes `tried` if it types and fits `parts`.
fn adopt(out: &mut Graph, tried: Graph, parts: &Parts, whole: &mut Vec<Whole>) -> bool {
    let why = match retype(&tried) {
        Ok(typed) => match misfit(&typed, parts) {
            None => {
                *out = typed;
                return true;
            }
            Some(why) => why,
        },
        Err(e) => e.to_string(),
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

/// The first split axis `parts` cannot divide.
fn misfit(g: &Graph, parts: &Parts) -> Option<String> {
    for n in &g.nodes {
        let Placement::Split { axis } = n.ty.placement else {
            continue;
        };
        let size = poly(&g.dims, &n.ty.shape.0[axis as usize]).ok()?;
        if size
            .as_const()
            .and_then(|s| parts.counts(s as usize))
            .is_none()
        {
            let (shape, parts) = (&n.ty.shape, &parts.0);
            return Some(format!(
                "axis {axis} of {shape} does not divide by {parts:?}"
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

/// Whether an add reads `id`, through scales at most.
fn summed(g: &Graph, id: NodeId) -> bool {
    let readers = (0..g.nodes.len())
        .map(|i| NodeId(i as u32))
        .filter(|n| g.node(*n).args.contains(&id));
    for reader in readers {
        let Op::Row(r) = &g.node(reader).op else {
            continue;
        };
        if r.0.contains(&Stage::Add) {
            return true;
        }
        let scales = r.0.iter().all(|s| matches!(s, Stage::Scale(_)));
        if scales && summed(g, reader) {
            return true;
        }
    }
    false
}

/// Every gathered set split by matrix, each mix summed back.
fn route(g: &Graph) -> Graph {
    let mut out = g.clone();
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
        }
    }
    let mut at = 0;
    while at < out.nodes.len() {
        let id = NodeId(at as u32);
        at += 1;
        if let Op::Mix(_) = out.node(id).op {
            let sum = Node {
                op: Op::Row(Row(vec![Stage::AllReduce])),
                args: vec![id],
                ty: out.node(id).ty.clone(),
                numerics: Some(Numerics::default()),
            };
            insert(&mut out, id, sum);
            at += 1;
        }
    }
    out
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
        let new = NodeId(id.0 + 1);
        for r in &mut g.regions {
            if r.outputs == [new] && r.end == NodeId(new.0 + 1) {
                r.outputs = vec![id];
                r.end = new;
            }
        }
        at += 1;
    }
    g
}

/// The node under any slices of `id`.
fn set_of(g: &Graph, mut id: NodeId) -> NodeId {
    while let Op::View(crate::node::View::Slice { .. }) = &g.node(id).op {
        id = g.node(id).args[0];
    }
    id
}

/// A rank-two weight no earlier split has claimed.
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

/// Puts `node` after `at`. Reads of `at` move to it.
fn insert(g: &mut Graph, at: NodeId, node: Node) {
    let new = NodeId(at.0 + 1);
    let shift = |n: &mut NodeId| {
        if *n >= new {
            n.0 += 1;
        }
    };
    let redirect = |n: &mut NodeId| {
        if *n == at {
            *n = new;
        }
    };
    for n in &mut g.nodes[new.0 as usize..] {
        n.args.iter_mut().for_each(|a| {
            shift(a);
            redirect(a);
        });
    }
    g.nodes.insert(new.0 as usize, node);
    for e in &mut g.entries {
        e.outputs.iter_mut().for_each(|o| {
            shift(&mut o.node);
            redirect(&mut o.node);
        });
    }
    for r in &mut g.regions {
        shift(&mut r.start);
        // The end is past the last, so it moves.
        shift(&mut r.end);
        for n in r.inputs.iter_mut().chain(&mut r.outputs) {
            shift(n);
            redirect(n);
        }
    }
}

/// Retypes every op in order. Kept rows follow their appends.
fn retype(g: &Graph) -> Result<Graph, crate::infer::TypeError> {
    let mut out = g.clone();
    for i in 0..out.nodes.len() {
        if let Op::Append(k) = out.nodes[i].op {
            let rows = out.node(out.nodes[i].args[0]).ty.placement;
            out.kept[k as usize].placement = match rows {
                Placement::Split { axis } => Placement::Split { axis: axis + 1 },
                p => p,
            };
        }
        let args: Vec<_> = out.nodes[i]
            .args
            .iter()
            .map(|a| &out.nodes[a.0 as usize].ty)
            .collect();
        let ty = infer(&out, &out.nodes[i].op, &args)?;
        out.nodes[i].ty = ty;
    }
    Ok(out)
}
