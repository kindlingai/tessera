//! Row ownership: each rank keeps its share of summed rows.

use crate::infer::TypeError;
use crate::node::{Composite, Graph, Node, NodeId, Op, Region, Row};
use crate::numerics::Numerics;
use crate::shard::{open, retype_node};
use crate::stages::Stage;
use crate::types::Placement;
use crate::verify::{VerifyError, verify};

fn gather_op() -> Op {
    Op::Row(Row(vec![Stage::GatherRows]))
}

fn owned(g: &Graph, id: NodeId) -> bool {
    g.node(id).ty.placement == Placement::Owned
}

/// A gather of `from`: one before `before`, else new.
fn gathered(g: &mut Graph, from: NodeId, at: usize, before: usize) -> (NodeId, bool) {
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
        numerics: Some(Numerics::default()),
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

/// Whether a region's ops split their rows among ranks.
fn splits(r: &Region) -> bool {
    matches!(r.kind, Composite::Moe | Composite::Mlp)
}

/// Gathers owned inputs of split regions starting at `at`.
fn enter_regions(g: &mut Graph, at: usize) -> bool {
    let starts = |g: &Graph, at: usize| {
        let here = move |r: &&Region| r.start.0 as usize == at && splits(r);
        let regions = g.regions.iter().filter(here);
        regions
            .flat_map(|r| r.inputs.clone())
            .find(|i| owned(g, *i))
    };
    let Some(from) = starts(g, at) else {
        return false;
    };
    let (whole, new) = gathered(g, from, at, at);
    let first = at + usize::from(new);
    for r in 0..g.regions.len() {
        if g.regions[r].start.0 as usize != first || !splits(&g.regions[r]) {
            continue;
        }
        let end = g.regions[r].end.0 as usize;
        for n in &mut g.nodes[first..end] {
            n.args
                .iter_mut()
                .filter(|a| **a == from)
                .for_each(|a| *a = whole);
        }
        g.regions[r].inputs = g.reads(NodeId(first as u32), NodeId(end as u32));
    }
    true
}

/// Sums over rows scatter. Readers of all rows gather.
pub fn own_rows(g: &Graph) -> Result<Graph, VerifyError> {
    let mut out = g.clone();
    for n in &mut out.nodes {
        if let Op::Row(r) = &mut n.op
            && n.ty.rank() == 2
        {
            for s in r.0.iter_mut().filter(|s| **s == Stage::AllReduce) {
                *s = Stage::ReduceScatter;
            }
        }
    }
    let mut at = 0;
    while at < out.nodes.len() {
        if enter_regions(&mut out, at) {
            continue;
        }
        let Err(error) = retype_node(&mut out, at) else {
            at += 1;
            continue;
        };
        let args = out.nodes[at].args.clone();
        let Some(from) = args.into_iter().find(|a| owned(&out, *a)) else {
            return Err(failed(at, error));
        };
        let (whole, new) = gathered(&mut out, from, at, at);
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
            let (whole, new) = gathered(&mut out, from, from.0 as usize + 1, all);
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
