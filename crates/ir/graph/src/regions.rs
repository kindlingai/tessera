//! Checks that composite regions are sealed and nest.

use crate::infer::{ErrorKind, TypeError};
use crate::node::{Graph, NodeId, Region};

fn inside(r: &Region, id: NodeId) -> bool {
    r.start <= id && id < r.end
}

fn fail(i: usize, msg: String) -> Result<(), TypeError> {
    Err(TypeError {
        kind: ErrorKind::Region,
        message: format!("region {i}: {msg}"),
    })
}

/// Checks each region's range, inputs, outputs and nesting.
pub(crate) fn check(g: &Graph) -> Result<(), TypeError> {
    for (i, r) in g.regions.iter().enumerate() {
        if r.start >= r.end || r.end.0 as usize > g.nodes.len() {
            return fail(i, format!("bad range {}..{}", r.start.0, r.end.0));
        }
        if r.inputs != g.reads(r.start, r.end) {
            return fail(i, format!("inputs {:?} differ from its reads", r.inputs));
        }
        if r.outputs.is_empty() || !r.outputs.iter().all(|&o| inside(r, o)) {
            return fail(i, format!("outputs {:?} must be inside it", r.outputs));
        }
        let later = g.nodes[r.end.0 as usize..].iter().flat_map(|n| &n.args);
        let entries = g
            .entries
            .iter()
            .flat_map(|e| e.outputs.iter().map(|o| &o.node));
        if let Some(leak) = later
            .chain(entries)
            .find(|&&a| inside(r, a) && !r.outputs.contains(&a))
        {
            return fail(i, format!("node {} leaks out", leak.0));
        }
        for (j, o) in g.regions[..i].iter().enumerate() {
            let apart = o.end <= r.start || r.end <= o.start;
            let nested =
                (o.start <= r.start && r.end <= o.end) || (r.start <= o.start && o.end <= r.end);
            let same = o.start == r.start && o.end == r.end;
            if !apart && (!nested || same) {
                return fail(i, format!("overlaps region {j} without nesting"));
            }
        }
    }
    Ok(())
}
