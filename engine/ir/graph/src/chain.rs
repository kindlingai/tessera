//! Numbers a model's layers in the order they first run.

use crate::node::{Composite, Graph, Op, Region};
use std::collections::BTreeMap;

/// The weights nodes inside `r` read, sorted.
fn read_weights(g: &Graph, r: &Region) -> Vec<u32> {
    let mut out = Vec::new();
    for n in &g.nodes[r.start.0 as usize..r.end.0 as usize] {
        for a in &n.args {
            if let Op::Weight(w) = g.node(*a).op {
                out.push(w);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Each outermost layer region and its number, by weights read.
pub fn layers(g: &Graph) -> Vec<(u32, usize)> {
    let mut found = Vec::new();
    for (i, r) in g.regions.iter().enumerate() {
        let nested = |o: &Region| {
            o.kind == Composite::Layer && o != r && o.start <= r.start && r.end <= o.end
        };
        if r.kind == Composite::Layer && !g.regions.iter().any(nested) {
            found.push(i);
        }
    }
    found.sort_by_key(|&i| g.regions[i].start);
    let mut seen: BTreeMap<Vec<u32>, u32> = BTreeMap::new();
    let mut out = Vec::with_capacity(found.len());
    for i in found {
        let next = seen.len() as u32;
        let k = *seen.entry(read_weights(g, &g.regions[i])).or_insert(next);
        out.push((k, i));
    }
    out
}
