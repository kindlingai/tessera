//! The front end's graph, its sharded form, and their link.

use std::error::Error;
use tessera_graph::{DimName, EntryKind, Graph, NodeId, Op, Parts, Row, Size, Stage, Whole, shard};
use tessera_models::preset;

/// A graph before and after the ranks split it.
pub struct Model {
    pub name: String,
    pub source: Graph,
    pub sharded: Graph,
    /// Weights the split left whole, and why.
    pub whole: Vec<Whole>,
    /// Per sharded node, the source node it computes or sums.
    pub origin: Vec<NodeId>,
    /// Per sharded node, whether sharding added it.
    pub added: Vec<bool>,
}

/// One step graph by name. `full` takes the released sizes.
pub fn named(name: &str, full: bool) -> Result<Graph, String> {
    let p = preset::named(name)?;
    let graph = match full {
        true => p.step.map(|step| step(4096)),
        false => p.small.map(|small| small()),
    };
    graph.ok_or_else(|| format!("`{name}` has no graph at these sizes"))
}

/// A graph from its JSON form.
pub fn read(path: &str) -> Result<Graph, Box<dyn Error>> {
    Ok(Graph::from_json(&std::fs::read_to_string(path)?)?)
}

fn sum() -> Op {
    Op::Row(Row(vec![Stage::AllReduce]))
}

impl Model {
    pub fn new(name: String, source: Graph, parts: &Parts) -> Result<Self, Box<dyn Error>> {
        let tessera_graph::Sharded {
            graph: sharded,
            whole,
        } = shard(&source, parts)?;
        // With no transport, sums carry F32 as host sums do.
        let sharded = sharded.wired(tessera_graph::DType::F32);
        let (mut origin, mut added) = (Vec::new(), Vec::new());
        let mut at = 0;
        for n in &sharded.nodes {
            let same = source.nodes.get(at).is_some_and(|s| s.op == n.op);
            if n.op == sum() && !same {
                origin.push(origin[n.args[0].0 as usize]);
                added.push(true);
                continue;
            }
            origin.push(NodeId(at as u32));
            added.push(false);
            at += 1;
        }
        Ok(Self {
            name,
            source,
            sharded,
            whole,
            origin,
            added,
        })
    }

    pub fn step_entry(&self) -> bool {
        self.sharded.entries[0].kind == EntryKind::Step
    }

    /// Sizes for one launch of `rows` new rows at `context`.
    pub fn sizes(&self, rows: u64, context: u64) -> Vec<(DimName, u64)> {
        let mut out = Vec::new();
        for d in &self.sharded.dims {
            let Size::Bounded(max) = d.size else {
                continue;
            };
            let size = match d.name {
                DimName::Seqs | DimName::Segments => 1,
                DimName::Rows => rows.next_power_of_two(),
                DimName::Context => context,
                _ => max,
            };
            out.push((d.name, size));
        }
        out
    }
}
