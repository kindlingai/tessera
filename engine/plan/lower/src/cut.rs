//! Cuts nodes out of a graph, to check one rule.

use graph::{Entry, EntryKind, Graph, Node, NodeId, Op, Output, Port, Role};

/// A graph of `nodes` alone. Their reads become its inputs.
pub fn cut(g: &Graph, nodes: &[NodeId]) -> Graph {
    let mut out = Graph::new(g.dims.clone());
    out.kept.clone_from(&g.kept);
    let reads: Vec<NodeId> = nodes
        .iter()
        .flat_map(|n| &g.node(*n).args)
        .filter(|a| !nodes.contains(a))
        .fold(Vec::new(), |mut seen, a| {
            if !seen.contains(a) {
                seen.push(*a);
            }
            seen
        });
    let port = |id: &NodeId| Port {
        role: Role::Tokens,
        ty: g.node(*id).ty.clone(),
    };
    let inputs: Vec<Port> = reads.iter().map(port).collect();
    for (at, p) in inputs.iter().enumerate() {
        out.nodes.push(Node {
            op: Op::Input {
                entry: 0,
                port: at as u32,
            },
            args: Vec::new(),
            ty: p.ty.clone(),
            numerics: None,
        });
    }
    // Where each old node now sits.
    let moved = |old: &NodeId| {
        let at = reads.iter().position(|r| r == old);
        let at = at.or_else(|| Some(reads.len() + nodes.iter().position(|n| n == old)?));
        NodeId(at.expect("an input or a cut node") as u32)
    };
    for id in nodes {
        let mut node = g.node(*id).clone();
        node.args = node.args.iter().map(moved).collect();
        out.nodes.push(node);
    }
    let last = NodeId(out.nodes.len() as u32 - 1);
    out.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs,
        outputs: vec![Output {
            role: Role::Logits,
            node: last,
        }],
    });
    out
}
