//! Cuts nodes out of a graph, to check one rule.

use graph::{Entry, EntryKind, Graph, Node, NodeId, Op, Output, Port, Role};

/// Cuts nodes, keeping weights and exposing other reads as inputs.
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
    let mut inputs = Vec::new();
    for id in &reads {
        let ty = g.node(*id).ty.clone();
        let op = match g.node(*id).op {
            Op::Weight(w) => {
                let at = out.weights.len() as u32;
                out.weights.push(g.weights[w as usize].clone());
                Op::Weight(at)
            }
            _ => {
                let port = inputs.len() as u32;
                inputs.push(Port {
                    role: Role::Tokens,
                    ty: ty.clone(),
                });
                Op::Input { entry: 0, port }
            }
        };
        out.nodes.push(Node {
            op,
            args: Vec::new(),
            ty,
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
