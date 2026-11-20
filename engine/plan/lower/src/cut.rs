//! Cuts nodes, or one region, out of a graph.

use tessera_graph::{Entry, EntryKind, Graph, Node, NodeId, Op, Output, Port, Region, Role};

/// A graph of `nodes` alone. Their reads become its inputs.
pub fn cut(g: &Graph, nodes: &[NodeId]) -> Graph {
    let last = *nodes.last().expect("nodes to cut");
    carve(g, nodes, &[last], false)
}

/// Region `r` alone, with its weights, roles and inner regions.
pub fn cut_region(g: &Graph, r: &Region) -> Graph {
    let nodes: Vec<NodeId> = (r.start.0..r.end.0).map(NodeId).collect();
    carve(g, &nodes, &r.outputs, true)
}

/// `nodes` as a graph. `keep` keeps weights, roles and regions.
fn carve(g: &Graph, nodes: &[NodeId], outputs: &[NodeId], keep: bool) -> Graph {
    let mut out = Graph::new(g.dims.clone());
    out.kept.clone_from(&g.kept);
    let mut reads: Vec<NodeId> = Vec::new();
    for n in nodes {
        for a in &g.node(*n).args {
            if !nodes.contains(a) && !reads.contains(a) {
                reads.push(*a);
            }
        }
    }
    let mut inputs: Vec<Port> = Vec::new();
    for id in &reads {
        let node = g.node(*id);
        let op = match node.op {
            Op::Weight(w) if keep => {
                out.weights.push(g.weights[w as usize].clone());
                Op::Weight(out.weights.len() as u32 - 1)
            }
            _ => {
                let role = match node.op {
                    Op::Input { entry, port } if keep => {
                        g.entries[entry as usize].inputs[port as usize].role
                    }
                    _ => Role::Tokens,
                };
                inputs.push(Port {
                    role,
                    ty: node.ty.clone(),
                });
                Op::Input {
                    entry: 0,
                    port: inputs.len() as u32 - 1,
                }
            }
        };
        out.nodes.push(Node {
            op,
            args: Vec::new(),
            ty: node.ty.clone(),
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
    let (first, last) = (nodes[0], nodes[nodes.len() - 1]);
    for r in &g.regions {
        if !keep || r.start < first || r.end.0 > last.0 + 1 {
            continue;
        }
        out.regions.push(Region {
            kind: r.kind,
            start: moved(&r.start),
            end: NodeId(moved(&NodeId(r.end.0 - 1)).0 + 1),
            inputs: r.inputs.iter().map(moved).collect(),
            outputs: r.outputs.iter().map(moved).collect(),
        });
    }
    let outputs = outputs.iter().map(|o| Output {
        role: Role::Logits,
        node: moved(o),
    });
    out.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs,
        outputs: outputs.collect(),
    });
    out
}
