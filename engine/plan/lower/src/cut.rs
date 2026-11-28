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

const UNSET: u32 = u32::MAX;
const IN_CUT: u32 = u32::MAX - 1;

/// Maps original node ids into the carved graph.
enum Map<'a> {
    Small {
        reads: &'a [NodeId],
        nodes: &'a [NodeId],
    },
    Dense {
        slots: Vec<u32>,
        base: u32,
    },
}

impl Map<'_> {
    fn moved(&self, old: &NodeId) -> NodeId {
        let at = match self {
            Map::Small { reads, nodes } => reads
                .iter()
                .position(|r| r == old)
                .or_else(|| Some(reads.len() + nodes.iter().position(|n| n == old)?)),
            Map::Dense { slots, .. } => slots
                .get(old.0 as usize)
                .copied()
                .filter(|&v| v != UNSET)
                .map(|v| v as usize),
        };
        NodeId(at.expect("an input or a cut node") as u32)
    }

    fn in_cut(&self, id: &NodeId) -> bool {
        match self {
            Map::Small { nodes, .. } => nodes.contains(id),
            Map::Dense { slots, base } => slots
                .get(id.0 as usize)
                .copied()
                .is_some_and(|v| v != UNSET && v >= *base),
        }
    }
}

/// `nodes` as a graph. `keep` keeps weights, roles and regions.
fn carve(g: &Graph, nodes: &[NodeId], outputs: &[NodeId], keep: bool) -> Graph {
    let mut out = Graph::new(g.dims.clone());
    out.kept.clone_from(&g.kept);
    let mut reads: Vec<NodeId> = Vec::new();
    let dense = nodes.len() > 16;
    let mut slots = if dense {
        let mut s = vec![UNSET; g.nodes.len()];
        for &n in nodes {
            s[n.0 as usize] = IN_CUT;
        }
        s
    } else {
        Vec::new()
    };
    for n in nodes {
        for a in &g.node(*n).args {
            if dense {
                let slot = &mut slots[a.0 as usize];
                if *slot == UNSET {
                    *slot = reads.len() as u32;
                    reads.push(*a);
                }
            } else if !nodes.contains(a) && !reads.contains(a) {
                reads.push(*a);
            }
        }
    }
    let map = if dense {
        let base = reads.len() as u32;
        for (i, &n) in nodes.iter().enumerate() {
            let slot = &mut slots[n.0 as usize];
            if *slot == IN_CUT {
                *slot = base + i as u32;
            }
        }
        Map::Dense { slots, base }
    } else {
        Map::Small {
            reads: &reads,
            nodes,
        }
    };
    out.nodes.reserve(reads.len() + nodes.len());
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
    let moved = |old: &NodeId| map.moved(old);
    for id in nodes {
        let mut node = g.node(*id).clone();
        for arg in &mut node.args {
            *arg = moved(arg);
        }
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
    let mut out_list: Vec<Output> = outputs
        .iter()
        .map(|o| Output {
            role: Role::Logits,
            node: moved(o),
        })
        .collect();
    if keep && let Some(entry) = g.entries.first() {
        for o in &entry.outputs {
            if matches!(o.role, Role::Calib(_)) && map.in_cut(&o.node) {
                out_list.push(Output {
                    role: o.role,
                    node: moved(&o.node),
                });
            }
        }
    }
    out.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs,
        outputs: out_list,
    });
    out
}
