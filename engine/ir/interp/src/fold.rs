//! Weight-only values, computed once and kept as weights.

use crate::run::run;
use crate::tensor::Error;
use tessera_graph::{
    DType, Entry, EntryKind, Graph, NodeId, Op, Output, PathSeg, Role, View, Weight, verify,
};
use tessera_mem::Tensor;

/// Whether each node is fixed-size floats from weights.
fn fixed(g: &Graph) -> Vec<bool> {
    let mut fixed = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        let sized = n.ty.shape.elements(&g.dims).ok().and_then(|p| p.as_const());
        fixed[i] = sized.is_some()
            && n.ty.dtype == DType::F32
            && match &n.op {
                Op::Weight(_) => true,
                op if op.stateful() => false,
                _ => !n.args.is_empty() && n.args.iter().all(|a| fixed[a.0 as usize]),
            };
    }
    fixed
}

/// Whether each node feeds some entry's output.
fn live(g: &Graph) -> Vec<bool> {
    let mut live = vec![false; g.nodes.len()];
    for o in g.entries.iter().flat_map(|e| &e.outputs) {
        live[o.node.0 as usize] = true;
    }
    for i in (0..g.nodes.len()).rev() {
        if live[i] {
            for a in &g.nodes[i].args {
                live[a.0 as usize] = true;
            }
        }
    }
    live
}

/// Weight-only nodes that runtime nodes read, or outputs.
pub(crate) fn frontier(g: &Graph) -> Vec<NodeId> {
    let (fixed, live) = (fixed(g), live(g));
    let mut wanted = vec![false; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if live[i] && !fixed[i] {
            for a in &n.args {
                wanted[a.0 as usize] = true;
            }
        }
    }
    for o in g.entries.iter().flat_map(|e| &e.outputs) {
        wanted[o.node.0 as usize] = true;
    }
    // A reshape costs nothing, so its reader keeps the view.
    for i in (0..g.nodes.len()).rev() {
        let n = &g.nodes[i];
        if wanted[i] && fixed[i] && matches!(n.op, Op::View(View::Reshape(_))) {
            wanted[i] = false;
            wanted[n.args[0].0 as usize] = true;
        }
    }
    let computed = |i: &usize| fixed[*i] && wanted[*i];
    let ops = (0..g.nodes.len()).filter(computed);
    let ops = ops.filter(|i| !matches!(g.nodes[*i].op, Op::Weight(_)));
    ops.map(|i| NodeId(i as u32)).collect()
}

/// `g` with its weight-only values computed once, as new weights.
pub fn fold(g: &Graph, weights: &[Tensor]) -> Result<(Graph, Vec<Tensor>), Error> {
    let cut = frontier(g);
    if cut.is_empty() {
        return Ok((g.clone(), weights.to_vec()));
    }
    let mut probe = g.clone();
    let outputs = cut.iter().map(|&node| Output {
        role: Role::Logits,
        node,
    });
    probe.entries.push(Entry {
        kind: EntryKind::Forward,
        inputs: Vec::new(),
        outputs: outputs.collect(),
    });
    let values = run(&probe, weights, probe.entries.len() - 1, &[])?;
    let (mut out, mut all) = (g.clone(), weights.to_vec());
    for (&at, value) in cut.iter().zip(values) {
        let node = &mut out.nodes[at.0 as usize];
        out.weights.push(Weight {
            path: vec![PathSeg::Field("folded".into()), PathSeg::Index(at.0)],
            ty: node.ty.clone(),
        });
        node.op = Op::Weight(out.weights.len() as u32 - 1);
        node.args.clear();
        node.numerics = None;
        all.push(value);
    }
    let reads: Vec<_> = (out.regions.iter())
        .map(|r| out.reads(r.start, r.end))
        .collect();
    for (r, inputs) in out.regions.iter_mut().zip(reads) {
        r.inputs = inputs;
    }
    verify(&out).map_err(|e| Error::Mismatch(format!("folded graph: {e:?}")))?;
    Ok((out, all))
}
