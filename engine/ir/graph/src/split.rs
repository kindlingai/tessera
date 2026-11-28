//! Splits a step graph into two microbatches for comm/compute overlap.

use crate::dims::{Dim, DimExpr, DimName, Shape, Size};
use crate::infer::{ErrorKind, infer};
use crate::node::{Composite, EntryKind, Graph, Node, NodeId, Op, Region, View};
use crate::types::{DType, TensorType};
use crate::verify::{VerifyError, verify};
use DimName::Rows;

/// Half the batch row count for two-microbatch execution.
pub const HALF_ROWS: DimName = DimName::named("HalfRows");

fn err(node: Option<u32>, message: String) -> VerifyError {
    VerifyError {
        node,
        error: ErrorKind::Shape.err(message),
    }
}

fn mentions(e: &DimExpr, name: DimName) -> bool {
    match e {
        DimExpr::Dim(n) => *n == name,
        DimExpr::Const(_) => false,
        DimExpr::Mul(a, b) | DimExpr::Add(a, b) => mentions(a, name) || mentions(b, name),
    }
}

fn has_rows(s: &Shape) -> bool {
    s.0.iter().any(|e| mentions(e, Rows))
}

fn by_rows(ty: &TensorType) -> bool {
    match ty.shape.0.split_first() {
        Some((DimExpr::Dim(Rows), rest)) => !rest.iter().any(|e| mentions(e, Rows)),
        _ => false,
    }
}

fn sub_expr(e: &DimExpr, with: &DimExpr) -> DimExpr {
    match e {
        DimExpr::Dim(Rows) => with.clone(),
        DimExpr::Mul(a, b) => {
            DimExpr::Mul(Box::new(sub_expr(a, with)), Box::new(sub_expr(b, with)))
        }
        DimExpr::Add(a, b) => {
            DimExpr::Add(Box::new(sub_expr(a, with)), Box::new(sub_expr(b, with)))
        }
        other => other.clone(),
    }
}

fn sub_half(e: &DimExpr) -> DimExpr {
    sub_expr(e, &DimExpr::Dim(HALF_ROWS))
}

fn sub_double(e: &DimExpr) -> DimExpr {
    sub_expr(e, &(DimExpr::Const(2) * DimExpr::Dim(HALF_ROWS)))
}

fn push(g: &mut Graph, op: Op, args: Vec<NodeId>) -> NodeId {
    g.nodes.push(Node {
        op,
        args,
        ty: TensorType::new([HALF_ROWS], DType::F32),
        numerics: None,
    });
    NodeId(g.nodes.len() as u32 - 1)
}

fn join_halves(g: &mut Graph, a: NodeId, b: NodeId, orig: &TensorType) -> NodeId {
    if orig.rank() == 1 {
        return push(g, Op::View(View::Concat), vec![a, b]);
    }
    let flat_half = orig
        .shape
        .0
        .iter()
        .map(sub_half)
        .reduce(|acc, d| acc * d)
        .unwrap_or(DimExpr::Const(1));
    let fa = push(
        g,
        Op::View(View::Reshape(Shape(vec![flat_half.clone()]))),
        vec![a],
    );
    let fb = push(g, Op::View(View::Reshape(Shape(vec![flat_half]))), vec![b]);
    let cat = push(g, Op::View(View::Concat), vec![fa, fb]);
    let full = Shape(orig.shape.0.iter().map(sub_double).collect());
    push(g, Op::View(View::Reshape(full)), vec![cat])
}

/// Duplicates `entry`'s row-parallel body into two half-row microbatches.
pub fn split_batch(g: &Graph, entry: usize) -> Result<Graph, VerifyError> {
    let is_step = g.entries.get(entry).map(|e| e.kind) == Some(EntryKind::Step);
    let rows_dim = g.dims.iter().find(|d| d.name == Rows);
    let (
        true,
        Some(Dim {
            size: Size::Bounded(max_rows),
            ..
        }),
    ) = (is_step, rows_dim)
    else {
        return Err(err(
            None,
            format!("entry {entry} is not a bounded-row step"),
        ));
    };
    if *max_rows < 2 {
        return Err(err(None, "Rows bound is smaller than 2".into()));
    }
    let n = g.nodes.len();
    let mut from_entry = vec![false; n];
    let mut row_input = vec![false; n];
    let mut is_split = vec![false; n];
    for (i, node) in g.nodes.iter().enumerate() {
        match node.op {
            Op::Input { entry: e, .. } => {
                from_entry[i] = e as usize == entry;
                row_input[i] = from_entry[i] && by_rows(&node.ty);
            }
            _ => {
                from_entry[i] = node.args.iter().any(|a| from_entry[a.0 as usize]);
                let reads_row = node
                    .args
                    .iter()
                    .any(|a| row_input[a.0 as usize] || is_split[a.0 as usize]);
                if from_entry[i] && (has_rows(&node.ty.shape) || (node.op.keeps() && reads_row)) {
                    if node.op.recurrent() {
                        return Err(err(
                            Some(i as u32),
                            "split_batch does not support recurrent state ops".into(),
                        ));
                    }
                    if let Op::Attend(ref a) = node.op
                        && !a.mask.causal()
                    {
                        return Err(err(
                            Some(i as u32),
                            "split_batch requires causal attention".into(),
                        ));
                    }
                    is_split[i] = true;
                }
            }
        }
    }
    // Clone nodes inside non-Model regions that contain split ops.
    let mut clone_node = is_split.clone();
    let mut split_regions = vec![false; g.regions.len()];
    for (ri, r) in g.regions.iter().enumerate() {
        if r.kind == Composite::Model {
            continue;
        }
        let span = r.start.0 as usize..r.end.0 as usize;
        if span.clone().any(|i| is_split[i]) {
            split_regions[ri] = true;
            for i in span {
                clone_node[i] = true;
            }
        }
    }
    let mut post_split = vec![false; n];
    for (i, node) in g.nodes.iter().enumerate() {
        if clone_node[i] {
            continue;
        }
        if node
            .args
            .iter()
            .any(|a| is_split[a.0 as usize] || post_split[a.0 as usize])
        {
            post_split[i] = true;
        }
    }
    for (i, node) in g.nodes.iter().enumerate() {
        if !post_split[i] {
            continue;
        }
        for a in &node.args {
            if is_split[a.0 as usize] && !by_rows(&g.node(*a).ty) {
                return Err(err(
                    Some(a.0),
                    "split node read after split is not axis-0 by Rows".into(),
                ));
            }
        }
    }
    for o in &g.entries[entry].outputs {
        let idx = o.node.0 as usize;
        if is_split[idx] && !by_rows(&g.node(o.node).ty) {
            return Err(err(
                Some(o.node.0),
                "split output is not axis-0 by Rows".into(),
            ));
        }
    }

    let mut out = g.clone();
    out.nodes.clear();
    out.regions.clear();
    if !out.dims.iter().any(|d| d.name == HALF_ROWS) {
        out.dims.push(Dim::bounded(HALF_ROWS, max_rows / 2));
    }
    for port in &mut out.entries[entry].inputs {
        if by_rows(&port.ty) {
            port.ty.shape = Shape(port.ty.shape.0.iter().map(sub_double).collect());
        }
    }

    let mut map_a = vec![NodeId(0); n];
    let mut map_b = vec![NodeId(0); n];
    let mut map_post = vec![NodeId(0); n];

    // Phase 0: shared inputs, weights, and non-split pre-body nodes.
    for (i, node) in g.nodes.iter().enumerate() {
        if clone_node[i] || post_split[i] {
            continue;
        }
        let args = node.args.iter().map(|a| map_a[a.0 as usize]).collect();
        let id = push(&mut out, node.op.clone(), args);
        out.nodes[id.0 as usize].numerics = node.numerics;
        map_post[i] = id;
        if row_input[i] {
            let sa = push(
                &mut out,
                Op::View(View::Slice {
                    axis: 0,
                    start: DimExpr::Const(0),
                    len: DimExpr::Dim(HALF_ROWS),
                }),
                vec![id],
            );
            let sb = push(
                &mut out,
                Op::View(View::Slice {
                    axis: 0,
                    start: DimExpr::Dim(HALF_ROWS),
                    len: DimExpr::Dim(HALF_ROWS),
                }),
                vec![id],
            );
            map_a[i] = sa;
            map_b[i] = sb;
        } else {
            map_a[i] = id;
            map_b[i] = id;
        }
    }

    // Emit half A, then half B, duplicating regions.
    for half in 0..2 {
        let map = if half == 0 { &mut map_a } else { &mut map_b };
        for (i, node) in g.nodes.iter().enumerate() {
            if !clone_node[i] {
                continue;
            }
            let args = node.args.iter().map(|a| map[a.0 as usize]).collect();
            let op = node.op.map_dims(sub_half);
            let id = push(&mut out, op, args);
            out.nodes[id.0 as usize].numerics = node.numerics;
            map[i] = id;
            if half == 0 && !is_split[i] {
                map_post[i] = id;
            }
        }
        for (ri, r) in g.regions.iter().enumerate() {
            if !split_regions[ri] {
                continue;
            }
            let start = map[r.start.0 as usize];
            let end = NodeId(map[r.end.0 as usize - 1].0 + 1);
            let outputs = r.outputs.iter().map(|o| map[o.0 as usize]).collect();
            let inputs = out.reads(start, end);
            out.regions.push(Region {
                kind: r.kind,
                start,
                end,
                inputs,
                outputs,
            });
        }
    }

    // Join split values read after the body, then emit tail.
    let mut joined: Vec<Option<NodeId>> = vec![None; n];
    let get_joined = |out: &mut Graph, joined: &mut [Option<NodeId>], idx: usize| -> NodeId {
        if let Some(id) = joined[idx] {
            return id;
        }
        let id = join_halves(out, map_a[idx], map_b[idx], &g.nodes[idx].ty);
        joined[idx] = Some(id);
        id
    };
    for (i, node) in g.nodes.iter().enumerate() {
        if !post_split[i] {
            continue;
        }
        let mut args = Vec::with_capacity(node.args.len());
        for a in &node.args {
            let idx = a.0 as usize;
            let mapped = if is_split[idx] {
                get_joined(&mut out, &mut joined, idx)
            } else {
                map_post[idx]
            };
            args.push(mapped);
        }
        let op = node.op.map_dims(sub_double);
        let id = push(&mut out, op, args);
        out.nodes[id.0 as usize].numerics = node.numerics;
        map_post[i] = id;
    }

    for e in 0..out.entries.len() {
        for oi in 0..out.entries[e].outputs.len() {
            let idx = out.entries[e].outputs[oi].node.0 as usize;
            let node = if e == entry && is_split[idx] {
                get_joined(&mut out, &mut joined, idx)
            } else {
                map_post[idx]
            };
            out.entries[e].outputs[oi].node = node;
        }
    }

    // Re-type every node and run full graph verification.
    for i in 0..out.nodes.len() {
        let node = &out.nodes[i];
        let args: Vec<&TensorType> = node.args.iter().map(|a| &out.node(*a).ty).collect();
        let ty = infer(&out, &node.op, &args).map_err(|error| VerifyError {
            node: Some(i as u32),
            error,
        })?;
        out.nodes[i].ty = ty;
    }
    verify(&out)?;
    Ok(out)
}
