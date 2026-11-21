//! Steps compute rows past their last append only where returned.

use crate::dims::{Dim, DimExpr, DimName, Shape};
use crate::infer::{ErrorKind, TypeError, infer};
use crate::node::{EntryKind, Graph, Node, NodeId, Op, Port, Role, Select, View};
use crate::types::{DType, TensorType};
use crate::verify::{VerifyError, verify};
use DimName::{Picks, Rows};

/// Which of a node's rows its readers use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Need {
    Unused,
    /// The rows in the `Picks` port.
    Picked,
    All,
}

/// Step `entry` with a `Picks` port. Outputs hold those rows.
pub fn trim(g: &Graph, entry: usize) -> Result<Graph, VerifyError> {
    let top = |message: String| VerifyError {
        node: None,
        error: TypeError {
            kind: ErrorKind::Shape,
            message,
        },
    };
    let rows = g.dims.iter().find(|d| d.name == Rows);
    let step = g.entries.get(entry).map(|e| e.kind) == Some(EntryKind::Step);
    let (Some(rows), true) = (rows, step) else {
        return Err(top(format!("entry {entry} is not a step over rows")));
    };
    let need = needs(g, entry);
    let mut anchors = vec![None; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if need[i] != Need::Picked {
            continue;
        }
        for (j, a) in n.args.iter().enumerate() {
            if gathers(g, &need, i, j) {
                earliest(&mut anchors[a.0 as usize], hoist(g, i));
            }
        }
    }
    for o in &g.entries[entry].outputs {
        let at = o.node.0 as usize;
        if need[at] == Need::All {
            if !gatherable(&g.node(o.node).ty) {
                return Err(top(format!("output {at} has no rows to pick")));
            }
            earliest(&mut anchors[at], g.nodes.len());
        }
    }
    let mut out = rebuild(g, entry, &need, &anchors);
    // A layer or model with picked rows splits apart.
    let regions = std::mem::take(&mut out.regions);
    for (r, was) in regions.into_iter().zip(&g.regions) {
        let mut span = was.start.0 as usize..was.end.0 as usize;
        if was.kind.block() || !span.any(|i| need[i] == Need::Picked) {
            out.regions.push(r);
        }
    }
    if !out.dims.iter().any(|d| d.name == Picks) {
        out.dims.push(Dim {
            name: Picks,
            size: rows.size,
        });
    }
    retype(&mut out)?;
    verify(&out)?;
    Ok(out)
}

fn earliest(was: &mut Option<usize>, at: usize) {
    *was = Some(was.map_or(at, |w| w.min(at)));
}

/// Gathers for reader `i` go before its outermost region.
fn hoist(g: &Graph, i: usize) -> usize {
    let id = NodeId(i as u32);
    let around = g.regions.iter().filter(|r| r.kind.block());
    let around = around.filter(|r| r.start <= id && id < r.end);
    around.map(|r| r.start.0 as usize).fold(i, usize::min)
}

/// Whether reader `i` reads argument `j` through a gather.
fn gathers(g: &Graph, need: &[Need], i: usize, j: usize) -> bool {
    let n = &g.nodes[i];
    let a = g.node(n.args[j]);
    need[n.args[j].0 as usize] == Need::All && by_rows(&a.ty) && rowwise(&n.op, j, &a.ty)
}

/// Each node's need, with regions and gathers kept whole.
fn needs(g: &Graph, entry: usize) -> Vec<Need> {
    let mut forced = vec![false; g.nodes.len()];
    loop {
        let need = sweep(g, entry, &forced);
        let mut more = false;
        for r in g.regions.iter().filter(|r| r.kind.block()) {
            let span = r.start.0 as usize..r.end.0 as usize;
            let all = span
                .clone()
                .any(|i| need[i] == Need::All && by_rows(&g.nodes[i].ty));
            for i in span.filter(|_| all) {
                if need[i] == Need::Picked {
                    forced[i] = true;
                    more = true;
                }
            }
        }
        for (i, n) in g.nodes.iter().enumerate() {
            let stuck = |j: usize| gathers(g, &need, i, j) && !gatherable(&g.node(n.args[j]).ty);
            if need[i] == Need::Picked && (0..n.args.len()).any(stuck) {
                forced[i] = true;
                more = true;
            }
        }
        if !more {
            return need;
        }
    }
}

/// One pass from the outputs back to the inputs.
fn sweep(g: &Graph, entry: usize, forced: &[bool]) -> Vec<Need> {
    let mut need = vec![Need::Unused; g.nodes.len()];
    for (i, n) in g.nodes.iter().enumerate() {
        if forced[i] || matches!(n.op, Op::Append(_)) {
            need[i] = Need::All;
        }
    }
    for (e, en) in g.entries.iter().enumerate() {
        let want = if e == entry { Need::Picked } else { Need::All };
        for o in &en.outputs {
            let at = &mut need[o.node.0 as usize];
            *at = (*at).max(want);
        }
    }
    for i in (0..g.nodes.len()).rev() {
        let n = &g.nodes[i];
        let mixed = |(j, a): (usize, &NodeId)| {
            let ty = &g.node(*a).ty;
            rowwise(&n.op, j, ty) && !by_rows(ty) && ty.shape.0.iter().any(|e| mentions(e, Rows))
        };
        let fixed = matches!(n.op, Op::Input { .. } | Op::Weight(_) | Op::Append(_));
        let fixed = fixed && !follows(g, &n.op);
        if need[i] == Need::Picked
            && (fixed || !by_rows(&n.ty) || n.args.iter().enumerate().any(mixed))
        {
            need[i] = Need::All;
        }
        for (j, a) in n.args.iter().enumerate() {
            let want = match need[i] {
                Need::Unused => Need::Unused,
                Need::Picked if rowwise(&n.op, j, &g.node(*a).ty) => Need::Picked,
                _ => Need::All,
            };
            let at = &mut need[a.0 as usize];
            *at = (*at).max(want);
        }
    }
    need
}

/// Whether `op` is a mask or candidate input.
fn follows(g: &Graph, op: &Op) -> bool {
    let Op::Input { entry, port } = op else {
        return false;
    };
    let port = g.entries[*entry as usize].inputs.get(*port as usize);
    port.is_some_and(|p| matches!(p.role, Role::Allowed | Role::Candidates))
}

/// Whether each result row reads only its row of `j`.
fn rowwise(op: &Op, j: usize, arg: &TensorType) -> bool {
    match op {
        Op::Row(_) | Op::Cast(_) => true,
        Op::View(View::Reshape(_) | View::Concat) => true,
        Op::View(View::Slice { axis, .. } | View::Expand { axis, .. }) => *axis != 0,
        Op::View(View::Permute(p)) => p.first() == Some(&0),
        Op::Select(Select::Take | Select::TopK { .. }) => true,
        Op::Select(Select::Gather) => j == 1,
        Op::Matmul => j == 0 || arg.rank() >= 3,
        Op::Attend(_) => j != 1 && j != 2,
        Op::Mean(axis) | Op::Mix(axis) => *axis != 0,
        Op::Input { .. }
        | Op::Weight(_)
        | Op::Mask(_)
        | Op::Append(_)
        | Op::Delta(_)
        | Op::Conv => false,
    }
}

/// Whether rows lie on axis 0 and nowhere else.
fn by_rows(ty: &TensorType) -> bool {
    match ty.shape.0.split_first() {
        Some((DimExpr::Dim(Rows), rest)) => !rest.iter().any(|e| mentions(e, Rows)),
        _ => false,
    }
}

/// Whether a gather kernel can pick rows of `ty`.
fn gatherable(ty: &TensorType) -> bool {
    by_rows(ty) && ty.rank() >= 2 && ty.dtype.is_float()
}

fn mentions(e: &DimExpr, name: DimName) -> bool {
    match e {
        DimExpr::Dim(n) => *n == name,
        DimExpr::Const(_) => false,
        DimExpr::Mul(a, b) | DimExpr::Add(a, b) => mentions(a, name) || mentions(b, name),
    }
}

fn renamed(e: &DimExpr) -> DimExpr {
    match e {
        DimExpr::Dim(Rows) => DimExpr::Dim(Picks),
        DimExpr::Mul(a, b) => DimExpr::Mul(Box::new(renamed(a)), Box::new(renamed(b))),
        DimExpr::Add(a, b) => DimExpr::Add(Box::new(renamed(a)), Box::new(renamed(b))),
        other => other.clone(),
    }
}

/// `op` over picked rows: its sizes name `Picks` for `Rows`.
fn picked(op: &Op) -> Op {
    match op {
        Op::View(View::Reshape(s)) => {
            Op::View(View::Reshape(Shape(s.0.iter().map(renamed).collect())))
        }
        Op::View(View::Slice { axis, start, len }) => Op::View(View::Slice {
            axis: *axis,
            start: renamed(start),
            len: renamed(len),
        }),
        Op::View(View::Expand { axis, to }) => Op::View(View::Expand {
            axis: *axis,
            to: renamed(to),
        }),
        Op::Select(Select::TopK { k }) => Op::Select(Select::TopK { k: renamed(k) }),
        other => other.clone(),
    }
}

/// The nodes again, with each gather before `anchors` names.
fn rebuild(g: &Graph, entry: usize, need: &[Need], anchors: &[Option<usize>]) -> Graph {
    let mut out = g.clone();
    out.nodes.clear();
    for (i, n) in g.nodes.iter().enumerate() {
        if let (Op::Input { entry, port }, Need::Picked) = (&n.op, need[i]) {
            let ty = &mut out.entries[*entry as usize].inputs[*port as usize].ty;
            ty.shape = Shape(ty.shape.0.iter().map(renamed).collect());
        }
    }
    let port = out.entries[entry].inputs.len() as u32;
    out.entries[entry].inputs.push(Port {
        role: Role::Picks,
        ty: TensorType::new([Picks], DType::U32),
    });
    let mut due = vec![Vec::new(); g.nodes.len() + 1];
    for (s, at) in anchors.iter().enumerate() {
        if let Some(at) = at {
            due[*at].push(s);
        }
    }
    let mut map: Vec<NodeId> = Vec::with_capacity(g.nodes.len());
    let mut gathered: Vec<Option<NodeId>> = vec![None; g.nodes.len()];
    let mut picks = None;
    for (i, sources) in due.iter().enumerate() {
        for &s in sources {
            let ids = *picks.get_or_insert_with(|| {
                let op = Op::Input {
                    entry: entry as u32,
                    port,
                };
                push(&mut out, op, Vec::new())
            });
            let op = Op::Select(Select::Gather);
            gathered[s] = Some(push(&mut out, op, vec![map[s], ids]));
        }
        let Some(n) = g.nodes.get(i) else {
            break;
        };
        let pick = need[i] == Need::Picked;
        let mut args = Vec::with_capacity(n.args.len());
        for (j, a) in n.args.iter().enumerate() {
            let via = gathered[a.0 as usize].filter(|_| pick && gathers(g, need, i, j));
            args.push(via.unwrap_or(map[a.0 as usize]));
        }
        let op = if pick { picked(&n.op) } else { n.op.clone() };
        let id = push(&mut out, op, args);
        out.nodes[id.0 as usize].numerics = n.numerics;
        map.push(id);
    }
    for (e, en) in out.entries.iter_mut().enumerate() {
        for o in &mut en.outputs {
            let was = o.node.0 as usize;
            o.node = gathered[was].filter(|_| e == entry).unwrap_or(map[was]);
        }
    }
    for r in &mut out.regions {
        r.start = map[r.start.0 as usize];
        r.end = NodeId(map[r.end.0 as usize - 1].0 + 1);
        r.outputs.iter_mut().for_each(|o| *o = map[o.0 as usize]);
    }
    for k in 0..out.regions.len() {
        let (start, end) = (out.regions[k].start, out.regions[k].end);
        out.regions[k].inputs = out.reads(start, end);
    }
    out
}

/// Appends a node, typed later by `retype`.
fn push(g: &mut Graph, op: Op, args: Vec<NodeId>) -> NodeId {
    g.nodes.push(Node {
        op,
        args,
        ty: TensorType::new([Picks], DType::U32),
        numerics: None,
    });
    NodeId(g.nodes.len() as u32 - 1)
}

/// Types every node again, in order.
fn retype(g: &mut Graph) -> Result<(), VerifyError> {
    for i in 0..g.nodes.len() {
        let n = &g.nodes[i];
        let args: Vec<&TensorType> = n.args.iter().map(|a| &g.node(*a).ty).collect();
        let ty = infer(g, &n.op, &args).map_err(|error| VerifyError {
            node: Some(i as u32),
            error,
        })?;
        g.nodes[i].ty = ty;
    }
    Ok(())
}
