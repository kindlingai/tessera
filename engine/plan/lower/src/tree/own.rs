//! Owned-row extents, argument slicing, and sliding-ring sizing for `Tree`.

use crate::extent::Extent;
use crate::tree::{Node, RING_CHUNK, T, Term, Tree};
use crate::{Error, Size};
use std::collections::HashMap;
use tessera_graph::{DimExpr, DimName, Graph, Mask, NodeId, Op, Parts, Placement, Select, View};

/// Ring capacity for sliding-window kept `index`, if smaller than context.
pub fn sliding_ring(g: &Graph, index: u32) -> Option<usize> {
    let k = g.kept.get(index as usize)?;
    let DimExpr::Dim(axis) = k.shape.0.get(1)? else {
        return None;
    };
    let context = g.dims.iter().find(|d| d.name == *axis)?.size.max() as usize;
    let appends: Vec<NodeId> = g
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.op == Op::Append(index))
        .map(|(i, _)| NodeId(i as u32))
        .collect();
    if appends.is_empty() {
        return None;
    }
    let mut max_back: Option<usize> = None;
    for n in &g.nodes {
        let reads = n
            .args
            .iter()
            .enumerate()
            .filter(|(_, a)| appends.contains(a));
        for (pos, _) in reads {
            let Op::Attend(a) = &n.op else {
                return None;
            };
            let Mask::Back(back) = a.mask else {
                return None;
            };
            if !a.ragged || a.picked || (pos != 1 && pos != 2) {
                return None;
            }
            let b = back as usize;
            max_back = Some(max_back.map_or(b, |m| m.max(b)));
        }
    }
    let back = max_back?;
    let rows_max = g
        .dims
        .iter()
        .find(|d| d.name == DimName::Rows)
        .map_or(RING_CHUNK, |d| d.size.max() as usize);
    let chunk = rows_max.min(RING_CHUNK);
    let raw = back + 1 + chunk;
    let ring = if rows_max < 64 {
        raw
    } else {
        raw.next_multiple_of(64)
    };
    (ring < context).then_some(ring)
}

/// Restricts the owned axis of `extents` to this rank's share.
pub(crate) fn own_extents(
    n: &tessera_graph::Node,
    extents: &mut [Extent],
    parts: &Parts,
    rank: u32,
) -> Result<(), Error> {
    let Placement::Owned { axis } = n.ty.placement else {
        return Ok(());
    };
    if parts.ranks() <= 1 {
        return Ok(());
    }
    let a = axis as usize;
    let whole = extents[a].most();
    let fixed = extents[a] == Extent::Fixed(whole);
    let (_, count) = parts.rows(whole, rank as usize);
    if !fixed || count == 0 {
        return Err(Size::Split {
            axis: a,
            shape: n.ty.shape.clone(),
            parts: parts.clone(),
        }
        .into());
    }
    extents[a] = Extent::Fixed(count);
    Ok(())
}

/// Slices `slot` on axis 0, pushing through rank-1 row gathers.
fn slice_axis0(
    tree: &mut Tree<'_>,
    slot: T,
    first: usize,
    count: usize,
    sliced: &mut HashMap<(T, usize, usize), T>,
) -> T {
    let key = (slot, first, count);
    if let Some(&t) = sliced.get(&key) {
        return t;
    }
    let src = tree.node(slot).clone();
    let whole = src.shape[0];
    let mut extents = src.extents;
    extents[0] = Extent::Fixed(count);
    let shape = extents.iter().map(Extent::most).collect();
    let (term, args) = if src.term == Term::Op(Op::Select(Select::Gather))
        && tree.node(src.args[1]).shape == [whole]
    {
        let cut_ids = slice_axis0(tree, src.args[1], first, count, sliced);
        (src.term, vec![src.args[0], cut_ids])
    } else {
        let view = View::Slice {
            axis: 0,
            start: DimExpr::Const(first as u64),
            len: DimExpr::Const(count as u64),
        };
        (Term::Op(Op::View(view)), vec![slot])
    };
    let t = tree.add(
        Node {
            term,
            args,
            shape,
            extents,
            form: src.form,
            from: src.from,
        },
        None,
    );
    sliced.insert(key, t);
    t
}

/// Slices replicated args matching an owned arg's leading axis.
pub(crate) fn slice_owned_args(
    tree: &mut Tree<'_>,
    g: &Graph,
    n: &tessera_graph::Node,
    args: &mut [T],
    sliced: &mut HashMap<(T, usize, usize), T>,
) {
    if tree.parts.ranks() <= 1 {
        return;
    }
    let owned = n
        .args
        .iter()
        .find(|a| g.node(**a).ty.placement == Placement::Owned { axis: 0 });
    let Some(owned) = owned else {
        return;
    };
    let lead = &g.node(*owned).ty.shape.0[0];
    for (a, slot) in n.args.iter().zip(args.iter_mut()) {
        let ty = &g.node(*a).ty;
        if ty.placement != Placement::Replicated || ty.shape.0.first() != Some(lead) {
            continue;
        }
        let whole = tree.node(*slot).shape[0];
        let (first, count) = tree.parts.rows(whole, tree.rank as usize);
        if count == whole && first == 0 {
            continue;
        }
        *slot = slice_axis0(tree, *slot, first, count, sliced);
    }
}
