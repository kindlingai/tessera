//! Owned-row extents, argument slicing, and sliding-ring sizing for `Tree`.

use crate::extent::Extent;
use crate::tree::{Node, RING_CHUNK, T, Term, Tree};
use crate::{Error, Size};
use std::collections::HashMap;
use tessera_graph::{DimExpr, DimName, Graph, Mask, NodeId, Op, Parts, Placement, View};

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

/// Restricts axis 0 of `extents` to this rank's owned rows.
pub(crate) fn own_extents(
    n: &tessera_graph::Node,
    extents: &mut [Extent],
    parts: &Parts,
    rank: u32,
) -> Result<(), Error> {
    if n.ty.placement != Placement::Owned || parts.ranks() <= 1 {
        return Ok(());
    }
    let whole = extents[0].most();
    let fixed = extents[0] == Extent::Fixed(whole);
    let (_, count) = parts.rows(whole, rank as usize);
    if !fixed || count == 0 {
        return Err(Size::Split {
            axis: 0,
            shape: n.ty.shape.clone(),
            parts: parts.clone(),
        }
        .into());
    }
    extents[0] = Extent::Fixed(count);
    Ok(())
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
        .find(|a| g.node(**a).ty.placement == Placement::Owned);
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
        let key = (*slot, first, count);
        let cut = match sliced.get(&key) {
            Some(&t) => t,
            None => {
                let src = tree.node(*slot).clone();
                let mut extents = src.extents;
                extents[0] = Extent::Fixed(count);
                let shape = extents.iter().map(Extent::most).collect();
                let view = View::Slice {
                    axis: 0,
                    start: DimExpr::Const(first as u64),
                    len: DimExpr::Const(count as u64),
                };
                let t = tree.add(
                    Node {
                        term: Term::Op(Op::View(view)),
                        args: vec![*slot],
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
        };
        *slot = cut;
    }
}
