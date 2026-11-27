//! Owned-row extents, argument slicing, and sliding-ring sizing for `Tree`.

use crate::extent::Extent;
use crate::tree::{Node, RING_CHUNK, T, Term, Tree};
use crate::{Error, Size};
use std::collections::HashMap;
use tessera_codegen::kept::{Layout, self};
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

/// Precomputes sparse-kept flags and sliding-ring capacities in one graph pass.
pub(crate) fn kept_flags(g: &Graph) -> (Vec<bool>, Vec<Option<usize>>) {
    let n_kept = g.kept.len();
    if n_kept == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut append_of: Vec<Option<u32>> = vec![None; g.nodes.len()];
    let mut kept_of: Vec<Option<u32>> = vec![None; g.nodes.len()];
    let mut sparse_found = vec![false; n_kept];
    let mut sparse_bad = vec![false; n_kept];
    let mut ring_max: Vec<Option<usize>> = vec![None; n_kept];
    let mut ring_bad = vec![false; n_kept];
    for (i, n) in g.nodes.iter().enumerate() {
        if let Op::Append(k) = n.op {
            append_of[i] = Some(k);
            kept_of[i] = Some(k);
        }
        for (pos, a) in n.args.iter().enumerate() {
            if let Some(k) = kept_of[a.0 as usize]
                && (k as usize) < n_kept
            {
                match &n.op {
                    Op::View(_) => kept_of[i] = Some(k),
                    Op::Attend(att) if att.picked => sparse_found[k as usize] = true,
                    _ => sparse_bad[k as usize] = true,
                }
            }
            if let Some(k) = append_of[a.0 as usize]
                && (k as usize) < n_kept
            {
                if let Op::Attend(att) = &n.op
                    && let Mask::Back(back) = att.mask
                    && att.ragged
                    && !att.picked
                    && (pos == 1 || pos == 2)
                {
                    let b = back as usize;
                    ring_max[k as usize] = Some(ring_max[k as usize].map_or(b, |m| m.max(b)));
                } else {
                    ring_bad[k as usize] = true;
                }
            }
        }
    }
    let sparse_kept = sparse_found
        .into_iter()
        .zip(sparse_bad)
        .map(|(f, b)| f && !b)
        .collect();
    let rows_max = g
        .dims
        .iter()
        .find(|d| d.name == DimName::Rows)
        .map_or(RING_CHUNK, |d| d.size.max() as usize);
    let chunk = rows_max.min(RING_CHUNK);
    let kept_rings = (0..n_kept)
        .map(|index| {
            if ring_bad[index] {
                return None;
            }
            let back = ring_max[index]?;
            let DimExpr::Dim(axis) = g.kept[index].shape.0.get(1)? else {
                return None;
            };
            let context = g.dims.iter().find(|d| d.name == *axis)?.size.max() as usize;
            let raw = back + 1 + chunk;
            let ring = if rows_max < 64 {
                raw
            } else {
                raw.next_multiple_of(64)
            };
            (ring < context).then_some(ring)
        })
        .collect();
    (sparse_kept, kept_rings)
}

impl<'a> Tree<'a> {
    /// Indexes kept appends, sparsity, and sliding-window rings.
    pub(crate) fn index_kept(&mut self) {
        let n_kept = self.g.kept.len();
        let mut kept_appends = vec![None; n_kept];
        for (idx, n) in self.nodes.iter().enumerate() {
            if let Term::Op(Op::Append(k)) = &n.term
                && let Some(slot) = kept_appends.get_mut(*k as usize)
                && slot.is_none()
            {
                *slot = Some(T(idx as u32));
            }
        }
        self.kept_appends = kept_appends;
        let (sparse_kept, kept_rings) = kept_flags(self.g);
        self.sparse_kept = sparse_kept;
        self.kept_rings = kept_rings;
    }

    /// The node appending to kept `index`, and rows per slot.
    pub fn kept_rows(&self, index: u32) -> Option<(T, usize)> {
        let t = (*self.kept_appends.get(index as usize)?)?;
        let DimExpr::Dim(axis) = &self.g.kept[index as usize].shape.0[1] else {
            return None;
        };
        // A slot holds the axis's bound, whatever this launch sees.
        let most = self.g.dims.iter().find(|d| d.name == *axis);
        Some((t, most.map_or(0, |d| d.size.max() as usize)))
    }

    /// Append node, layout, and rows per slot for kept `index`.
    pub fn kept_layout(&self, index: u32) -> Option<(T, &'static dyn Layout, usize)> {
        let (t, context) = self.kept_rows(index)?;
        match self.kept_rings.get(index as usize).copied().flatten() {
            Some(ring) => Some((t, kept::RING, ring)),
            None => Some((t, self.kept, context)),
        }
    }

    /// True if kept `index` feeds only picked attention.
    pub fn is_sparse_kept(&self, index: u32) -> bool {
        self.sparse_kept
            .get(index as usize)
            .copied()
            .unwrap_or(false)
    }

    /// True if any kept tensor is sparse-read.
    pub fn has_sparse_kept(&self) -> bool {
        self.sparse_kept.iter().any(|&s| s)
    }
}
