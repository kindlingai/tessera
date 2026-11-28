//! Typing of row stages.

use crate::dims::{Dim, DimExpr, Shape};
use crate::infer::{ErrorKind, TypeError, finite, front, need, poly, same_shape, side};
use crate::stages::{Check, Place, Stage, Takes};
use crate::types::{DType, Placement, TensorType};

/// Integer positions lead the rows of `x`, on every rank.
fn positions(
    dims: &[Dim],
    x: &TensorType,
    pos: &TensorType,
    p: Placement,
) -> Result<(), TypeError> {
    let lead = pos.rank().min(x.rank() - 1);
    let fits = pos.rank() < x.rank() && same_shape(dims, &front(&x.shape, lead), &pos.shape)?;
    need(fits, ErrorKind::Shape, || {
        format!("positions {} do not lead {}", pos.shape, x.shape)
    })?;
    need(pos.dtype.is_int(), ErrorKind::DType, || {
        format!("positions must be integers, got {:?}", pos.dtype)
    })?;
    let last = x.rank() as u32 - 1;
    let placed = pos.placement == Placement::Replicated
        && match p {
            Placement::Split { axis } => axis != last,
            Placement::Owned { axis } => axis == 0 && axis != last,
            _ => true,
        };
    need(placed, ErrorKind::Placement, || {
        format!("{p:?} rows cannot follow {:?} positions", pos.placement)
    })
}

/// Multi-axis integer coordinates lead `x` with trailing axis count.
fn mrope_axes(
    dims: &[Dim],
    s: &Stage,
    x: &TensorType,
    pos: &TensorType,
    p: Placement,
) -> Result<(), TypeError> {
    let Stage::Mrope { sections, .. } = *s else {
        unreachable!("mrope stage expected");
    };
    let lead = pos.rank().saturating_sub(1);
    let need_axes = if sections[2] > 0 {
        3
    } else if sections[1] > 0 {
        2
    } else {
        1
    };
    let axes = pos
        .shape
        .0
        .last()
        .map(|e| poly(dims, e))
        .transpose()?
        .and_then(|n| n.as_const());
    let fits = pos.rank() >= 2
        && lead < x.rank()
        && same_shape(dims, &front(&x.shape, lead), &front(&pos.shape, lead))?
        && axes.is_some_and(|n| n >= need_axes);
    need(fits, ErrorKind::Shape, || {
        format!("axes {} do not fit {}", pos.shape, x.shape)
    })?;
    let sum: u64 = sections.iter().map(|&n| u64::from(n)).sum();
    if let Some(width) = poly(dims, x.shape.0.last().expect("rank checked"))?.as_const() {
        need(sum * 2 == width, ErrorKind::Shape, || {
            format!("mrope sections {sections:?} do not cover width {width}")
        })?;
    }
    need(pos.dtype.is_int(), ErrorKind::DType, || {
        format!("axes must be integers, got {:?}", pos.dtype)
    })?;
    let last = x.rank() as u32 - 1;
    let placed = pos.placement == Placement::Replicated && p != Placement::Split { axis: last };
    need(placed, ErrorKind::Placement, || {
        format!("{p:?} rows cannot follow {:?} axes", pos.placement)
    })
}

/// The mask words, without any trailing edit words.
fn masked(words: &DimExpr) -> &DimExpr {
    match words {
        DimExpr::Add(mask, _) => mask,
        whole => whole,
    }
}

/// One `u32` word per 32 entries of each row.
fn allowed(dims: &[Dim], x: &TensorType, bits: &TensorType) -> Result<(), TypeError> {
    let lead = x.rank() - 1;
    let words = |s: &DimExpr| s.poly(dims).ok().and_then(|n| n.as_const());
    let width = words(&x.shape.0[lead]);
    let fits = bits.rank() == x.rank()
        && same_shape(dims, &front(&x.shape, lead), &front(&bits.shape, lead))?
        && words(masked(&bits.shape.0[lead])) == width.map(|w| w.div_ceil(32));
    need(fits, ErrorKind::Shape, || {
        format!("mask {} does not cover {}", bits.shape, x.shape)
    })?;
    side("masks", bits, DType::U32, Placement::Replicated)
}

pub(crate) fn row(
    dims: &[Dim],
    stages: &[Stage],
    args: &[&TensorType],
) -> Result<TensorType, TypeError> {
    let x = args[0];
    need(x.rank() >= 1, ErrorKind::Shape, || {
        "a row op needs rank 1 or more".into()
    })?;
    need(x.dtype.is_float(), ErrorKind::DType, || {
        format!("row stages need floats, got {:?}", x.dtype)
    })?;
    let last = x.rank() as u32 - 1;
    let width = x.shape.0.last().expect("rank checked");
    let mut p = x.placement;
    let mut rest = args[1..].iter();
    for s in stages {
        let f = s.facts();
        f.check.holds()?;
        match *s {
            Stage::L2Clip { eps, .. } => {
                need(eps.is_finite() && eps >= 0.0, ErrorKind::Value, || {
                    format!("l2 clip epsilon is {eps}")
                })?
            }
            Stage::LogScale { alpha, .. } => finite(alpha, "log scale alpha")?,
            _ => {}
        }
        let operand = (f.operand != Takes::Nothing).then(|| *rest.next().expect("arity checked"));
        if let Some(o) = operand {
            takes(dims, f.operand, s, x, o, p)?;
        }
        if f.blocks > 1 {
            need(blocks(dims, width, f.blocks)?, ErrorKind::Shape, || {
                format!("{s:?} needs rows in blocks of {}, got {width}", f.blocks)
            })?;
        }
        p = place(f.place, s, p, x, operand)?;
        if f.whole {
            whole(p, last, s)?;
        }
    }
    Ok(TensorType {
        shape: x.shape.clone(),
        dtype: x.dtype,
        placement: p,
    })
}

impl Check {
    fn holds(self) -> Result<(), TypeError> {
        match self {
            Check::Pass => Ok(()),
            Check::Finite(x, what) => finite(x, what),
            Check::Positive(x, what) => need(x.is_finite() && x > 0.0, ErrorKind::Value, || {
                format!("{what} is {x}")
            }),
            Check::PowerOfTwo(n) => need(n.is_power_of_two(), ErrorKind::Shape, || {
                format!("blocks of {n} are not a power of two")
            }),
        }
    }
}

/// Checks operand `o` of stage `s` against row input `x`.
fn takes(
    dims: &[Dim],
    kind: Takes,
    s: &Stage,
    x: &TensorType,
    o: &TensorType,
    p: Placement,
) -> Result<(), TypeError> {
    match kind {
        Takes::Nothing => Ok(()),
        Takes::Broadcast => {
            broadcast(dims, &x.shape, &o.shape)?;
            need(o.dtype == x.dtype, ErrorKind::DType, || {
                format!("{s:?} mixes {:?} and {:?}", x.dtype, o.dtype)
            })
        }
        Takes::RowWeight => {
            let width = std::slice::from_ref(x.shape.0.last().expect("rank checked"));
            need(
                same_shape(dims, &o.shape, &Shape(width.to_vec()))?,
                ErrorKind::Shape,
                || format!("row weight {} does not match rows of {}", o.shape, x.shape),
            )?;
            side("row weights", o, x.dtype, Placement::Replicated)
        }
        Takes::Positions => positions(dims, x, o, p),
        Takes::Axes => mrope_axes(dims, s, x, o, p),
        Takes::Bits => allowed(dims, x, o),
    }
}

/// The placement stage `s` gives a row at `p`.
fn place(
    rule: Place,
    s: &Stage,
    p: Placement,
    x: &TensorType,
    operand: Option<&TensorType>,
) -> Result<Placement, TypeError> {
    let nonlinear = |ok| {
        need(ok, ErrorKind::Placement, || {
            format!("{s:?} is nonlinear, so it cannot read {p:?} values")
        })
        .map(|()| p)
    };
    let (ok, text, crossed) = match rule {
        Place::Reduce => (
            p == Placement::Partial,
            "needs partial sums",
            Placement::Replicated,
        ),
        Place::Gather => (
            p == Placement::Split {
                axis: x.rank() as u32 - 1,
            },
            "needs a split last axis",
            Placement::Replicated,
        ),
        Place::Scatter => (
            p == Placement::Partial && x.rank() >= 2,
            "needs partial sums over rows",
            Placement::Owned { axis: 0 },
        ),
        Place::Join => (
            p == Placement::Owned { axis: 0 },
            "needs owned rows",
            Placement::Replicated,
        ),
        Place::Own => (
            p == Placement::Replicated && x.rank() >= 2,
            "needs replicated rows",
            Placement::Owned { axis: 0 },
        ),
        _ => (true, "", p),
    };
    need(ok, ErrorKind::Placement, || {
        format!("{s:?} {text}, got {p:?}")
    })?;
    match rule {
        Place::Keep => Ok(p),
        Place::Sum | Place::Product => {
            let o = operand.expect("joining stages take an operand");
            combine(rule == Place::Sum, s, p, o.placement, x.rank() - o.rank())
        }
        Place::Replicated => nonlinear(p == Placement::Replicated),
        Place::Nonlinear(keeps_zero) => {
            nonlinear(p != Placement::Partial && (p != Placement::Routed || keeps_zero))
        }
        Place::Reduce | Place::Gather | Place::Scatter | Place::Join | Place::Own => Ok(crossed),
    }
}

/// The operand must match the row input's trailing axes.
fn broadcast(dims: &[Dim], x: &Shape, o: &Shape) -> Result<(), TypeError> {
    let off = x.rank().checked_sub(o.rank());
    let fits = match off {
        Some(off) => same_shape(dims, &Shape(x.0[off..].to_vec()), o)?,
        None => false,
    };
    need(fits, ErrorKind::Shape, || {
        format!("{o} does not broadcast to {x}")
    })
}

/// A sum or product of `p` and operand placement `q`.
fn combine(
    sum: bool,
    s: &Stage,
    p: Placement,
    q: Placement,
    lead: usize,
) -> Result<Placement, TypeError> {
    use Placement::*;
    let clash = || ErrorKind::Placement.err(format!("{s:?} cannot combine {p:?} with {q:?}"));
    match (p, q) {
        (Partial, Partial) if sum => Ok(Partial),
        (Partial, Replicated) if !sum => Ok(Partial),
        (Routed, Routed) => Ok(Routed),
        (Routed, Replicated) | (Replicated, Routed) if !sum => Ok(Routed),
        // A rank takes its own rows of a whole operand.
        (Owned { axis: 0 }, Replicated) => Ok(Owned { axis: 0 }),
        (Owned { axis: 0 }, Owned { axis: 0 }) | (Replicated, Owned { axis: 0 }) if lead == 0 => {
            Ok(Owned { axis: 0 })
        }
        (Owned { axis }, Replicated) if (axis as usize) < lead => Ok(Owned { axis }),
        (Owned { axis: a }, Owned { axis: b }) if b + lead as u32 == a => Ok(Owned { axis: a }),
        (_, Owned { .. }) | (Owned { .. }, _) => Err(clash()),
        (a, b) if shift(b, lead) == a && a != Partial => Ok(a),
        (Split { axis }, Replicated) if (axis as usize) < lead => Ok(p),
        _ => Err(clash()),
    }
}

/// An operand's split axis, counted in the row input's axes.
pub(crate) fn shift(q: Placement, lead: usize) -> Placement {
    match q {
        Placement::Split { axis } => Placement::Split {
            axis: axis + lead as u32,
        },
        q => q,
    }
}

/// `x` without `axis`, as a sum over whole values.
pub(crate) fn mean(axis: u32, x: &TensorType) -> Result<TensorType, TypeError> {
    need(
        x.placement != Placement::Partial,
        ErrorKind::Placement,
        || "a mean cannot read partial sums".into(),
    )?;
    collapse(axis, x)
}

/// Drops `axis` from `x`, turning routed slices into partial sums.
fn collapse(axis: u32, x: &TensorType) -> Result<TensorType, TypeError> {
    need(x.dtype.is_float(), ErrorKind::DType, || {
        format!("a mean needs floats, got {:?}", x.dtype)
    })?;
    need((axis as usize) < x.rank(), ErrorKind::Shape, || {
        format!("no axis {axis} in {}", x.shape)
    })?;
    need(
        !matches!(x.placement, Placement::Owned { .. }),
        ErrorKind::Placement,
        || "a mean cannot read owned rows".into(),
    )?;
    let mut shape = x.shape.0.clone();
    shape.remove(axis as usize);
    // Each rank sums its own rows.
    let placement = match x.placement {
        Placement::Routed => Placement::Partial,
        p => p,
    };
    Ok(TensorType {
        shape: Shape(shape),
        placement,
        ..x.clone()
    })
}

/// `x` without `axis`, each slice scaled by `w[.., axis]`.
pub(crate) fn mix(
    dims: &[Dim],
    axis: u32,
    x: &TensorType,
    w: &TensorType,
) -> Result<TensorType, TypeError> {
    let out = collapse(axis, x)?;
    let lead = axis as usize + 1;
    let fits = w.rank() == lead && same_shape(dims, &front(&x.shape, lead), &w.shape)?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "mix weights {} do not match {} up to axis {axis}",
            w.shape, x.shape
        )
    })?;
    side("mix weights", w, x.dtype, Placement::Replicated)?;
    Ok(out)
}

/// Whether a known `width` is a multiple of `block`.
fn blocks(dims: &[Dim], width: &DimExpr, block: u32) -> Result<bool, TypeError> {
    let n = width
        .poly(dims)
        .map_err(|d| ErrorKind::Reference.err(format!("{d:?} is not a dimension")))?;
    Ok(n.as_const()
        .is_none_or(|n| n.is_multiple_of(u64::from(block))))
}

fn whole(p: Placement, last: u32, s: &Stage) -> Result<(), TypeError> {
    let ok = match p {
        Placement::Replicated => true,
        Placement::Split { axis } | Placement::Owned { axis } => axis < last,
        Placement::Partial | Placement::Routed => false,
    };
    need(ok, ErrorKind::Placement, || {
        format!("{s:?} needs whole rows, got {p:?}")
    })
}
