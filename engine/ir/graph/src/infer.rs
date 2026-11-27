//! Per-op typing rules: shapes, dtypes and placements.

use crate::attend::{attend, heads_fit};
use crate::dims::{Dim, DimExpr, DimName, Poly, Shape};
use crate::node::{Graph, Op, View};
use crate::select;
use crate::stages::{Check, Place, Stage, Takes};
use crate::types::{DType, Placement, TensorType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Arity,
    /// A bad op parameter, like an infinite constant.
    Value,
    Reference,
    Shape,
    DType,
    Placement,
    Numerics,
    Region,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{kind:?} error: {message}")]
pub struct TypeError {
    pub kind: ErrorKind,
    pub message: String,
}

pub(crate) fn need(
    ok: bool,
    kind: ErrorKind,
    message: impl FnOnce() -> String,
) -> Result<(), TypeError> {
    if ok {
        return Ok(());
    }
    Err(TypeError {
        kind,
        message: message(),
    })
}

fn undeclared(name: DimName) -> TypeError {
    TypeError {
        kind: ErrorKind::Reference,
        message: format!("undeclared dimension {name:?}"),
    }
}

pub(crate) fn poly(dims: &[Dim], e: &DimExpr) -> Result<Poly, TypeError> {
    e.poly(dims).map_err(undeclared)
}

pub(crate) fn elements(dims: &[Dim], s: &Shape) -> Result<Poly, TypeError> {
    s.elements(dims).map_err(undeclared)
}

pub(crate) fn same(dims: &[Dim], a: &DimExpr, b: &DimExpr) -> Result<bool, TypeError> {
    Ok(poly(dims, a)? == poly(dims, b)?)
}

pub(crate) fn same_shape(dims: &[Dim], a: &Shape, b: &Shape) -> Result<bool, TypeError> {
    if a.rank() != b.rank() {
        return Ok(false);
    }
    for (x, y) in a.0.iter().zip(&b.0) {
        if !same(dims, x, y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The first `n` axes.
pub(crate) fn front(s: &Shape, n: usize) -> Shape {
    Shape(s.0[..n].to_vec())
}

fn finite(x: f32, what: &str) -> Result<(), TypeError> {
    need(x.is_finite(), ErrorKind::Value, || format!("{what} is {x}"))
}

/// The type `op` gives `args`, or why they don't fit.
pub fn infer(g: &Graph, op: &Op, args: &[&TensorType]) -> Result<TensorType, TypeError> {
    let want = op.facts().args;
    need(args.len() == want, ErrorKind::Arity, || {
        format!("{op:?} takes {want} arguments, got {}", args.len())
    })?;
    let missing = || TypeError {
        kind: ErrorKind::Reference,
        message: format!("{op:?} names nothing in the graph"),
    };
    match op {
        Op::Input { entry, port } => g
            .entries
            .get(*entry as usize)
            .and_then(|e| e.inputs.get(*port as usize))
            .map(|p| p.ty.clone())
            .ok_or_else(missing),
        Op::Weight(i) => g
            .weights
            .get(*i as usize)
            .map(|w| w.ty.clone())
            .ok_or_else(missing),
        Op::Matmul => matmul(&g.dims, args[0], args[1]),
        Op::Conv(c) => {
            need(c.dilation >= 1, ErrorKind::Value, || {
                format!("conv dilation is {}", c.dilation)
            })?;
            match c.state {
                Some(i) => {
                    let kept = g.kept.get(i as usize).ok_or_else(missing)?;
                    conv_rows(&g.dims, kept, c.dilation, args)
                }
                None => conv(&g.dims, args[0], args[1]),
            }
        }
        Op::Attend(a) => {
            finite(a.scale, "attention scale")?;
            need(!a.ragged || a.mask.causal(), ErrorKind::Value, || {
                format!("ragged attention is causal only, got {:?}", a.mask)
            })?;
            attend(&g.dims, a, args)
        }
        Op::Delta(d) => {
            finite(d.scale, "delta rule scale")?;
            match d.state {
                Some(i) => {
                    let kept = g.kept.get(i as usize).ok_or_else(missing)?;
                    delta_rows(&g.dims, kept, args)
                }
                None => delta(&g.dims, args),
            }
        }
        Op::Row(r) => row(&g.dims, &r.0, args),
        Op::Mask(_) => select::mask(args[0]),
        Op::Append(i) => {
            let kept = g.kept.get(*i as usize).ok_or_else(missing)?;
            append(&g.dims, kept, args)
        }
        Op::Select(s) => select::select(&g.dims, s, args),
        Op::View(View::Concat) => select::concat(&g.dims, args[0], args[1]),
        Op::View(v) => select::view(&g.dims, v, args[0]),
        Op::Mean(axis) => mean(*axis, args[0]),
        Op::Mix(axis) => mix(&g.dims, *axis, args[0], args[1]),
        Op::Cast(dtype) => {
            // Rounding each rank's share is not rounding the sum.
            need(
                args[0].placement != Placement::Partial,
                ErrorKind::Placement,
                || "cast of partial sums".into(),
            )?;
            // No kernel or interp path casts other types yet.
            let plain = |d| matches!(d, DType::F32 | DType::I32 | DType::U32);
            need(
                plain(args[0].dtype) && plain(*dtype),
                ErrorKind::DType,
                || format!("no cast from {:?} to {dtype:?}", args[0].dtype),
            )?;
            Ok(TensorType {
                dtype: *dtype,
                ..args[0].clone()
            })
        }
    }
}

fn matmul(dims: &[Dim], x: &TensorType, w: &TensorType) -> Result<TensorType, TypeError> {
    use Placement::*;
    // The batch axes of `x` lead those of `w`.
    let lead = w.rank().saturating_sub(2);
    let own = x.rank().saturating_sub(1).min(lead);
    let fits = w.rank() >= 2
        && x.rank() >= 1
        && same_shape(dims, &front(&x.shape, own), &front(&w.shape, own))?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "matmul takes x [.., K] and w [.., N, K], got {} and {}",
            x.shape, w.shape
        )
    })?;
    let k = x.shape.0.last().expect("rank checked");
    need(
        same(dims, k, &w.shape.0[lead + 1])?,
        ErrorKind::Shape,
        || format!("matmul K differs: x {} and w {}", x.shape, w.shape),
    )?;
    need(
        x.dtype.is_float() && x.dtype == w.dtype,
        ErrorKind::DType,
        || {
            format!(
                "matmul needs matching floats, got {:?} and {:?}",
                x.dtype, w.dtype
            )
        },
    )?;
    let last = x.rank() as u32 - 1;
    let n = lead as u32;
    let out_last = (lead + x.rank() - 1 - own) as u32;
    let placement = match (x.placement, w.placement) {
        (Replicated, Replicated) => Replicated,
        // Zero rows stay zero, so routed rows stay routed.
        (Replicated | Routed, Routed) | (Routed, Replicated) => Routed,
        (Replicated, Split { axis }) if axis == n => Split { axis: out_last },
        (Split { axis }, Replicated) if (n..last).contains(&axis) => Split { axis },
        (Partial, Replicated) => Partial,
        // A whole weight serves any share of the rows.
        (Owned { axis }, Replicated) if axis < last && ((axis as usize) >= own || axis == 0) => {
            let out_axis = if (axis as usize) < own {
                axis
            } else {
                (lead + axis as usize - own) as u32
            };
            Owned { axis: out_axis }
        }
        (Split { axis }, Split { axis: k }) if axis == last && k == n + 1 => Partial,
        // Both split or owned along one shared batch axis.
        (Split { axis }, Split { axis: k }) if axis == k && (axis as usize) < own => Split { axis },
        (Owned { axis }, Owned { axis: k }) if axis == k && (axis as usize) < own => Owned { axis },
        (xp, wp) => {
            return Err(TypeError {
                kind: ErrorKind::Placement,
                message: format!("matmul cannot combine x {xp:?} with w {wp:?}"),
            });
        }
    };
    let mut shape = w.shape.0[..lead].to_vec();
    shape.extend_from_slice(&x.shape.0[own..x.rank() - 1]);
    shape.push(w.shape.0[lead].clone());
    Ok(TensorType {
        shape: Shape(shape),
        dtype: x.dtype,
        placement,
    })
}

/// Rows `[Seqs, Context, C]` through taps `[C, K]`.
fn conv(dims: &[Dim], x: &TensorType, w: &TensorType) -> Result<TensorType, TypeError> {
    let fits = x.rank() == 3 && taps_fit(dims, x, w)?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "conv takes x [Seqs, Context, C] and w [C, K] or [Seqs, Context, C, K], got {} and {}",
            x.shape, w.shape
        )
    })?;
    conv_types(x, w)?;
    Ok(x.clone())
}

/// Ragged `[Rows, C]` through taps `[C, K]` with kept state.
fn conv_rows(
    dims: &[Dim],
    kept: &TensorType,
    dilation: u32,
    a: &[&TensorType],
) -> Result<TensorType, TypeError> {
    let (x, w, slots, positions) = (a[0], a[1], a[2], a[3]);
    let taps = match w.shape.0.last() {
        Some(k) => poly(dims, k)?.as_const(),
        None => None,
    };
    let hist = kept
        .shape
        .0
        .get(1)
        .and_then(|e| poly(dims, e).ok()?.as_const());
    let want = taps
        .and_then(|k| k.checked_sub(1))
        .map(|k| k * u64::from(dilation));
    let fits = x.rank() == 2
        && kept.rank() == 3
        && taps_fit(dims, x, w)?
        && same(dims, &x.shape.0[1], &kept.shape.0[2])?
        && want.is_some_and(|n| n > 0 && hist == Some(n));
    need(fits, ErrorKind::Shape, || {
        format!(
            "ragged conv takes x [Rows, C], w [C, K] or [Rows, C, K] and state [Seqs, (K-1)*d, C], got {}, {}, {}",
            x.shape, w.shape, kept.shape
        )
    })?;
    conv_types(x, w)?;
    need(kept.dtype == DType::F32, ErrorKind::DType, || {
        format!("conv state must be f32, got {:?}", kept.dtype)
    })?;
    need(
        kept.placement == Placement::Replicated,
        ErrorKind::Placement,
        || format!("conv state must be replicated, got {:?}", kept.placement),
    )?;
    row_ids(dims, &x.shape, slots, positions)?;
    Ok(x.clone())
}

/// Whether `w` is `[C, K]` or per-row taps.
fn taps_fit(dims: &[Dim], x: &TensorType, w: &TensorType) -> Result<bool, TypeError> {
    let Some(c) = x.shape.0.last() else {
        return Ok(false);
    };
    match w.rank() == 2 {
        true => same(dims, c, &w.shape.0[0]),
        false => {
            Ok(w.rank() == x.rank() + 1 && same_shape(dims, &front(&w.shape, x.rank()), &x.shape)?)
        }
    }
}

fn conv_types(x: &TensorType, w: &TensorType) -> Result<(), TypeError> {
    need(
        x.dtype.is_float() && x.dtype == w.dtype,
        ErrorKind::DType,
        || {
            format!(
                "conv needs matching floats, got {:?} and {:?}",
                x.dtype, w.dtype
            )
        },
    )?;
    let whole = [x, w].iter().all(|t| t.placement == Placement::Replicated);
    need(whole, ErrorKind::Placement, || {
        format!(
            "conv takes whole tensors, got {:?} and {:?}",
            x.placement, w.placement
        )
    })
}

/// Value heads group over the query and key heads.
fn delta(dims: &[Dim], a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    let fits = [q, k, v].iter().all(|t| t.rank() == 4)
        && same_shape(dims, &q.shape, &k.shape)?
        && same_shape(dims, &front(&q.shape, 2), &front(&v.shape, 2))?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "the delta rule takes [Seqs, Context, Heads, HeadDim] rows, got {}, {} and {}",
            q.shape, k.shape, v.shape
        )
    })?;
    heads_fit(dims, [v, q, k], 2, false)?;
    delta_gates(dims, v, &a[3..5], 3)?;
    Ok(v.clone())
}

/// Ragged delta rule with kept per-slot state.
fn delta_rows(dims: &[Dim], kept: &TensorType, a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    let fits = [q, k, v].iter().all(|t| t.rank() == 3)
        && kept.rank() == 4
        && same_shape(dims, &q.shape, &k.shape)?
        && same(dims, &q.shape.0[0], &v.shape.0[0])?
        && same(dims, &kept.shape.0[1], &v.shape.0[1])?
        && same(dims, &kept.shape.0[2], &q.shape.0[2])?
        && same(dims, &kept.shape.0[3], &v.shape.0[2])?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "ragged delta takes [Rows, Heads, Dim] and state [Seqs, Heads, Dk, Dv], got {}, {}, {}, {}",
            q.shape, k.shape, v.shape, kept.shape
        )
    })?;
    let heads = poly(dims, &v.shape.0[1])?.as_const();
    let kv_heads = poly(dims, &q.shape.0[1])?.as_const();
    if let (Some(h), Some(kv)) = (heads, kv_heads) {
        need(kv != 0 && h % kv == 0, ErrorKind::Shape, || {
            format!("{h} value heads do not group over {kv} key heads")
        })?;
    }
    need(
        q.dtype.is_float() && q.dtype == k.dtype && k.dtype == v.dtype && kept.dtype == DType::F32,
        ErrorKind::DType,
        || "ragged delta needs matching floats and f32 state".into(),
    )?;
    let p = [v.placement, q.placement, k.placement, kept.placement];
    let split = [Placement::Split { axis: 1 }; 4];
    need(
        p == [Placement::Replicated; 4] || p == split,
        ErrorKind::Placement,
        || format!("ragged delta needs whole or head-split tensors, got {p:?}"),
    )?;
    delta_gates(dims, v, &a[3..5], 2)?;
    row_ids(dims, &v.shape, a[5], a[6])?;
    Ok(v.clone())
}

fn delta_gates(
    dims: &[Dim],
    v: &TensorType,
    gates: &[&TensorType],
    lead: usize,
) -> Result<(), TypeError> {
    for gate in gates {
        let fits = same_shape(dims, &gate.shape, &front(&v.shape, lead))?;
        need(fits, ErrorKind::Shape, || {
            format!("gates {} do not match values {}", gate.shape, v.shape)
        })?;
        need(gate.dtype == v.dtype, ErrorKind::DType, || {
            format!("gates are {:?}, values {:?}", gate.dtype, v.dtype)
        })?;
        need(gate.placement == v.placement, ErrorKind::Placement, || {
            format!("gates are {:?}, values {:?}", gate.placement, v.placement)
        })?;
    }
    Ok(())
}

/// One integer key list per query, over `lead` axes.
pub(crate) fn key_lists(
    dims: &[Dim],
    ids: &TensorType,
    q: &TensorType,
    lead: usize,
) -> Result<(), TypeError> {
    let fits = ids.rank() == lead + 1
        && same_shape(dims, &front(&ids.shape, lead), &front(&q.shape, lead))?;
    need(fits, ErrorKind::Shape, || {
        format!("key lists {} do not match queries {}", ids.shape, q.shape)
    })?;
    need(ids.dtype.is_int(), ErrorKind::DType, || {
        format!("key lists must be integers, got {:?}", ids.dtype)
    })?;
    let placed = match q.placement {
        Placement::Owned { axis: 0 } => {
            matches!(
                ids.placement,
                Placement::Replicated | Placement::Owned { axis: 0 }
            )
        }
        _ => ids.placement == Placement::Replicated,
    };
    need(placed, ErrorKind::Placement, || {
        format!(
            "key lists {:?} do not match queries {:?}",
            ids.placement, q.placement
        )
    })
}

/// Queries' type with the values' width.
pub(crate) fn valued(q: &TensorType, v: &TensorType) -> TensorType {
    let mut out = q.clone();
    let width = v.shape.0.last().expect("rank checked").clone();
    *out.shape.0.last_mut().expect("rank checked") = width;
    out
}

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
    need(bits.dtype == DType::U32, ErrorKind::DType, || {
        format!("masks must be u32, got {:?}", bits.dtype)
    })?;
    need(
        bits.placement == Placement::Replicated,
        ErrorKind::Placement,
        || "masks must be replicated".into(),
    )
}

fn row(dims: &[Dim], stages: &[Stage], args: &[&TensorType]) -> Result<TensorType, TypeError> {
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
            need(o.dtype == x.dtype, ErrorKind::DType, || {
                "row weight dtype differs".into()
            })?;
            need(
                o.placement == Placement::Replicated,
                ErrorKind::Placement,
                || "row weights must be replicated".into(),
            )
        }
        Takes::Positions => positions(dims, x, o, p),
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
    let last = x.rank() as u32 - 1;
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
        Place::Reduce => need(p == Placement::Partial, ErrorKind::Placement, || {
            format!("{s:?} needs partial sums, got {p:?}")
        })
        .map(|()| Placement::Replicated),
        Place::Gather => need(
            p == Placement::Split { axis: last },
            ErrorKind::Placement,
            || format!("{s:?} needs a split last axis, got {p:?}"),
        )
        .map(|()| Placement::Replicated),
        Place::Scatter => need(
            p == Placement::Partial && x.rank() >= 2,
            ErrorKind::Placement,
            || format!("{s:?} needs partial sums over rows, got {p:?}"),
        )
        .map(|()| Placement::Owned { axis: 0 }),
        Place::Join => need(
            p == Placement::Owned { axis: 0 },
            ErrorKind::Placement,
            || format!("{s:?} needs owned rows, got {p:?}"),
        )
        .map(|()| Placement::Replicated),
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
        (_, Owned { .. }) | (Owned { .. }, _) => Err(TypeError {
            kind: ErrorKind::Placement,
            message: format!("{s:?} cannot combine {p:?} with {q:?}"),
        }),
        (a, b) if shift(b, lead) == a && a != Partial => Ok(a),
        (Split { axis }, Replicated) if (axis as usize) < lead => Ok(p),
        _ => Err(TypeError {
            kind: ErrorKind::Placement,
            message: format!("{s:?} cannot combine {p:?} with {q:?}"),
        }),
    }
}

/// An operand's split axis, counted in the row input's axes.
fn shift(q: Placement, lead: usize) -> Placement {
    match q {
        Placement::Split { axis } => Placement::Split {
            axis: axis + lead as u32,
        },
        q => q,
    }
}

/// `x` without `axis`, as a sum over whole values.
fn mean(axis: u32, x: &TensorType) -> Result<TensorType, TypeError> {
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
fn mix(dims: &[Dim], axis: u32, x: &TensorType, w: &TensorType) -> Result<TensorType, TypeError> {
    let out = collapse(axis, x)?;
    let lead = axis as usize + 1;
    let fits = w.rank() == lead && same_shape(dims, &front(&x.shape, lead), &w.shape)?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "mix weights {} do not match {} up to axis {axis}",
            w.shape, x.shape
        )
    })?;
    need(w.dtype == x.dtype, ErrorKind::DType, || {
        "mix weights dtype differs".into()
    })?;
    need(
        w.placement == Placement::Replicated,
        ErrorKind::Placement,
        || "mix weights must be replicated".into(),
    )?;
    Ok(out)
}

/// Whether a known `width` is a multiple of `block`.
fn blocks(dims: &[Dim], width: &DimExpr, block: u32) -> Result<bool, TypeError> {
    let n = width.poly(dims).map_err(|d| TypeError {
        kind: ErrorKind::Reference,
        message: format!("{d:?} is not a dimension"),
    })?;
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

/// `slots` and `positions`: one integer each per row of `rows`.
pub(crate) fn row_ids(
    dims: &[Dim],
    rows: &Shape,
    slots: &TensorType,
    positions: &TensorType,
) -> Result<(), TypeError> {
    for (what, t) in [("slots", slots), ("positions", positions)] {
        let fits = t.rank() == 1 && same(dims, &t.shape.0[0], &rows.0[0])?;
        need(fits, ErrorKind::Shape, || {
            format!("{what} {} do not match rows {rows}", t.shape)
        })?;
        need(t.dtype.is_int(), ErrorKind::DType, || {
            format!("{what} must be integers, got {:?}", t.dtype)
        })?;
    }
    Ok(())
}

/// Rows go to kept `[Seqs, Context, ..]` by slot and position.
fn append(dims: &[Dim], kept: &TensorType, a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (rows, slots, positions) = (a[0], a[1], a[2]);
    let rest = |t: &TensorType, skip| Shape(t.shape.0.iter().skip(skip).cloned().collect());
    let fits = kept.rank() >= 2
        && rows.rank() + 1 == kept.rank()
        && same_shape(dims, &rest(kept, 2), &rest(rows, 1))?;
    need(fits, ErrorKind::Shape, || {
        format!("rows {} cannot join kept {}", rows.shape, kept.shape)
    })?;
    row_ids(dims, &rows.shape, slots, positions)?;
    need(rows.dtype == kept.dtype, ErrorKind::DType, || {
        format!("rows are {:?}, kept rows {:?}", rows.dtype, kept.dtype)
    })?;
    // Kept rows have one more leading axis than new rows.
    need(
        kept.placement == shift(rows.placement, 1)
            && !matches!(
                rows.placement,
                Placement::Partial | Placement::Routed | Placement::Owned { .. }
            ),
        ErrorKind::Placement,
        || {
            let (r, k) = (rows.placement, kept.placement);
            format!("{r:?} rows cannot join {k:?} kept rows")
        },
    )?;
    Ok(kept.clone())
}
