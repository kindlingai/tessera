//! Per-op typing rules: shapes, dtypes and placements.

use crate::dims::{Dim, DimExpr, DimName, Poly, Shape};
use crate::node::{Graph, Op, Select, Stage, View};
use crate::select;
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

fn same(dims: &[Dim], a: &DimExpr, b: &DimExpr) -> Result<bool, TypeError> {
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

/// The output axis with the same elements before it.
fn moved_axis(dims: &[Dim], from: &Shape, to: &Shape, axis: u32) -> Result<Option<u32>, TypeError> {
    let before = elements(dims, &Shape(from.0[..axis as usize].to_vec()))?;
    let one = Poly::constant(1);
    let mut prefix = one.clone();
    for (b, e) in to.0.iter().enumerate() {
        let size = poly(dims, e)?;
        if prefix == before && size != one {
            return Ok(Some(b as u32));
        }
        prefix = prefix.product(&size);
    }
    Ok(None)
}

fn arity(op: &Op) -> usize {
    match op {
        Op::Input { .. } | Op::Weight(_) => 0,
        Op::Select(Select::TopK { .. }) | Op::Mask(_) | Op::Cast(_) | Op::Mean(_) => 1,
        Op::Append(_) => 3,
        Op::Matmul | Op::Select(_) | Op::View(View::Concat) | Op::Mix(_) => 2,
        Op::Attend(a) => 3 + usize::from(a.picked) + 2 * usize::from(a.ragged),
        Op::Delta(_) => 5,
        Op::Conv => 2,
        Op::Row(r) => 1 + r.0.iter().map(Stage::operands).sum::<usize>(),
        Op::View(_) => 1,
    }
}

/// The type `op` gives `args`, or why they don't fit.
pub fn infer(g: &Graph, op: &Op, args: &[&TensorType]) -> Result<TensorType, TypeError> {
    let want = arity(op);
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
        Op::Conv => conv(&g.dims, args[0], args[1]),
        Op::Attend(a) if a.ragged => {
            finite(a.scale, "attention scale")?;
            need(a.mask.causal(), ErrorKind::Value, || {
                format!("ragged attention is causal only, got {:?}", a.mask)
            })?;
            attend_rows(&g.dims, args)
        }
        Op::Attend(a) => {
            finite(a.scale, "attention scale")?;
            attend(&g.dims, args)
        }
        Op::Delta(d) => {
            finite(d.scale, "delta rule scale")?;
            delta(&g.dims, args)
        }
        Op::Row(r) => row(&g.dims, &r.0, args),
        Op::Mask(_) => select::mask(args[0]),
        Op::Append(i) => {
            let kept = g.kept.get(*i as usize).ok_or_else(missing)?;
            append(&g.dims, kept, args)
        }
        Op::Select(s) => select::select(&g.dims, s, args),
        Op::View(View::Concat) => concat(&g.dims, args[0], args[1]),
        Op::View(v) => view(&g.dims, v, args[0]),
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
        (Split { axis }, Split { axis: k }) if axis == last && k == n + 1 => Partial,
        // Both split along one shared batch axis: each part alone.
        (Split { axis }, Split { axis: k }) if axis == k && (axis as usize) < own => Split { axis },
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

/// Keys match values, heads group, split by query head.
fn heads_fit(dims: &[Dim], [q, k, v]: [&TensorType; 3], q_heads: usize) -> Result<(), TypeError> {
    // Values may be narrower than keys, as in MLA.
    need(
        same_shape(dims, &front(&k.shape, 3), &front(&v.shape, 3))?,
        ErrorKind::Shape,
        || format!("keys {} and values {} differ", k.shape, v.shape),
    )?;
    let heads = poly(dims, &q.shape.0[q_heads])?.as_const();
    let kv_heads = poly(dims, &k.shape.0[2])?.as_const();
    if let (Some(h), Some(kv)) = (heads, kv_heads) {
        need(kv != 0 && h % kv == 0, ErrorKind::Shape, || {
            format!("{h} query heads do not group over {kv} key heads")
        })?;
    }
    need(
        q.dtype.is_float() && q.dtype == k.dtype && k.dtype == v.dtype,
        ErrorKind::DType,
        || "attention needs one float dtype".into(),
    )?;
    let p = [q.placement, k.placement, v.placement];
    let split = [q_heads as u32, 2, 2].map(|axis| Placement::Split { axis });
    // One key head serves query heads split any way.
    let shared = kv_heads == Some(1) && p[1..] == [Placement::Replicated; 2] && p[0] == split[0];
    need(
        p == [Placement::Replicated; 3] || p == split || shared,
        ErrorKind::Placement,
        || format!("attention needs whole or head-split tensors, got {p:?}"),
    )
}

fn attend(dims: &[Dim], a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    need(
        a[..3].iter().all(|t| t.rank() == 4),
        ErrorKind::Shape,
        || "attention takes [Seqs, Context, Heads, HeadDim] tensors".into(),
    )?;
    // Query and key lengths may differ, as in decode.
    for axis in [0, 3] {
        need(
            same(dims, &q.shape.0[axis], &k.shape.0[axis])?,
            ErrorKind::Shape,
            || {
                format!(
                    "queries {} and keys {} differ on axis {axis}",
                    q.shape, k.shape
                )
            },
        )?;
    }
    heads_fit(dims, [q, k, v], 2)?;
    if let Some(ids) = a.get(3) {
        key_lists(dims, ids, q, 2)?;
    }
    Ok(valued(q, v))
}

/// Rows `[Seqs, Context, C]` through taps `[C, K]`.
fn conv(dims: &[Dim], x: &TensorType, w: &TensorType) -> Result<TensorType, TypeError> {
    let fits = x.rank() == 3 && w.rank() == 2 && same(dims, &x.shape.0[2], &w.shape.0[0])?;
    need(fits, ErrorKind::Shape, || {
        format!(
            "conv takes x [Seqs, Context, C] and w [C, K], got {} and {}",
            x.shape, w.shape
        )
    })?;
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
    })?;
    Ok(x.clone())
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
    heads_fit(dims, [v, q, k], 2)?;
    for gate in &a[3..] {
        let fits = same_shape(dims, &gate.shape, &front(&v.shape, 3))?;
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
    Ok(v.clone())
}

/// One integer key list per query, over `lead` axes.
fn key_lists(dims: &[Dim], ids: &TensorType, q: &TensorType, lead: usize) -> Result<(), TypeError> {
    let fits = ids.rank() == lead + 1
        && same_shape(dims, &front(&ids.shape, lead), &front(&q.shape, lead))?;
    need(fits, ErrorKind::Shape, || {
        format!("key lists {} do not match queries {}", ids.shape, q.shape)
    })?;
    need(ids.dtype.is_int(), ErrorKind::DType, || {
        format!("key lists must be integers, got {:?}", ids.dtype)
    })?;
    need(
        ids.placement == Placement::Replicated,
        ErrorKind::Placement,
        || "key lists must be replicated".into(),
    )
}

/// Queries' type with the values' width.
fn valued(q: &TensorType, v: &TensorType) -> TensorType {
    let mut out = q.clone();
    let width = v.shape.0.last().expect("rank checked").clone();
    *out.shape.0.last_mut().expect("rank checked") = width;
    out
}

/// Positions must lead `x`. Then `x` keeps whole pairs.
fn rope(dims: &[Dim], x: &TensorType, pos: &TensorType, p: Placement) -> Result<(), TypeError> {
    let width = poly(dims, x.shape.0.last().expect("rank checked"))?;
    let even = width.as_const().is_none_or(|w| w % 2 == 0);
    need(even, ErrorKind::Shape, || {
        format!("rope needs an even width, got {}", x.shape)
    })?;
    positions(dims, x, pos, p)
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
    let placed = pos.placement == Placement::Replicated && p != Placement::Split { axis: last };
    need(placed, ErrorKind::Placement, || {
        format!("{p:?} rows cannot follow {:?} positions", pos.placement)
    })
}

/// One `u32` word per 32 entries of each row.
fn allowed(dims: &[Dim], x: &TensorType, bits: &TensorType) -> Result<(), TypeError> {
    let lead = x.rank() - 1;
    let words = |s: &DimExpr| s.poly(dims).ok().and_then(|n| n.as_const());
    let width = words(&x.shape.0[lead]);
    let fits = bits.rank() == x.rank()
        && same_shape(dims, &front(&x.shape, lead), &front(&bits.shape, lead))?
        && words(&bits.shape.0[lead]) == width.map(|w| w.div_ceil(32));
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
    let mut p = x.placement;
    let mut rest = args[1..].iter();
    for s in stages {
        let operand = (s.operands() == 1).then(|| rest.next().expect("arity checked"));
        match (s, operand) {
            (Stage::Add | Stage::Mul, Some(o)) => {
                broadcast(dims, &x.shape, &o.shape)?;
                need(o.dtype == x.dtype, ErrorKind::DType, || {
                    format!("{s:?} mixes {:?} and {:?}", x.dtype, o.dtype)
                })?;
                p = combine(s, p, o.placement, x.rank() - o.rank())?;
            }
            (Stage::Scale(by), _) => finite(*by, "scale")?,
            (Stage::RmsNorm { eps }, Some(w)) => {
                finite(*eps, "norm epsilon")?;
                let width = std::slice::from_ref(x.shape.0.last().expect("rank checked"));
                need(
                    same_shape(dims, &w.shape, &Shape(width.to_vec()))?,
                    ErrorKind::Shape,
                    || format!("norm weight {} does not match rows of {}", w.shape, x.shape),
                )?;
                need(w.dtype == x.dtype, ErrorKind::DType, || {
                    "norm weight dtype differs".into()
                })?;
                need(
                    w.placement == Placement::Replicated,
                    ErrorKind::Placement,
                    || "norm weights must be replicated".into(),
                )?;
                whole(p, last, s)?;
            }
            (Stage::Rms { eps }, _) => {
                finite(*eps, "norm epsilon")?;
                whole(p, last, s)?;
            }
            (Stage::Hadamard { block }, _) => {
                let width = x.shape.0.last().expect("rank checked");
                let fits = block.is_power_of_two() && blocks(dims, width, *block)?;
                need(fits, ErrorKind::Shape, || {
                    format!("rows of {width} do not hold whole blocks of {block}")
                })?;
                // Within a row, so routed zero rows stay zero.
                if p != Placement::Routed {
                    whole(p, last, s)?;
                }
            }
            (Stage::Softmax | Stage::Center | Stage::UnitSum, _) => whole(p, last, s)?,
            (Stage::Rope { theta, .. }, Some(pos)) => {
                need(theta.is_finite() && *theta > 0.0, ErrorKind::Value, || {
                    format!("rope base is {theta}")
                })?;
                rope(dims, x, pos, p)?;
            }
            (Stage::Causal, Some(pos)) => {
                positions(dims, x, pos, p)?;
                whole(p, last, s)?;
            }
            (Stage::Allow, Some(bits)) => {
                allowed(dims, x, bits)?;
                whole(p, last, s)?;
            }
            (Stage::Sin | Stage::Cos, _) => {
                need(p == Placement::Replicated, ErrorKind::Placement, || {
                    format!("{s:?} is nonlinear, so it cannot read {p:?} values")
                })?;
            }
            (Stage::Act(a), _) => {
                // Routed zeros stay zero only where the activation keeps zero.
                let ok = p != Placement::Partial && (p != Placement::Routed || a.keeps_zero());
                need(ok, ErrorKind::Placement, || {
                    format!("{s:?} is nonlinear, so it cannot read {p:?} values")
                })?;
            }
            (Stage::AllReduce, _) => {
                need(p == Placement::Partial, ErrorKind::Placement, || {
                    format!("all-reduce needs partial sums, got {p:?}")
                })?;
                p = Placement::Replicated;
            }
            _ => {}
        }
    }
    Ok(TensorType {
        shape: x.shape.clone(),
        dtype: x.dtype,
        placement: p,
    })
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

fn combine(s: &Stage, p: Placement, q: Placement, lead: usize) -> Result<Placement, TypeError> {
    use Placement::*;
    match (p, q) {
        (Partial, Partial) if *s == Stage::Add => Ok(Partial),
        (Partial, Replicated) if *s == Stage::Mul => Ok(Partial),
        (Routed, Routed) => Ok(Routed),
        (Routed, Replicated) | (Replicated, Routed) if *s == Stage::Mul => Ok(Routed),
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
    need(x.dtype.is_float(), ErrorKind::DType, || {
        format!("a mean needs floats, got {:?}", x.dtype)
    })?;
    need((axis as usize) < x.rank(), ErrorKind::Shape, || {
        format!("no axis {axis} in {}", x.shape)
    })?;
    need(
        x.placement != Placement::Partial,
        ErrorKind::Placement,
        || "a mean cannot read partial sums".into(),
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
    let out = mean(axis, x)?;
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
        Placement::Split { axis } => axis < last,
        Placement::Partial | Placement::Routed => false,
    };
    need(ok, ErrorKind::Placement, || {
        format!("{s:?} needs whole rows, got {p:?}")
    })
}

fn view(dims: &[Dim], v: &View, x: &TensorType) -> Result<TensorType, TypeError> {
    let mut out = x.clone();
    match v {
        View::Reshape(s) => {
            need(
                elements(dims, &x.shape)? == elements(dims, s)?,
                ErrorKind::Shape,
                || format!("reshape {} to {s} changes the element count", x.shape),
            )?;
            if let Placement::Split { axis } = x.placement {
                let to = moved_axis(dims, &x.shape, s, axis)?;
                need(to.is_some(), ErrorKind::Placement, || {
                    format!("reshape {} to {s} breaks split axis {axis}", x.shape)
                })?;
                out.placement = Placement::Split {
                    axis: to.expect("checked"),
                };
            }
            out.shape = s.clone();
        }
        View::Slice { axis, start, len } => {
            let a = *axis as usize;
            need(a < x.rank(), ErrorKind::Shape, || {
                format!("no axis {a} in {}", x.shape)
            })?;
            let end = poly(dims, start)?.sum(&poly(dims, len)?);
            let size = poly(dims, &x.shape.0[a])?;
            let fits = match (end.as_const(), size.as_const()) {
                (Some(e), Some(s)) => e <= s,
                _ => size.covers(&end),
            };
            need(fits, ErrorKind::Shape, || {
                format!("slice {start}+{len} runs past axis {a} of {}", x.shape)
            })?;
            need(
                x.placement != Placement::Split { axis: *axis },
                ErrorKind::Placement,
                || "slice along the split axis".into(),
            )?;
            out.shape.0[a] = len.clone();
        }
        View::Expand { axis, to } => {
            let a = *axis as usize;
            let one = a < x.rank() && poly(dims, &x.shape.0[a])? == Poly::constant(1);
            need(one, ErrorKind::Shape, || {
                format!("axis {a} of {} is not size one", x.shape)
            })?;
            poly(dims, to)?;
            out.shape.0[a] = to.clone();
        }
        View::Concat => unreachable!("concat takes two arguments"),
        View::Permute(perm) => {
            let mut seen: Vec<u32> = perm.clone();
            seen.sort();
            need(
                seen.iter().copied().eq(0..x.rank() as u32),
                ErrorKind::Shape,
                || format!("{perm:?} does not permute {}", x.shape),
            )?;
            out.shape = Shape(
                perm.iter()
                    .map(|&i| x.shape.0[i as usize].clone())
                    .collect(),
            );
            if let Placement::Split { axis } = x.placement {
                let to = perm.iter().position(|&i| i == axis).expect("checked");
                out.placement = Placement::Split { axis: to as u32 };
            }
        }
    }
    Ok(out)
}

fn concat(dims: &[Dim], a: &TensorType, b: &TensorType) -> Result<TensorType, TypeError> {
    let lead = a.rank().saturating_sub(1);
    let fits = a.rank() >= 1
        && a.rank() == b.rank()
        && same_shape(dims, &front(&a.shape, lead), &front(&b.shape, lead))?;
    need(fits, ErrorKind::Shape, || {
        format!("{} and {} differ before the last axis", a.shape, b.shape)
    })?;
    need(a.dtype == b.dtype, ErrorKind::DType, || {
        format!("concat mixes {:?} and {:?}", a.dtype, b.dtype)
    })?;
    let split = Placement::Split { axis: lead as u32 };
    need(
        a.placement == b.placement && a.placement != split,
        ErrorKind::Placement,
        || format!("concat cannot join {:?} and {:?}", a.placement, b.placement),
    )?;
    let mut out = a.clone();
    out.shape.0[lead] = a.shape.0[lead].clone() + b.shape.0[lead].clone();
    Ok(out)
}

/// `slots` and `positions`: one integer each per row of `rows`.
fn row_ids(
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

/// Queries `[Rows, Heads, HeadDim]` over kept `[Seqs, Context, KvHeads, HeadDim]`.
fn attend_rows(dims: &[Dim], a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    need(
        q.rank() == 3 && k.rank() == 4 && v.rank() == 4,
        ErrorKind::Shape,
        || "ragged attention takes [Rows, Heads, HeadDim] queries over kept rows".into(),
    )?;
    need(
        same(dims, &q.shape.0[2], &k.shape.0[3])?,
        ErrorKind::Shape,
        || {
            format!(
                "queries {} and keys {} differ in head width",
                q.shape, k.shape
            )
        },
    )?;
    heads_fit(dims, [q, k, v], 1)?;
    row_ids(dims, &q.shape, a[3], a[4])?;
    if let Some(ids) = a.get(5) {
        key_lists(dims, ids, q, 1)?;
    }
    Ok(valued(q, v))
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
            && !matches!(rows.placement, Placement::Partial | Placement::Routed),
        ErrorKind::Placement,
        || {
            let (r, k) = (rows.placement, kept.placement);
            format!("{r:?} rows cannot join {k:?} kept rows")
        },
    )?;
    Ok(kept.clone())
}
