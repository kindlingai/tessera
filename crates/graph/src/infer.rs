//! Per-op typing rules: shapes, dtypes and placements.

use crate::dims::{Dim, DimExpr, DimName, Poly, Shape};
use crate::node::{Graph, Op, Select, Stage, View};
use crate::types::{Placement, TensorType};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Arity,
    /// A constant that JSON cannot hold, like infinity.
    Value,
    Reference,
    Shape,
    DType,
    Placement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    pub kind: ErrorKind,
    pub message: String,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} error: {}", self.kind, self.message)
    }
}

impl std::error::Error for TypeError {}

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

fn poly(dims: &[Dim], e: &DimExpr) -> Result<Poly, TypeError> {
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
        Op::Matmul | Op::Select(_) => 2,
        Op::Attend(_) => 3,
        Op::Row(r) => 1 + r.0.iter().map(Stage::operands).sum::<usize>(),
        Op::View(_) | Op::Cast(_) => 1,
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
        Op::Attend(a) => {
            finite(a.scale, "attention scale")?;
            attend(&g.dims, args)
        }
        Op::Row(r) => row(&g.dims, &r.0, args),
        Op::Select(Select::Gather) => gather(args[0], args[1]),
        Op::View(v) => view(&g.dims, v, args[0]),
        Op::Cast(dtype) => {
            // Rounding each rank's share is not rounding the sum.
            need(
                args[0].placement != Placement::Partial,
                ErrorKind::Placement,
                || "cast of partial sums".into(),
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
    need(w.rank() == 2 && x.rank() >= 1, ErrorKind::Shape, || {
        format!(
            "matmul takes x [.., K] and w [N, K], got {} and {}",
            x.shape, w.shape
        )
    })?;
    let k = x.shape.0.last().expect("rank checked");
    need(same(dims, k, &w.shape.0[1])?, ErrorKind::Shape, || {
        format!("matmul K differs: x {} and w {}", x.shape, w.shape)
    })?;
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
    let placement = match (x.placement, w.placement) {
        (Replicated, Replicated) => Replicated,
        (Replicated, Split { axis: 0 }) => Split { axis: last },
        (Split { axis }, Replicated) if axis < last => Split { axis },
        (Partial, Replicated) => Partial,
        (Split { axis }, Split { axis: 1 }) if axis == last => Partial,
        (xp, wp) => {
            return Err(TypeError {
                kind: ErrorKind::Placement,
                message: format!("matmul cannot combine x {xp:?} with w {wp:?}"),
            });
        }
    };
    let mut shape = x.shape.0.clone();
    *shape.last_mut().expect("rank checked") = w.shape.0[0].clone();
    Ok(TensorType {
        shape: Shape(shape),
        dtype: x.dtype,
        placement,
    })
}

fn attend(dims: &[Dim], a: &[&TensorType]) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    need(a.iter().all(|t| t.rank() == 4), ErrorKind::Shape, || {
        "attention takes [Seqs, Context, Heads, HeadDim] tensors".into()
    })?;
    need(
        same_shape(dims, &k.shape, &v.shape)?,
        ErrorKind::Shape,
        || format!("keys {} and values {} differ", k.shape, v.shape),
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
    let heads = poly(dims, &q.shape.0[2])?.as_const();
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
    let p = q.placement;
    need(
        k.placement == p && v.placement == p,
        ErrorKind::Placement,
        || "queries, keys and values must share a placement".into(),
    )?;
    need(
        matches!(p, Placement::Replicated | Placement::Split { axis: 2 }),
        ErrorKind::Placement,
        || format!("attention needs whole or head-split tensors, got {p:?}"),
    )?;
    Ok(q.clone())
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
            (Stage::Softmax, _) => whole(p, last, s)?,
            (Stage::Act(_), _) => need(p != Placement::Partial, ErrorKind::Placement, || {
                format!("{s:?} is nonlinear, so it cannot read partial sums")
            })?,
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

fn whole(p: Placement, last: u32, s: &Stage) -> Result<(), TypeError> {
    let ok = match p {
        Placement::Replicated => true,
        Placement::Split { axis } => axis < last,
        Placement::Partial => false,
    };
    need(ok, ErrorKind::Placement, || {
        format!("{s:?} needs whole rows, got {p:?}")
    })
}

fn gather(table: &TensorType, ids: &TensorType) -> Result<TensorType, TypeError> {
    use Placement::*;
    need(table.rank() >= 2, ErrorKind::Shape, || {
        format!("gather reads rows of a table, got {}", table.shape)
    })?;
    need(ids.dtype.is_int(), ErrorKind::DType, || {
        format!("gather indices must be integers, got {:?}", ids.dtype)
    })?;
    need(ids.placement == Replicated, ErrorKind::Placement, || {
        "gather indices must be replicated".into()
    })?;
    let lead = ids.rank() as u32;
    let placement = match table.placement {
        Replicated => Replicated,
        // Ranks without the row contribute zeros.
        Split { axis: 0 } => Partial,
        Split { axis } => Split {
            axis: axis - 1 + lead,
        },
        Partial => {
            return Err(TypeError {
                kind: ErrorKind::Placement,
                message: "gather cannot read a partial table".into(),
            });
        }
    };
    let mut shape = ids.shape.0.clone();
    shape.extend(table.shape.0[1..].iter().cloned());
    Ok(TensorType {
        shape: Shape(shape),
        dtype: table.dtype,
        placement,
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
                _ => end == size,
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
