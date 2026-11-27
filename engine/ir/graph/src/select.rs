//! Typing rules for picks by index and for masks.

use crate::dims::{Dim, DimExpr, Poly, Shape};
use crate::infer::{ErrorKind, TypeError, elements, front, need, poly, same, same_shape};
use crate::node::{Select, View};
use crate::types::{DType, Placement, TensorType};

pub(crate) fn select(
    dims: &[Dim],
    s: &Select,
    args: &[&TensorType],
) -> Result<TensorType, TypeError> {
    match s {
        Select::Gather => gather(args[0], args[1]),
        Select::Take => take(dims, args[0], args[1]),
        Select::TopK { k } => top_k(dims, k, args[0]),
    }
}

/// Whether each rank holds the last `axes` axes whole.
fn whole(t: &TensorType, axes: usize) -> bool {
    match t.placement {
        Placement::Replicated => true,
        Placement::Split { axis } | Placement::Owned { axis } => (axis as usize) + axes < t.rank(),
        Placement::Partial | Placement::Routed => false,
    }
}

fn floats(t: &TensorType, what: &str) -> Result<(), TypeError> {
    need(t.dtype.is_float(), ErrorKind::DType, || {
        format!("{what} reads floats, got {:?}", t.dtype)
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
        // Ranks without the row hold zeros.
        Split { axis: 0 } => Routed,
        Split { axis } => Split {
            axis: axis - 1 + lead,
        },
        Partial | Routed | Owned { .. } => {
            return Err(TypeError {
                kind: ErrorKind::Placement,
                message: format!("gather cannot read a {:?} table", table.placement),
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

fn take(dims: &[Dim], x: &TensorType, ids: &TensorType) -> Result<TensorType, TypeError> {
    let lead = x.rank().saturating_sub(1);
    let fits = x.rank() >= 1
        && x.rank() == ids.rank()
        && same_shape(dims, &front(&x.shape, lead), &front(&ids.shape, lead))?;
    need(fits, ErrorKind::Shape, || {
        format!("indices {} do not match rows of {}", ids.shape, x.shape)
    })?;
    need(ids.dtype.is_int(), ErrorKind::DType, || {
        format!("take indices must be integers, got {:?}", ids.dtype)
    })?;
    need(
        matches!(x.placement, Placement::Replicated | Placement::Split { .. })
            && ids.placement == x.placement,
        ErrorKind::Placement,
        || format!("take cannot read {:?} at {:?}", x.placement, ids.placement),
    )?;
    Ok(TensorType {
        shape: ids.shape.clone(),
        dtype: x.dtype,
        placement: x.placement,
    })
}

fn top_k(dims: &[Dim], k: &DimExpr, x: &TensorType) -> Result<TensorType, TypeError> {
    need(x.rank() >= 1, ErrorKind::Shape, || {
        "top-k needs rank 1 or more".into()
    })?;
    floats(x, "top-k")?;
    let fixed = poly(dims, k)?.as_const().is_some_and(|k| k > 0);
    need(fixed, ErrorKind::Shape, || {
        format!("top-k needs a fixed positive count, got {k}")
    })?;
    need(
        matches!(x.placement, Placement::Replicated | Placement::Split { .. })
            || matches!(x.placement, Placement::Owned { axis } if (axis as usize) + 1 < x.rank()),
        ErrorKind::Placement,
        || format!("top-k cannot read {:?}", x.placement),
    )?;
    let mut shape = x.shape.0.clone();
    *shape.last_mut().expect("rank checked") = k.clone();
    Ok(TensorType {
        shape: Shape(shape),
        dtype: DType::U32,
        placement: x.placement,
    })
}

pub(crate) fn mask(x: &TensorType) -> Result<TensorType, TypeError> {
    need(x.rank() >= 2, ErrorKind::Shape, || {
        format!("a mask takes [.., Queries, Keys], got {}", x.shape)
    })?;
    floats(x, "a mask")?;
    need(whole(x, 2), ErrorKind::Placement, || {
        format!("a mask needs whole score rows, got {:?}", x.placement)
    })?;
    Ok(x.clone())
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

pub(crate) fn view(dims: &[Dim], v: &View, x: &TensorType) -> Result<TensorType, TypeError> {
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
            } else if let Placement::Owned { axis } = x.placement {
                let to = moved_axis(dims, &x.shape, s, axis)?;
                let kept = match to {
                    Some(b) => same(dims, &x.shape.0[axis as usize], &s.0[b as usize])?,
                    None => false,
                };
                need(kept, ErrorKind::Placement, || {
                    format!("reshape {} to {s} breaks owned axis {axis}", x.shape)
                })?;
                out.placement = Placement::Owned {
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
                !matches!(
                    x.placement,
                    Placement::Split { axis: p } | Placement::Owned { axis: p } if p == *axis
                ),
                ErrorKind::Placement,
                || "slice along a split or owned axis".into(),
            )?;
            out.shape.0[a] = len.clone();
        }
        View::Expand { axis, to } => {
            let a = *axis as usize;
            let one = a < x.rank() && poly(dims, &x.shape.0[a])? == Poly::constant(1);
            need(one, ErrorKind::Shape, || {
                format!("axis {a} of {} is not size one", x.shape)
            })?;
            need(
                x.placement != Placement::Owned { axis: *axis },
                ErrorKind::Placement,
                || "expand along an owned axis".into(),
            )?;
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
            } else if let Placement::Owned { axis } = x.placement {
                let to = perm.iter().position(|&i| i == axis).expect("checked");
                out.placement = Placement::Owned { axis: to as u32 };
            }
        }
    }
    Ok(out)
}

pub(crate) fn concat(
    dims: &[Dim],
    a: &TensorType,
    b: &TensorType,
) -> Result<TensorType, TypeError> {
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
    let on_last = matches!(
        a.placement,
        Placement::Split { axis } | Placement::Owned { axis } if axis == lead as u32
    );
    need(
        a.placement == b.placement && !on_last,
        ErrorKind::Placement,
        || format!("concat cannot join {:?} and {:?}", a.placement, b.placement),
    )?;
    let mut out = a.clone();
    out.shape.0[lead] = a.shape.0[lead].clone() + b.shape.0[lead].clone();
    Ok(out)
}
