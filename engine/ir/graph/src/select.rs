//! Typing rules for picks by index and for masks.

use crate::dims::{Dim, DimExpr, Shape};
use crate::infer::{ErrorKind, TypeError, front, need, poly, same_shape};
use crate::node::Select;
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
        Placement::Split { axis } => (axis as usize) + axes < t.rank(),
        Placement::Partial | Placement::Routed | Placement::Owned => false,
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
        Partial | Routed | Owned => {
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
        matches!(x.placement, Placement::Replicated | Placement::Split { .. }),
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
