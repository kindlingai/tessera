//! Typing rules for attention and its optional arguments.

use crate::dims::Dim;
use crate::infer::{
    ErrorKind, TypeError, front, groups, key_lists, need, poly, row_ids, same, same_shape, valued,
};
use crate::node::Attend;
use crate::types::{Placement, TensorType};

/// Keys match values, heads group, split by query head.
pub(crate) fn heads_fit(
    dims: &[Dim],
    [q, k, v]: [&TensorType; 3],
    q_heads: usize,
    kv_axis: usize,
    ragged: bool,
) -> Result<(), TypeError> {
    // Values may be narrower than keys, as in MLA.
    need(
        same_shape(
            dims,
            &front(&k.shape, kv_axis + 1),
            &front(&v.shape, kv_axis + 1),
        )?,
        ErrorKind::Shape,
        || format!("keys {} and values {} differ", k.shape, v.shape),
    )?;
    let kv_heads = groups(
        dims,
        &q.shape.0[q_heads],
        &k.shape.0[kv_axis],
        "query",
        "key",
    )?;
    need(
        q.dtype.is_float() && q.dtype == k.dtype && k.dtype == v.dtype,
        ErrorKind::DType,
        || "attention needs one float dtype".into(),
    )?;
    let p = [q.placement, k.placement, v.placement];
    let split =
        [q_heads as u32, kv_axis as u32, kv_axis as u32].map(|axis| Placement::Split { axis });
    // One key head serves query heads split any way.
    let shared = kv_heads == Some(1) && p[1..] == [Placement::Replicated; 2] && p[0] == split[0];
    // Replicated caches serve owned query rows directly.
    let owned = ragged
        && p == [
            Placement::Owned { axis: 0 },
            Placement::Replicated,
            Placement::Replicated,
        ];
    need(
        p == [Placement::Replicated; 3] || p == split || shared || owned,
        ErrorKind::Placement,
        || format!("attention needs whole or head-split tensors, got {p:?}"),
    )
}

pub(crate) fn attend(
    dims: &[Dim],
    op: &Attend,
    a: &[&TensorType],
) -> Result<TensorType, TypeError> {
    let (q, k, v) = (a[0], a[1], a[2]);
    let (q_heads, kv_axis) = if op.packed() {
        need(
            a[..3].iter().all(|t| t.rank() == 3),
            ErrorKind::Shape,
            || "packed attention takes [Rows, Heads, HeadDim] tensors".into(),
        )?;
        for axis in [0, 2] {
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
        let b = a[3];
        let b_fit = b.rank() == 2
            && same(dims, &b.shape.0[0], &q.shape.0[0])?
            && poly(dims, &b.shape.0[1])?.as_const() == Some(2);
        need(b_fit, ErrorKind::Shape, || {
            format!("bounds {} must be [Rows, 2] for {}", b.shape, q.shape)
        })?;
        need(b.dtype.is_int(), ErrorKind::DType, || {
            format!("bounds must be integers, got {:?}", b.dtype)
        })?;
        need(
            b.placement == Placement::Replicated,
            ErrorKind::Placement,
            || "bounds must be replicated".into(),
        )?;
        (1, 1)
    } else if op.ragged {
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
        row_ids(dims, &q.shape, a[3], a[4])?;
        (1, 2)
    } else {
        need(
            a[..3].iter().all(|t| t.rank() == 4),
            ErrorKind::Shape,
            || "attention takes [Seqs, Context, Heads, HeadDim] tensors".into(),
        )?;
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
        (2, 2)
    };
    heads_fit(dims, [q, k, v], q_heads, kv_axis, op.ragged)?;
    if let Some(i) = op.ids_arg() {
        key_lists(dims, a[i], q, q_heads)?;
    }
    if let Some(i) = op.sinks_arg() {
        let s = a[i];
        if s.rank() == 1 {
            let ok = same(dims, &s.shape.0[0], &q.shape.0[q_heads])?;
            need(ok, ErrorKind::Shape, || {
                format!("sinks {} vs {}", s.shape, q.shape)
            })?;
            let want = match q.placement {
                Placement::Split { axis } if axis == q_heads as u32 => Placement::Split { axis: 0 },
                _ => Placement::Replicated,
            };
            need(s.placement == want, ErrorKind::Placement, || {
                format!("sinks {:?} want {want:?}", s.placement)
            })?;
            let d_ok = s.dtype.is_float() && s.dtype == q.dtype;
            need(d_ok, ErrorKind::DType, || "sink dtype mismatch".into())?;
        } else {
            per_head(dims, "sinks", s, q, q_heads + 1)?;
        }
    }
    if let Some(i) = op.rel_bias_arg() {
        per_head(dims, "relative bias", a[i], q, q_heads + 2)?;
    }
    Ok(valued(q, v))
}

/// `t` leads with each query row and head.
fn per_head(
    dims: &[Dim],
    what: &str,
    t: &TensorType,
    q: &TensorType,
    rank: usize,
) -> Result<(), TypeError> {
    let lead = q.rank() - 1;
    let fits =
        t.rank() == rank && same_shape(dims, &front(&t.shape, lead), &front(&q.shape, lead))?;
    need(fits, ErrorKind::Shape, || {
        format!("{what} {} vs {}", t.shape, q.shape)
    })?;
    let d_ok = t.dtype.is_float() && t.dtype == q.dtype;
    need(d_ok, ErrorKind::DType, || format!("{what} dtype mismatch"))?;
    need(t.placement == q.placement, ErrorKind::Placement, || {
        format!("{what} {:?} want {:?}", t.placement, q.placement)
    })
}
