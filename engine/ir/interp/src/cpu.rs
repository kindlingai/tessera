//! Plain loops: the reference every plan is checked against.

use crate::round::Acc;
use crate::tensor::Error;
use graph::{Act, Attend, Mask, Numerics, Pairs, Stage};
use mem::Tensor;

/// The tanh approximation, as the GPU kernels compute it.
fn gelu(x: f32) -> f32 {
    const COEF: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (COEF * (x + 0.044_715 * x * x * x)).tanh())
}

fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::GeluTanh => gelu(x),
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Act::Relu => x.max(0.0),
    }
}

/// Pair `j`'s rotation at `pos`, in a row `width` wide.
pub(crate) fn angle(pos: usize, j: usize, width: usize, theta: f32) -> f32 {
    pos as f32 * theta.powf(-((2 * j) as f32) / width as f32)
}

fn turn(row: &mut [f32], pos: usize, theta: f32, pairs: Pairs) {
    let half = row.len() / 2;
    for j in 0..half {
        let (a, b) = match pairs {
            Pairs::Adjacent => (2 * j, 2 * j + 1),
            Pairs::Halves => (j, j + half),
        };
        let (sin, cos) = angle(pos, j, row.len(), theta).sin_cos();
        (row[a], row[b]) = (row[a] * cos - row[b] * sin, row[a] * sin + row[b] * cos);
    }
}

fn softmax(acc: Acc, row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum = acc.add(sum, *v);
    }
    for v in row.iter_mut() {
        *v /= sum;
    }
}

/// Keys before the first query, under causal end alignment.
pub(crate) fn past_keys(a: &Attend, queries: usize, keys: usize) -> Result<usize, Error> {
    match keys.checked_sub(queries) {
        Some(past) => Ok(past),
        None if a.mask != Mask::Causal => Ok(0),
        None => Err(Error::Mismatch(format!("{queries} queries, {keys} keys"))),
    }
}

/// The keys `[from, to)` a query at `position` sees.
pub(crate) fn seen(mask: Mask, position: usize, keys: usize) -> (usize, usize) {
    match mask {
        Mask::Causal => (0, (position + 1).min(keys)),
        Mask::Bidirectional => (0, keys),
        Mask::Within(n) => {
            let n = n as usize;
            (position.saturating_sub(n), (position + n + 1).min(keys))
        }
    }
}

// These sum in fixed order with exact math.

pub(crate) fn matmul(n: &Numerics, x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let (m, k) = x.rows();
    let n = w.shape[0];
    let (xs, ws) = (x.floats()?, w.floats()?);
    let mut out = vec![0.0; m * n];
    for (r, row) in out.chunks_mut(n).enumerate() {
        let xr = &xs[r * k..(r + 1) * k];
        for (c, o) in row.iter_mut().enumerate() {
            *o = acc.dot(xr, &ws[c * k..(c + 1) * k]);
        }
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().expect("verified rank") = n;
    Ok(Tensor::f32(shape, out))
}

pub(crate) fn attend(n: &Numerics, a: &Attend, qkv: [&Tensor; 3]) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let [q, k, v] = qkv;
    let [seqs, ctx, heads, d] = q.shape[..] else {
        return Err(Error::Mismatch("attention wants rank 4".into()));
    };
    let (keys, kv_heads) = (k.shape[1], k.shape[2]);
    let group = heads / kv_heads;
    let past = past_keys(a, ctx, keys)?;
    let (qs, ks, vs) = (q.floats()?, k.floats()?, v.floats()?);
    let mut out = vec![0.0; qs.len()];
    let mut p = vec![0.0; keys];
    for s in 0..seqs {
        for t in 0..ctx {
            for h in 0..heads {
                let qo = ((s * ctx + t) * heads + h) * d;
                let kv_at = |j: usize| ((s * keys + j) * kv_heads + h / group) * d;
                let (from, to) = seen(a.mask, past + t, keys);
                for (j, pj) in p[from..to].iter_mut().enumerate() {
                    let j = from + j;
                    *pj = acc.dot(&qs[qo..qo + d], &ks[kv_at(j)..kv_at(j) + d]) * a.scale;
                }
                softmax(acc, &mut p[from..to]);
                let o = &mut out[qo..qo + d];
                for (j, &pj) in p[from..to].iter().enumerate() {
                    let j = from + j;
                    for (oi, &vi) in o.iter_mut().zip(&vs[kv_at(j)..kv_at(j) + d]) {
                        *oi = acc.add(*oi, pj * vi);
                    }
                }
            }
        }
    }
    Ok(Tensor::f32(q.shape.clone(), out))
}

/// Each query row reads its slot up to its position.
pub(crate) fn attend_rows(
    n: &Numerics,
    a: &Attend,
    [q, k, v]: [&Tensor; 3],
    [slots, positions]: [&Tensor; 2],
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let [rows, heads, d] = q.shape[..] else {
        return Err(Error::Mismatch(
            "ragged attention wants rank 3 queries".into(),
        ));
    };
    let (keys, kv_heads) = (k.shape[1], k.shape[2]);
    let group = heads / kv_heads;
    let (qs, ks, vs) = (q.floats()?, k.floats()?, v.floats()?);
    let (slots, positions) = (slots.indices()?, positions.indices()?);
    let mut out = vec![0.0; qs.len()];
    let mut p = vec![0.0; keys];
    for r in 0..rows {
        let (s, seen) = (slots[r], positions[r] + 1);
        if seen > keys {
            let msg = format!("row {r} at position {} is past {keys} keys", seen - 1);
            return Err(Error::Mismatch(msg));
        }
        for h in 0..heads {
            let qo = (r * heads + h) * d;
            let kv_at = |j: usize| ((s * keys + j) * kv_heads + h / group) * d;
            for (j, pj) in p[..seen].iter_mut().enumerate() {
                *pj = acc.dot(&qs[qo..qo + d], &ks[kv_at(j)..kv_at(j) + d]) * a.scale;
            }
            softmax(acc, &mut p[..seen]);
            let o = &mut out[qo..qo + d];
            for (j, &pj) in p[..seen].iter().enumerate() {
                for (oi, &vi) in o.iter_mut().zip(&vs[kv_at(j)..kv_at(j) + d]) {
                    *oi = acc.add(*oi, pj * vi);
                }
            }
        }
    }
    Ok(Tensor::f32(q.shape.clone(), out))
}

pub(crate) fn stage(
    n: &Numerics,
    s: &Stage,
    x: &Tensor,
    operand: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let mut out = x.floats()?.to_vec();
    let (_, width) = x.rows();
    match (s, operand) {
        (Stage::Add | Stage::Mul, Some(o)) => {
            let o = o.floats()?;
            for (i, v) in out.iter_mut().enumerate() {
                let b = o[i % o.len()];
                *v = if *s == Stage::Add { *v + b } else { *v * b };
            }
        }
        (Stage::Scale(by), _) => out.iter_mut().for_each(|v| *v *= by),
        (Stage::RmsNorm { eps }, Some(w)) => {
            let w = w.floats()?;
            for row in out.chunks_mut(width) {
                let inv = rms_inverse(acc, row, *eps);
                row.iter_mut().zip(w).for_each(|(v, wi)| *v *= inv * wi);
            }
        }
        (Stage::Hadamard { block }, _) => {
            let block = *block as usize;
            for span in out.chunks_mut(block) {
                hadamard(span);
            }
        }
        (Stage::Rms { eps }, _) => {
            for row in out.chunks_mut(width) {
                let inv = rms_inverse(acc, row, *eps);
                row.iter_mut().for_each(|v| *v *= inv);
            }
        }
        (Stage::Act(a), _) => out.iter_mut().for_each(|v| *v = act(*a, *v)),
        (Stage::Softmax, _) => out.chunks_mut(width).for_each(|r| softmax(acc, r)),
        // One rank holds the whole sum already.
        (Stage::AllReduce, _) => {}
        (Stage::Rope { theta, pairs }, Some(at)) => {
            let at = at.indices()?;
            // Each position covers the same number of elements.
            let span = out.len() / at.len().max(1);
            for (block, &pos) in out.chunks_mut(span.max(1)).zip(&at) {
                for row in block.chunks_mut(width) {
                    turn(row, pos, *theta, *pairs);
                }
            }
        }
        (Stage::Center, _) => out.chunks_mut(width).for_each(|r| {
            let mean = acc.sum(r.iter().copied()) / width as f32;
            r.iter_mut().for_each(|v| *v -= mean);
        }),
        (Stage::UnitSum, _) => out.chunks_mut(width).for_each(|r| {
            let sum = acc.sum(r.iter().copied());
            r.iter_mut().for_each(|v| *v /= sum);
        }),
        _ => return Err(Error::Mismatch(format!("{s:?} is missing its operand"))),
    }
    Ok(Tensor::f32(x.shape.clone(), out))
}

/// The mean over `axis`, in fixed order.
pub(crate) fn mean(n: &Numerics, x: &Tensor, axis: usize) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let count = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let outer: usize = x.shape[..axis].iter().product();
    let xs = x.floats()?;
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let along = (0..count).map(|t| xs[(o * count + t) * inner + i]);
            out.push(acc.sum(along) / count as f32);
        }
    }
    let mut shape = x.shape.clone();
    shape.remove(axis);
    Ok(Tensor::f32(shape, out))
}

/// The Walsh-Hadamard transform of `x`, scaled to keep its norm.
fn hadamard(x: &mut [f32]) {
    let mut len = 1;
    while len < x.len() {
        for pair in x.chunks_mut(2 * len) {
            let (low, high) = pair.split_at_mut(len);
            for (a, b) in low.iter_mut().zip(high) {
                (*a, *b) = (*a + *b, *a - *b);
            }
        }
        len *= 2;
    }
    let scale = 1.0 / (x.len() as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= scale);
}

/// One over the row's root mean square.
fn rms_inverse(acc: Acc, row: &[f32], eps: f32) -> f32 {
    let mean = acc.sum(row.iter().map(|v| v * v)) / row.len() as f32;
    1.0 / (mean + eps).sqrt()
}

pub(crate) fn gather(table: &Tensor, ids: &Tensor) -> Result<Tensor, Error> {
    let rows = table.shape[0];
    let width = table.len() / rows.max(1);
    let ids_at = ids.indices()?;
    if let Some(bad) = ids_at.iter().find(|&&i| i >= rows) {
        return Err(Error::Mismatch(format!("index {bad} past {rows} rows")));
    }
    let at = ids_at
        .iter()
        .flat_map(|&r| (0..width).map(move |c| r * width + c));
    let mut shape = ids.shape.clone();
    shape.extend_from_slice(&table.shape[1..]);
    Ok(table.pick(shape, at))
}
