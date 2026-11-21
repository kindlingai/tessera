//! Plain loops: the reference every plan is checked against.

use crate::round::Acc;
use crate::tensor::Error;
use tessera_graph::{Act, Attend, Delta, Mask, Numerics, Pairs, Stage};
use tessera_mem::Tensor;

/// The tanh approximation the GPU kernels use.
fn gelu(x: f32) -> f32 {
    const COEF: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (COEF * (x + 0.044_715 * x * x * x)).tanh())
}

/// Exact for large `x`, where `e^x` overflows.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::GeluTanh => gelu(x),
        Act::Silu => x / (1.0 + (-x).exp()),
        Act::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Act::Relu => x.max(0.0),
        Act::Softplus => softplus(x),
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

/// Leading axes pair off. A shorter `x` is a prefix.
pub(crate) fn matmul(n: &Numerics, x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
    let ws = w.floats()?;
    let size: usize = w.shape[w.shape.len() - 2..].iter().product();
    product(n, x, &w.shape, |b| Some(&ws[b * size..(b + 1) * size]))
}

/// `matmul(x, gather(table, ids))`, the gathered matrices never built.
pub(crate) fn matmul_picked(
    n: &Numerics,
    x: &Tensor,
    [table, ids]: [&Tensor; 2],
    first: usize,
) -> Result<Tensor, Error> {
    let (ts, rows) = (table.floats()?, table.shape[0]);
    let size = table.len() / rows.max(1);
    let mut shape = ids.shape.clone();
    shape.extend_from_slice(&table.shape[1..]);
    let ids = ids.indices()?;
    // A row outside this rank's table reads zeros.
    let row = |b: usize| ids[b].checked_sub(first).filter(|r| *r < rows);
    product(n, x, &shape, |b| {
        row(b).map(|r| &ts[r * size..(r + 1) * size])
    })
}

/// `x` times each `[N, K]` matrix of `shape`.
fn product<'w>(
    n: &Numerics,
    x: &Tensor,
    shape: &[usize],
    matrix: impl Fn(usize) -> Option<&'w [f32]>,
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let (m, k) = x.rows();
    let lead = shape.len() - 2;
    let n = shape[lead];
    let batches: usize = shape[..lead].iter().product();
    let xs = x.floats()?;
    let zeros = vec![0.0; n * k];
    let rows = m.max(batches);
    let mut out = vec![0.0; rows * n];
    for (r, row) in out.chunks_mut(n).enumerate() {
        let (xr, wb) = (r / (rows / m.max(1)), r / (rows / batches.max(1)));
        let xr = &xs[xr * k..(xr + 1) * k];
        let wb = matrix(wb).unwrap_or(&zeros);
        for (c, o) in row.iter_mut().enumerate() {
            *o = acc.dot(xr, &wb[c * k..(c + 1) * k]);
        }
    }
    let mut shape = shape[..lead].to_vec();
    if x.shape.len() > lead {
        shape = x.shape.clone();
        shape.pop();
    }
    shape.push(n);
    Ok(Tensor::f32(shape, out))
}

/// Keys and values every query row of one attention reads.
struct Keys<'a> {
    acc: Acc,
    scale: f32,
    k: &'a [f32],
    v: &'a [f32],
}

impl Keys<'_> {
    /// Adds query `q` over the key rows `at` into `o`.
    fn row(
        &self,
        q: &[f32],
        at: impl Iterator<Item = usize> + Clone,
        p: &mut Vec<f32>,
        o: &mut [f32],
    ) {
        let (d, dv) = (q.len(), o.len());
        p.clear();
        for j in at.clone() {
            p.push(self.acc.dot(q, &self.k[j * d..(j + 1) * d]) * self.scale);
        }
        softmax(self.acc, p);
        for (j, &pj) in at.zip(p.iter()) {
            for (oi, &vi) in o.iter_mut().zip(&self.v[j * dv..(j + 1) * dv]) {
                *oi = self.acc.add(*oi, pj * vi);
            }
        }
    }
}

/// With `picked`, each query reads its listed keys in order.
pub(crate) fn attend(
    n: &Numerics,
    a: &Attend,
    qkv: [&Tensor; 3],
    picked: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let [q, k, v] = qkv;
    let [seqs, ctx, heads, d] = q.shape[..] else {
        return Err(Error::Mismatch("attention wants rank 4".into()));
    };
    let (keys, kv_heads, dv) = (k.shape[1], k.shape[2], v.shape[3]);
    let group = heads / kv_heads;
    let past = past_keys(a, ctx, keys)?;
    let qs = q.floats()?;
    let (k, v) = (k.floats()?, v.floats()?);
    let kv = Keys {
        acc,
        scale: a.scale,
        k,
        v,
    };
    let mut out = vec![0.0; seqs * ctx * heads * dv];
    let lists = picked.map(Tensor::indices).transpose()?;
    let mut p = Vec::with_capacity(keys);
    for s in 0..seqs {
        for t in 0..ctx {
            let (from, to) = seen(a.mask, past + t, keys);
            let all: Vec<usize> = match &lists {
                Some(ids) => {
                    let width = ids.len() / (seqs * ctx);
                    let list = &ids[(s * ctx + t) * width..][..width];
                    list.iter()
                        .copied()
                        .filter(|j| (from..to).contains(j))
                        .collect()
                }
                None => (from..to).collect(),
            };
            for h in 0..heads {
                let row = (s * ctx + t) * heads + h;
                let kv_at = |j: &usize| (s * keys + j) * kv_heads + h / group;
                let o = &mut out[row * dv..(row + 1) * dv];
                kv.row(
                    &qs[row * d..(row + 1) * d],
                    all.iter().map(kv_at),
                    &mut p,
                    o,
                );
            }
        }
    }
    Ok(Tensor::f32(vec![seqs, ctx, heads, dv], out))
}

/// Rows before a sequence's first read as zero.
pub(crate) fn conv(x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
    let [seqs, ctx, width] = x.shape[..] else {
        return Err(Error::Mismatch("conv wants rank 3".into()));
    };
    let taps = w.shape[1];
    let (xs, ws) = (x.floats()?, w.floats()?);
    let mut out = vec![0.0; xs.len()];
    for s in 0..seqs {
        for t in 0..ctx {
            let at = (s * ctx + t) * width;
            for c in 0..width {
                let mut sum = 0.0;
                for j in (taps - 1).saturating_sub(t)..taps {
                    let back = taps - 1 - j;
                    sum += ws[c * taps + j] * xs[at - back * width + c];
                }
                out[at + c] = sum;
            }
        }
    }
    Ok(Tensor::f32(x.shape.clone(), out))
}

/// Holds each value head's state as key columns.
pub(crate) fn delta(n: &Numerics, d: &Delta, args: [&Tensor; 5]) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let [q, k, v, g, beta] = args;
    let [seqs, ctx, kv_heads, dk] = q.shape[..] else {
        return Err(Error::Mismatch("the delta rule wants rank 4".into()));
    };
    let (heads, dv) = (v.shape[2], v.shape[3]);
    let group = heads / kv_heads;
    let (qs, ks, vs) = (q.floats()?, k.floats()?, v.floats()?);
    let (gs, rates) = (g.floats()?, beta.floats()?);
    let mut out = vec![0.0; seqs * ctx * heads * dv];
    let mut state = vec![0.0; dv * dk];
    let mut query = vec![0.0; dk];
    for s in 0..seqs {
        for h in 0..heads {
            state.fill(0.0);
            for t in 0..ctx {
                let row = (s * ctx + t) * heads + h;
                let at = ((s * ctx + t) * kv_heads + h / group) * dk;
                let key = &ks[at..at + dk];
                for (qi, &x) in query.iter_mut().zip(&qs[at..at + dk]) {
                    *qi = x * d.scale;
                }
                let decay = gs[row].exp();
                let values = &vs[row * dv..(row + 1) * dv];
                let o = &mut out[row * dv..(row + 1) * dv];
                for ((col, &value), oj) in state.chunks_mut(dk).zip(values).zip(o) {
                    col.iter_mut().for_each(|x| *x *= decay);
                    let change = (value - acc.dot(col, key)) * rates[row];
                    for (x, &ki) in col.iter_mut().zip(key) {
                        *x += ki * change;
                    }
                    *oj = acc.dot(col, &query);
                }
            }
        }
    }
    Ok(Tensor::f32(vec![seqs, ctx, heads, dv], out))
}

/// Rows read their slot to their position, or listed keys.
pub(crate) fn attend_rows(
    n: &Numerics,
    a: &Attend,
    [q, k, v]: [&Tensor; 3],
    [slots, positions]: [&Tensor; 2],
    picked: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let [rows, heads, d] = q.shape[..] else {
        return Err(Error::Mismatch(
            "ragged attention wants rank 3 queries".into(),
        ));
    };
    let (keys, kv_heads, dv) = (k.shape[1], k.shape[2], v.shape[3]);
    let group = heads / kv_heads;
    let qs = q.floats()?;
    let (k, v) = (k.floats()?, v.floats()?);
    let kv = Keys {
        acc,
        scale: a.scale,
        k,
        v,
    };
    let (slots, positions) = (slots.indices()?, positions.indices()?);
    let lists = picked.map(Tensor::indices).transpose()?;
    let mut out = vec![0.0; rows * heads * dv];
    let mut p = Vec::with_capacity(keys);
    for r in 0..rows {
        let (s, seen) = (slots[r], positions[r] + 1);
        if seen > keys {
            let msg = format!("row {r} at position {} is past {keys} keys", seen - 1);
            return Err(Error::Mismatch(msg));
        }
        let all: Vec<usize> = match &lists {
            Some(ids) => {
                let width = ids.len() / rows.max(1);
                let list = &ids[r * width..][..width];
                list.iter().copied().filter(|j| *j < seen).collect()
            }
            None => (0..seen).collect(),
        };
        for h in 0..heads {
            let row = r * heads + h;
            let kv_at = |j: &usize| (s * keys + j) * kv_heads + h / group;
            let o = &mut out[row * dv..(row + 1) * dv];
            kv.row(
                &qs[row * d..(row + 1) * d],
                all.iter().map(kv_at),
                &mut p,
                o,
            );
        }
    }
    Ok(Tensor::f32(vec![rows, heads, dv], out))
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
        (Stage::Sin, _) => out.iter_mut().for_each(|v| *v = v.sin()),
        (Stage::Cos, _) => out.iter_mut().for_each(|v| *v = v.cos()),
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
        (Stage::Causal, Some(at)) => {
            let at = at.indices()?;
            let span = out.len() / at.len().max(1);
            for (block, &pos) in out.chunks_mut(span.max(1)).zip(&at) {
                for row in block.chunks_mut(width) {
                    row.iter_mut()
                        .skip(pos + 1)
                        .for_each(|v| *v = f32::NEG_INFINITY);
                }
            }
        }
        (Stage::Allow, Some(bits)) => {
            let bits = bits.indices()?;
            let words = width.div_ceil(32);
            for (row, mask) in out.chunks_mut(width).zip(bits.chunks(words)) {
                for (i, v) in row.iter_mut().enumerate() {
                    if mask[i / 32] >> (i % 32) & 1 == 0 {
                        *v = f32::NEG_INFINITY;
                    }
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

/// The mean over `axis`, or with `weights` a weighted sum.
pub(crate) fn mean(
    n: &Numerics,
    x: &Tensor,
    axis: usize,
    weights: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let acc = Acc::new(n.accumulate)?;
    let count = x.shape[axis];
    let inner: usize = x.shape[axis + 1..].iter().product();
    let outer: usize = x.shape[..axis].iter().product();
    let xs = x.floats()?;
    let ws = weights.map(Tensor::floats).transpose()?;
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let weight = |t: usize| ws.map_or(1.0 / count as f32, |w| w[o * count + t]);
            let along = (0..count).map(|t| xs[(o * count + t) * inner + i] * weight(t));
            out.push(acc.sum(along));
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

/// Each row's entries at `ids`. Out-of-range ids read zero.
pub(crate) fn take(x: &Tensor, ids: &Tensor) -> Result<Tensor, Error> {
    let (rows, width) = x.rows();
    let (xs, at) = (x.floats()?, ids.indices()?);
    let picks = at.len() / rows.max(1);
    let mut out = Vec::with_capacity(at.len());
    for (j, &i) in at.iter().enumerate() {
        let row = &xs[(j / picks.max(1)) * width..][..width];
        out.push(row.get(i).copied().unwrap_or(0.0));
    }
    Ok(Tensor::f32(ids.shape.clone(), out))
}

/// Rows at `ids` from `first` on. Others read zero.
pub(crate) fn gather(table: &Tensor, ids: &Tensor, first: usize) -> Result<Tensor, Error> {
    let rows = table.shape[0];
    let width = table.len() / rows.max(1);
    let (all, size) = (table.bytes(), table.dtype().bytes() * width);
    let mut out = Vec::new();
    for r in ids.indices()? {
        match r.checked_sub(first).filter(|r| *r < rows) {
            Some(r) => out.extend_from_slice(&all[r * size..(r + 1) * size]),
            None => out.resize(out.len() + size, 0),
        }
    }
    let mut shape = ids.shape.clone();
    shape.extend_from_slice(&table.shape[1..]);
    Ok(Tensor::from_bytes(shape, table.dtype(), &out))
}
