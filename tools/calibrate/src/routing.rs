//! Realistic MoE routing feeds and kept-cache prefill helpers.

use std::fmt;
use tessera_engine::Error;
use tessera_graph::DimName::{self, Context, Experts, Hidden, Rows, TopK};
use tessera_graph::{DType, Graph, Op, Role, Select, Size};
use tessera_lower::{Device, Tensor, View, Weight, lower};
use tessera_run::Run;
use tessera_testkit::{Noise, fill};

/// Default Zipf exponent for realistic MoE expert skew.
pub(crate) const DEFAULT_SKEW: f64 = 1.0;

/// Summary of expert load across one routed batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RouteStats {
    pub active: usize,
    pub total: usize,
    pub min: usize,
    pub median: usize,
    pub max: usize,
    pub mean: f64,
}

impl fmt::Display for RouteStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ratio = if self.mean > 0.0 {
            self.max as f64 / self.mean
        } else {
            0.0
        };
        write!(
            f,
            "{}/{} active experts, tokens/expert min={} med={} max={} ({ratio:.1}x mean)",
            self.active, self.total, self.min, self.median, self.max
        )
    }
}

fn rand01(noise: &mut Noise) -> f64 {
    ((noise.draw() as f64 + 1.0) * 0.5).clamp(0.0, 0.999_999_9)
}

/// Draws `top_k` distinct expert ids under a Zipf law.
pub(crate) fn zipf_ids(
    rows: usize,
    experts: usize,
    top_k: usize,
    skew: f64,
    noise: &mut Noise,
) -> Vec<u32> {
    let k = top_k.min(experts);
    let mut order: Vec<u32> = (0..experts as u32).collect();
    for i in (1..experts).rev() {
        let j = (rand01(noise) * (i + 1) as f64) as usize;
        order.swap(i, j);
    }
    let mut cdf = Vec::with_capacity(experts);
    let mut sum = 0.0f64;
    for r in 0..experts {
        sum += ((r + 1) as f64).powf(-skew.max(0.0));
        cdf.push(sum);
    }
    let mut out = Vec::with_capacity(rows * k);
    let mut picked = Vec::with_capacity(k);
    for _ in 0..rows {
        picked.clear();
        while picked.len() < k {
            let mut tries = 0;
            let mut chosen = None;
            while tries < 8 {
                tries += 1;
                let target = rand01(noise) * sum;
                let rank = cdf.partition_point(|&c| c < target).min(experts - 1);
                let id = order[rank];
                if !picked.contains(&id) {
                    chosen = Some(id);
                    break;
                }
            }
            let id = match chosen {
                Some(id) => id,
                None => order
                    .iter()
                    .copied()
                    .find(|id| !picked.contains(id))
                    .unwrap_or(0),
            };
            picked.push(id);
        }
        out.extend_from_slice(&picked);
    }
    out
}

/// Generates positive per-row routing gates summing to 1.
pub(crate) fn zipf_gates(rows: usize, top_k: usize, noise: &mut Noise) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * top_k);
    for _ in 0..rows {
        let start = out.len();
        let mut sum = 0.0f32;
        for j in 0..top_k {
            let w = (-(j as f32) * 0.35 + noise.draw() * 0.15).exp();
            out.push(w);
            sum += w;
        }
        for v in &mut out[start..] {
            *v /= sum.max(1e-6);
        }
    }
    out
}

/// Computes active expert counts and load percentiles.
pub(crate) fn summarize_ids(ids: &[u32], experts: usize) -> RouteStats {
    let mut counts = vec![0usize; experts];
    for &id in ids {
        if let Some(slot) = counts.get_mut(id as usize) {
            *slot += 1;
        }
    }
    let mut active: Vec<usize> = counts.into_iter().filter(|&c| c > 0).collect();
    active.sort_unstable();
    let total_picks = ids.len() as f64;
    let mean = if experts > 0 {
        total_picks / experts as f64
    } else {
        0.0
    };
    let min = active.first().copied().unwrap_or(0);
    let median = active.get(active.len() / 2).copied().unwrap_or(0);
    let max = active.last().copied().unwrap_or(0);
    RouteStats {
        active: active.len(),
        total: experts,
        min,
        median,
        max,
        mean,
    }
}

/// Evaluates `name` from `sizes`, falling back to `g`'s fixed/bounded dim.
pub(crate) fn dim_value(g: &Graph, sizes: &[(DimName, u64)], name: DimName) -> Option<usize> {
    if let Some((_, v)) = sizes.iter().find(|(d, _)| *d == name) {
        return Some(*v as usize);
    }
    g.dims
        .iter()
        .find(|d| d.name == name)
        .map(|d| match d.size {
            Size::Fixed(v) | Size::Bounded(v) => v as usize,
        })
}

/// Extracts router rows `[experts, hidden]` when a weight feeds `TopK`.
pub(crate) fn extract_router(g: &Graph, loads: &[Weight]) -> Option<Vec<f32>> {
    let has_topk = g
        .nodes
        .iter()
        .any(|n| matches!(n.op, Op::Select(Select::TopK { .. })));
    if !has_topk {
        return None;
    }
    let experts = dim_value(g, &[], Experts)?;
    let hidden = dim_value(g, &[], Hidden)?;
    for (idx, w) in g.weights.iter().enumerate() {
        if w.ty.shape.0.len() != 2 {
            continue;
        }
        let shape = [
            w.ty.shape.0[0].poly(&g.dims).ok()?.eval(&|_| None)? as usize,
            w.ty.shape.0[1].poly(&g.dims).ok()?.eval(&|_| None)? as usize,
        ];
        if shape != [experts, hidden] {
            continue;
        }
        let Some(weight) = loads.get(idx) else {
            continue;
        };
        let vals = match weight {
            Weight::Floats(t) => t.floats().ok()?.to_vec(),
            Weight::Packed(form, bytes) => form.unpack_all(bytes),
            _ => continue,
        };
        if vals.len() == experts * hidden {
            return Some(vals);
        }
    }
    None
}

/// Builds unit-RMS hidden rows steered toward `ids` through `router`.
fn steered_hidden(
    rows: usize,
    hidden: usize,
    top_k: usize,
    ids: &[u32],
    router: &[f32],
    noise: &mut Noise,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let dst = &mut out[r * hidden..(r + 1) * hidden];
        for j in 0..top_k {
            let e = ids[r * top_k + j] as usize;
            let row = &router[e * hidden..(e + 1) * hidden];
            let gain = 1.0 / (j as f32 + 1.0);
            for (d, &w) in dst.iter_mut().zip(row) {
                *d += w * gain;
            }
        }
        let mut sq = 0.0f32;
        for d in dst.iter_mut() {
            *d += noise.draw() * 0.02;
            sq += *d * *d;
        }
        let inv = (hidden as f32 / sq.max(1e-6)).sqrt();
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }
    out
}

/// Builds clustered hidden rows without an unpacked router weight.
fn clustered_hidden(
    rows: usize,
    hidden: usize,
    experts: usize,
    top_k: usize,
    ids: &[u32],
    noise: &mut Noise,
) -> Vec<f32> {
    let mut centers = Vec::with_capacity(experts * hidden);
    for _ in 0..experts * hidden {
        centers.push(noise.draw());
    }
    steered_hidden(rows, hidden, top_k, ids, &centers, noise)
}

/// Builds routed feeds for `g` at `sizes` and `pos_offset`.
pub(crate) fn routed_feeds(
    g: &Graph,
    sizes: &[(DimName, u64)],
    router: Option<&[f32]>,
    skew: f64,
    pos_offset: u32,
    noise: &mut Noise,
) -> Result<(Vec<Tensor>, Option<RouteStats>), Error> {
    let experts = dim_value(g, sizes, Experts);
    let top_k = dim_value(g, sizes, TopK);
    let rows = dim_value(g, sizes, Rows).unwrap_or(1);
    let hidden = dim_value(g, sizes, Hidden);
    let has_picks = g.entries[0].inputs.iter().any(|p| p.role == Role::Picks);
    let moe_ids = match (experts, top_k) {
        (Some(e), Some(k)) if e > 0 && k > 0 => Some((e, k, zipf_ids(rows, e, k, skew, noise))),
        _ => None,
    };
    let stats = moe_ids.as_ref().map(|(e, _, ids)| summarize_ids(ids, *e));
    let mut out = Vec::with_capacity(g.entries[0].inputs.len());
    for port in &g.entries[0].inputs {
        let unsized_shape = || {
            Error::from(tessera_lower::Error::from(tessera_lower::Size::Shape(
                port.ty.shape.clone(),
            )))
        };
        let t = fill(g, &port.ty, sizes, noise).ok_or_else(unsized_shape)?;
        let tensor = match port.role {
            Role::Slots => Tensor::zeros(t.shape.clone(), t.dtype()),
            Role::Positions => {
                let (n, last) = (t.len(), *t.shape.last().unwrap_or(&1));
                let vals = (0..n).map(|i| pos_offset + (i % last) as u32).collect();
                Tensor::u32(t.shape.clone(), vals)
            }
            Role::Picks => match &moe_ids {
                Some((_, _, ids)) if ids.len() == t.len() => {
                    Tensor::u32(t.shape.clone(), ids.clone())
                }
                _ => t,
            },
            _ if has_picks
                && port.ty.dtype == DType::F32
                && top_k.is_some_and(|k| t.shape == [rows, k]) =>
            {
                let k = top_k.unwrap();
                Tensor::f32(t.shape.clone(), zipf_gates(rows, k, noise))
            }
            _ if !has_picks
                && port.ty.dtype == DType::F32
                && hidden.is_some_and(|h| t.shape == [rows, h]) =>
            {
                match &moe_ids {
                    Some((e, k, ids)) => {
                        let h = hidden.unwrap();
                        let vals = match router.filter(|mat| mat.len() == *e * h) {
                            Some(mat) => steered_hidden(rows, h, *k, ids, mat, noise),
                            None => clustered_hidden(rows, h, *e, *k, ids, noise),
                        };
                        Tensor::f32(t.shape.clone(), vals)
                    }
                    None => t,
                }
            }
            _ => t,
        };
        out.push(tensor);
    }
    Ok((out, stats))
}

/// Prefills kept KV caches for `0..(context - rows)`.
pub(crate) fn prefill_kept(
    runner: &mut dyn Run,
    g: &Graph,
    sizes: &[(DimName, u64)],
    device: &Device,
    noise: &mut Noise,
) -> Result<u32, Error> {
    if g.kept.is_empty() {
        return Ok(0);
    }
    let Some(context) = dim_value(g, sizes, Context) else {
        return Ok(0);
    };
    let rows = dim_value(g, sizes, Rows).unwrap_or(1);
    if context <= rows {
        return Ok(0);
    }
    let fill_total = context - rows;
    let max_chunk = g
        .dims
        .iter()
        .find(|d| d.name == Rows)
        .map_or(fill_total, |d| match d.size {
            Size::Fixed(v) | Size::Bounded(v) => v as usize,
        })
        .clamp(1, 512);
    let mut done = 0usize;
    while done < fill_total {
        let chunk = (fill_total - done).min(max_chunk);
        let mut chunk_sizes: Vec<(DimName, u64)> =
            sizes.iter().filter(|(d, _)| *d != Rows).copied().collect();
        chunk_sizes.push((Rows, chunk as u64));
        let plan = lower(g, 0, &chunk_sizes, device)?;
        runner.replan(&plan)?;
        let (fed, _) = routed_feeds(g, &chunk_sizes, None, 0.0, done as u32, noise)?;
        let views: Vec<View> = fed.iter().map(Tensor::view).collect();
        runner.feed(&views)?;
        runner.submit(0..plan.steps.len())?;
        done += chunk;
    }
    Ok(fill_total as u32)
}
