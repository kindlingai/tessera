//! Single-layer Hessian and routing parity verifier against reference files.

use std::fs;
use std::path::Path;
use tessera_checkpoint::{File, Offer};

/// Relative Frobenius and diagonal error for one Hessian tensor.
#[derive(Clone, Debug)]
pub struct Metric {
    pub file: String,
    pub tensor: String,
    pub rel_frob: f64,
    pub max_diag: f64,
}

/// Held-out router and activation parity summary (`inputs-LLL.safetensors`).
#[derive(Clone, Debug)]
pub struct InputsMetric {
    pub idx_match: f64,
    pub max_w_diff: f64,
    pub x_cos: f64,
}

/// Least share of matching picks, and least input cosine.
const MIN_IDX_MATCH: f64 = 0.99;
const MIN_X_COS: f64 = 0.999;
/// Largest gate difference on matching picks.
const MAX_W_DIFF: f64 = 2e-2;

impl InputsMetric {
    fn passes(&self) -> bool {
        self.idx_match >= MIN_IDX_MATCH && self.x_cos >= MIN_X_COS && self.max_w_diff <= MAX_W_DIFF
    }
}

/// Aggregated parity check result for one layer directory pair.
#[derive(Clone, Debug)]
pub struct ParityReport {
    pub matrices: Vec<Metric>,
    pub inputs: Option<InputsMetric>,
    pub worst_rel: f64,
    pub passed: bool,
}

fn triu_dim(len: usize) -> Result<usize, String> {
    let d = (((8 * len + 1) as f64).sqrt() as usize - 1) / 2;
    if d * (d + 1) / 2 != len {
        return Err(format!("invalid packed upper-triangle length {len}"));
    }
    Ok(d)
}

/// Relative Frobenius error and worst relative diagonal error.
fn rel_err(cells: impl Iterator<Item = (bool, f64, f32, f32)>) -> (f64, f64) {
    let (mut num, mut den, mut max_d) = (0.0f64, 0.0f64, 0.0f64);
    for (diag, wt, g, w) in cells {
        let (g, w) = (f64::from(g), f64::from(w));
        let diff = g - w;
        num += wt * diff * diff;
        den += wt * w * w;
        if diag {
            max_d = max_d.max(diff.abs() / w.abs().max(1e-12));
        }
    }
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    (rel, max_d)
}

/// Errors of `rows` packed upper triangles, off-diagonal cells counted twice.
fn triu_err(got: &[f32], want: &[f32], rows: usize) -> Result<(f64, f64), String> {
    if got.len() != want.len() || rows == 0 {
        return Err("packed upper-triangle length mismatch".into());
    }
    let d = triu_dim(got.len() / rows)?;
    let cells = (0..rows).flat_map(|_| (0..d).flat_map(move |r| (r..d).map(move |c| r == c)));
    let vals = got.iter().zip(want);
    Ok(rel_err(cells.zip(vals).map(|(diag, (&g, &w))| {
        (diag, if diag { 1.0 } else { 2.0 }, g, w)
    })))
}

/// Errors of one full `d` by `d` matrix.
fn dense_err(got: &[f32], want: &[f32], d: usize) -> Result<(f64, f64), String> {
    if got.len() != want.len() || got.len() != d * d {
        return Err("dense matrix length mismatch".into());
    }
    let cells = got.iter().zip(want).enumerate();
    Ok(rel_err(
        cells.map(|(i, (&g, &w))| (i / d == i % d, 1.0, g, w)),
    ))
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

fn compare_inputs(gf: &File, wf: &File) -> Result<InputsMetric, String> {
    let g_idx = decode_i32(&gf.bytes("topk_idx").map_err(|e| e.to_string())?);
    let w_idx = decode_i32(&wf.bytes("topk_idx").map_err(|e| e.to_string())?);
    if g_idx.len() != w_idx.len() || g_idx.is_empty() {
        return Err("inputs topk_idx length mismatch".into());
    }
    let k = wf
        .slices()
        .into_iter()
        .find(|s| s.name == "topk_idx")
        .and_then(|s| s.shape.get(1).copied())
        .unwrap_or(1)
        .max(1);

    let gw = gf.floats("topk_w").map_err(|e| e.to_string())?;
    let ww = wf.floats("topk_w").map_err(|e| e.to_string())?;
    if gw.len() != ww.len() || gw.len() != g_idx.len() {
        return Err("inputs topk_w length mismatch".into());
    }

    // Match picks as sets, since their order may differ.
    let mut hits = 0usize;
    let mut max_w_diff = 0.0f64;
    for r in 0..(g_idx.len() / k) {
        let base = r * k;
        for j in 0..k {
            let id = g_idx[base + j];
            if let Some(pos) = w_idx[base..base + k].iter().position(|&x| x == id) {
                hits += 1;
                let diff = (gw[base + j] as f64 - ww[base + pos] as f64).abs();
                max_w_diff = max_w_diff.max(diff);
            }
        }
    }
    let idx_match = hits as f64 / g_idx.len() as f64;

    let gx = gf.floats("x").map_err(|e| e.to_string())?;
    let wx = wf.floats("x").map_err(|e| e.to_string())?;
    if gx.len() != wx.len() {
        return Err("inputs x length mismatch".into());
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in gx.iter().zip(&wx) {
        let (af, bf) = (a as f64, b as f64);
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
    }
    let x_cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    Ok(InputsMetric {
        idx_match,
        max_w_diff,
        x_cos,
    })
}

/// Compares one layer's `HESSIAN.md` files against a reference directory.
pub fn compare_dirs(
    got_dir: &Path,
    want_dir: &Path,
    layer: usize,
    max_rel: f64,
) -> Result<ParityReport, String> {
    let hess_prefix = format!("hess-{layer:03}-");
    let inputs_name = format!("inputs-{layer:03}.safetensors");
    let mut names: Vec<String> = Vec::new();
    for entry in fs::read_dir(want_dir).map_err(|e| format!("{}: {e}", want_dir.display()))? {
        let name = entry
            .map_err(|e| e.to_string())?
            .file_name()
            .to_string_lossy()
            .into_owned();
        if (name.starts_with(&hess_prefix) || name == inputs_name) && name.ends_with(".safetensors")
        {
            names.push(name);
        }
    }
    names.sort();
    if names.is_empty() {
        return Err(format!("no layer {layer} files in {}", want_dir.display()));
    }

    let mut matrices = Vec::new();
    let mut inputs = None;
    let mut worst_rel = 0.0f64;

    for name in &names {
        let gp = got_dir.join(name);
        let wp = want_dir.join(name);
        if !gp.exists() {
            return Err(format!("missing output file {}", gp.display()));
        }
        let gf = File::open(&gp).map_err(|e| e.to_string())?;
        let wf = File::open(&wp).map_err(|e| e.to_string())?;
        if name == &inputs_name {
            inputs = Some(compare_inputs(&gf, &wf)?);
            continue;
        }
        let is_expert = name.contains("-e");
        for sl in wf.slices() {
            if sl.name == "count" || sl.name.starts_with("count_") {
                let gc = gf.i64s(&sl.name).map_err(|e| e.to_string())?;
                let wc = wf.i64s(&sl.name).map_err(|e| e.to_string())?;
                if gc != wc {
                    return Err(format!("{name}:{}: count mismatch", sl.name));
                }
                continue;
            }
            let gv = gf.floats(&sl.name).map_err(|e| e.to_string())?;
            let wv = wf.floats(&sl.name).map_err(|e| e.to_string())?;
            let (rel_frob, max_diag) = if is_expert {
                triu_err(&gv, &wv, sl.shape[0])?
            } else {
                dense_err(&gv, &wv, sl.shape[0])?
            };
            worst_rel = worst_rel.max(rel_frob);
            matrices.push(Metric {
                file: name.clone(),
                tensor: sl.name,
                rel_frob,
                max_diag,
            });
        }
    }
    let finite = (matrices.iter())
        .flat_map(|m| [m.rel_frob, m.max_diag])
        .chain(
            inputs
                .iter()
                .flat_map(|i| [i.idx_match, i.max_w_diff, i.x_cos]),
        )
        .all(f64::is_finite);
    let inputs_ok = inputs.as_ref().is_none_or(InputsMetric::passes);
    Ok(ParityReport {
        passed: finite && inputs_ok && worst_rel <= max_rel,
        matrices,
        inputs,
        worst_rel,
    })
}

/// Prints a compact parity table for CLI verification runs.
pub fn print_report(rep: &ParityReport, max_rel: f64) {
    for m in &rep.matrices {
        println!(
            "  {:36} {:16} rel_frob={:.3e} max_diag={:.3e}",
            m.file, m.tensor, m.rel_frob, m.max_diag
        );
    }
    if let Some(inp) = &rep.inputs {
        println!(
            "  inputs: idx_match={:.2}% max_w_diff={:.3e} x_cos={:.6}",
            inp.idx_match * 100.0,
            inp.max_w_diff,
            inp.x_cos
        );
    }
    let tag = if rep.passed { "PASS" } else { "FAIL" };
    println!(
        "parity {tag}: worst_rel_frob={:.3e} (limit {:.3e})",
        rep.worst_rel, max_rel
    );
}
