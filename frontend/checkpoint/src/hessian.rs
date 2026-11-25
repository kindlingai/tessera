//! exlr8 `HESSIAN.md` calibration file writers and `weights_id`.

use crate::Error;
use crate::st_write::{StTensor, le_bytes, write_file};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const EXPERT_LAYOUT: &str = "upper triangle, row-major, fp32";
const DENSE_LAYOUT: &str = "full, fp32";

/// Common metadata carried by every `HESSIAN.md` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HessianMeta {
    pub hessian_id: String,
    pub weights_id: String,
}

impl HessianMeta {
    pub fn validate(&self) -> Result<(), Error> {
        if self.hessian_id.trim().is_empty() {
            return Err(Error::Format("hessian_id must be non-empty".into()));
        }
        if !valid_hex(&self.weights_id) {
            return Err(Error::Format(format!(
                "weights_id must be 64 lowercase hex chars, got `{}`",
                self.weights_id
            )));
        }
        Ok(())
    }

    fn base(&self, layer: usize) -> Result<BTreeMap<String, String>, Error> {
        self.validate()?;
        let mut meta = BTreeMap::new();
        meta.insert("hessian_id".into(), self.hessian_id.clone());
        meta.insert("weights_id".into(), self.weights_id.clone());
        meta.insert("layer".into(), layer.to_string());
        Ok(meta)
    }
}

/// One dense square Hessian matrix and its token count.
pub struct DenseEntry<'a> {
    pub name: &'a str,
    pub dim: usize,
    pub data: &'a [f32],
    pub count: i64,
}

/// Checks 64 lowercase hex characters.
pub fn valid_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Packed upper-triangle element count `d * (d + 1) / 2`.
pub fn triu_len(d: usize) -> usize {
    d * (d + 1) / 2
}

/// Flat row-major upper-triangle index for `0 <= r <= c < d`.
pub(crate) fn triu_index(d: usize, r: usize, c: usize) -> usize {
    debug_assert!(r <= c && c < d);
    r * (2 * d + 1 - r) / 2 + (c - r)
}

pub(crate) fn pack_tiles_into<T: Copy>(tiles: &[T], d: usize, tile: usize, out: &mut [T]) {
    assert!(tile > 0 && d.is_multiple_of(tile));
    let blocks = d / tile;
    let total_tiles = blocks * (blocks + 1) / 2;
    assert_eq!(tiles.len(), total_tiles * tile * tile);
    assert_eq!(out.len(), triu_len(d));
    let mut tidx = 0usize;
    for br in 0..blocks {
        for bc in br..blocks {
            let base = tidx * tile * tile;
            for lr in 0..tile {
                let r = br * tile + lr;
                let lc_start = if br == bc { lr } else { 0 };
                let dst = triu_index(d, r, bc * tile + lc_start);
                let src = base + lr * tile + lc_start;
                let len = tile - lc_start;
                out[dst..dst + len].copy_from_slice(&tiles[src..src + len]);
            }
            tidx += 1;
        }
    }
}

/// Packs little-endian FP32 upper-triangle tiles into row-major upper-triangle bytes.
pub fn pack_tiles_bytes(tiles: &[u8], d: usize, tile: usize, out: &mut [u8]) {
    let (src, rem_s) = tiles.as_chunks::<4>();
    let (dst, rem_d) = out.as_chunks_mut::<4>();
    assert!(rem_s.is_empty() && rem_d.is_empty());
    pack_tiles_into(src, d, tile, dst);
}

/// Filename for routed expert shard `a..=b` at `layer`.
pub fn expert_name(layer: usize, a: usize, b: usize) -> String {
    format!("hess-{layer:03}-e{a:03}-{b:03}.safetensors")
}

/// Filename for dense or dense-heldout Hessians at `layer`.
pub fn dense_name(layer: usize, heldout: bool) -> String {
    if heldout {
        format!("hess-{layer:03}-dense-heldout.safetensors")
    } else {
        format!("hess-{layer:03}-dense.safetensors")
    }
}

/// Filename for held-out inputs at `layer`.
pub fn inputs_name(layer: usize) -> String {
    format!("inputs-{layer:03}.safetensors")
}

/// Filename for optional routing log at `layer`.
pub fn route_name(layer: usize) -> String {
    format!("route-{layer:03}.safetensors")
}

/// One file of routed expert Hessians as little-endian FP32 bytes.
pub struct ExpertShard<'a> {
    pub layer: usize,
    pub first: usize,
    /// Gate and up input triangles, one per expert.
    pub gu: &'a [u8],
    /// Packed upper triangles of the down input, one per expert.
    pub down: &'a [u8],
    pub counts: &'a [i64],
    /// Hidden and expert inner widths.
    pub dims: (usize, usize),
}

/// Writes `hess-LLL-eAAA-BBB.safetensors` matching `HESSIAN.md`.
pub fn write_experts(
    path: &Path,
    shard: &ExpertShard<'_>,
    meta: &HessianMeta,
) -> Result<(), Error> {
    let n = shard.counts.len();
    if n == 0 {
        return Err(Error::Format(
            "expert shard requires at least 1 expert".into(),
        ));
    }
    let (h, i) = shard.dims;
    let (th, ti) = (triu_len(h), triu_len(i));
    if h == 0 || i == 0 || shard.gu.len() != n * th * 4 || shard.down.len() != n * ti * 4 {
        return Err(Error::Format("expert tensor shape mismatch".into()));
    }
    let mut m = meta.base(shard.layer)?;
    let last = shard.first + n - 1;
    m.insert("experts".into(), format!("{}-{last}", shard.first));
    m.insert("layout".into(), EXPERT_LAYOUT.into());
    let ct_b = le_bytes(shard.counts, i64::to_le_bytes);
    let tensors = [
        StTensor {
            name: "gu",
            dtype: "F32",
            shape: vec![n, th],
            data: shard.gu,
        },
        StTensor {
            name: "down",
            dtype: "F32",
            shape: vec![n, ti],
            data: shard.down,
        },
        StTensor {
            name: "count",
            dtype: "I64",
            shape: vec![n],
            data: &ct_b,
        },
    ];
    write_file(path, &m, &tensors)
}

/// Writes `hess-LLL-dense.safetensors` or `-dense-heldout.safetensors`.
pub fn write_dense(
    path: &Path,
    layer: usize,
    entries: &[DenseEntry<'_>],
    heldout: bool,
    meta: &HessianMeta,
) -> Result<(), Error> {
    let mut m = meta.base(layer)?;
    m.insert("layout".into(), DENSE_LAYOUT.into());
    let seqs = if heldout { "held out" } else { "calibration" };
    m.insert("sequences".into(), seqs.into());
    let mut owned: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(entries.len() * 2);
    for e in entries {
        if e.dim == 0 || e.data.len() != e.dim * e.dim {
            return Err(Error::Format(format!(
                "bad shape for dense tensor {}",
                e.name
            )));
        }
        owned.push((
            e.name.into(),
            "F32",
            vec![e.dim, e.dim],
            le_bytes(e.data, f32::to_le_bytes),
        ));
        owned.push((
            format!("count_{}", e.name),
            "I64",
            vec![1],
            le_bytes(&[e.count], i64::to_le_bytes),
        ));
    }
    let tensors: Vec<StTensor<'_>> = owned
        .iter()
        .map(|(name, dtype, shape, data)| StTensor {
            name,
            dtype,
            shape: shape.clone(),
            data,
        })
        .collect();
    write_file(path, &m, &tensors)
}

/// Held-out MoE inputs and their routing for one layer.
pub struct Inputs<'a> {
    pub layer: usize,
    pub heldout_seqs: usize,
    pub x_bf16: &'a [u16],
    pub topk_idx: &'a [i32],
    pub topk_w: &'a [f32],
    pub hidden: usize,
    pub top_k: usize,
}

/// Writes `inputs-LLL.safetensors` matching `HESSIAN.md`.
pub fn write_inputs(path: &Path, inp: &Inputs<'_>, meta: &HessianMeta) -> Result<(), Error> {
    let (hidden, top_k) = (inp.hidden, inp.top_k);
    if inp.heldout_seqs == 0 || hidden == 0 || top_k == 0 {
        return Err(Error::Format("inputs require positive dimensions".into()));
    }
    if !inp.x_bf16.len().is_multiple_of(hidden) {
        return Err(Error::Format("x length not divisible by hidden".into()));
    }
    let n_tok = inp.x_bf16.len() / hidden;
    if n_tok == 0 || inp.topk_idx.len() != n_tok * top_k || inp.topk_w.len() != n_tok * top_k {
        return Err(Error::Format("inputs tensor shape mismatch".into()));
    }
    let mut m = meta.base(inp.layer)?;
    m.insert("heldout_sequences".into(), inp.heldout_seqs.to_string());
    let x_b = le_bytes(inp.x_bf16, u16::to_le_bytes);
    let idx_b = le_bytes(inp.topk_idx, i32::to_le_bytes);
    let w_b = le_bytes(inp.topk_w, f32::to_le_bytes);
    let tensors = [
        StTensor {
            name: "x",
            dtype: "BF16",
            shape: vec![n_tok, hidden],
            data: &x_b,
        },
        StTensor {
            name: "topk_idx",
            dtype: "I32",
            shape: vec![n_tok, top_k],
            data: &idx_b,
        },
        StTensor {
            name: "topk_w",
            dtype: "F32",
            shape: vec![n_tok, top_k],
            data: &w_b,
        },
    ];
    write_file(path, &m, &tensors)
}

/// Writes the optional per-token routing log `route-LLL.safetensors`.
pub fn write_route(
    path: &Path,
    layer: usize,
    rows: &str,
    idx_i16: &[i16],
    w_f16: &[u16],
    shape: [usize; 3],
    meta: &HessianMeta,
) -> Result<(), Error> {
    let total = shape[0] * shape[1] * shape[2];
    if total == 0 || idx_i16.len() != total || w_f16.len() != total {
        return Err(Error::Format("route tensor shape mismatch".into()));
    }
    let mut m = meta.base(layer)?;
    m.insert("rows".into(), rows.into());
    let idx_b = le_bytes(idx_i16, i16::to_le_bytes);
    let w_b = le_bytes(w_f16, u16::to_le_bytes);
    let tensors = [
        StTensor {
            name: "idx",
            dtype: "I16",
            shape: shape.to_vec(),
            data: &idx_b,
        },
        StTensor {
            name: "w",
            dtype: "F16",
            shape: shape.to_vec(),
            data: &w_b,
        },
    ];
    write_file(path, &m, &tensors)
}

/// Computes `weights_id` from raw `config.json` and index bytes.
pub fn weights_id(model_dir: &Path) -> Result<String, Error> {
    let cfg = fs::read(model_dir.join("config.json"))?;
    let idx = fs::read(model_dir.join("model.safetensors.index.json"))?;
    Ok(weights_id_bytes(&cfg, &idx))
}

/// Computes `weights_id` directly from raw byte slices.
pub(crate) fn weights_id_bytes(config: &[u8], index: &[u8]) -> String {
    sha256_hex(&[config, index])
}

/// Lowercase hex SHA-256 of `parts` read one after another.
pub fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut ctx = ring::digest::Context::new(&ring::digest::SHA256);
    for part in parts {
        ctx.update(part);
    }
    let mut out = String::with_capacity(64);
    for b in ctx.finish().as_ref() {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}
