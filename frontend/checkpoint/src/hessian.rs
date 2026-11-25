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

fn pack_tiles_into<T: Copy>(tiles: &[T], d: usize, tile: usize, out: &mut [T]) {
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

/// Converts packed upper-triangle tiles into row-major upper triangle.
pub fn pack_tiles(tiles: &[f32], d: usize, tile: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; triu_len(d)];
    pack_tiles_into(tiles, d, tile, &mut out);
    out
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

/// Writes `hess-LLL-eAAA-BBB.safetensors` from little-endian FP32 buffers.
#[allow(clippy::too_many_arguments)]
pub fn write_experts_bytes(
    path: &Path,
    layer: usize,
    first_expert: usize,
    gu_bytes: &[u8],
    down_bytes: &[u8],
    counts: &[i64],
    dims: (usize, usize),
    meta: &HessianMeta,
) -> Result<(), Error> {
    let n = counts.len();
    if n == 0 {
        return Err(Error::Format(
            "expert shard requires at least 1 expert".into(),
        ));
    }
    let (h, i) = dims;
    let (th, ti) = (triu_len(h), triu_len(i));
    if h == 0 || i == 0 || gu_bytes.len() != n * th * 4 || down_bytes.len() != n * ti * 4 {
        return Err(Error::Format("expert tensor shape mismatch".into()));
    }
    let mut m = meta.base(layer)?;
    let last = first_expert + n - 1;
    m.insert("experts".into(), format!("{first_expert}-{last}"));
    m.insert("layout".into(), EXPERT_LAYOUT.into());
    let ct_b = le_bytes(counts, i64::to_le_bytes);
    let tensors = [
        StTensor {
            name: "gu",
            dtype: "F32",
            shape: vec![n, th],
            data: gu_bytes,
        },
        StTensor {
            name: "down",
            dtype: "F32",
            shape: vec![n, ti],
            data: down_bytes,
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

/// Writes `hess-LLL-eAAA-BBB.safetensors` matching `HESSIAN.md`.
#[allow(clippy::too_many_arguments)]
pub fn write_experts(
    path: &Path,
    layer: usize,
    first_expert: usize,
    gu: &[f32],
    down: &[f32],
    counts: &[i64],
    dims: (usize, usize),
    meta: &HessianMeta,
) -> Result<(), Error> {
    let gu_b = le_bytes(gu, f32::to_le_bytes);
    let dn_b = le_bytes(down, f32::to_le_bytes);
    write_experts_bytes(path, layer, first_expert, &gu_b, &dn_b, counts, dims, meta)
}

/// Writes `hess-LLL-dense.safetensors` or `-dense-heldout.safetensors`.
pub fn write_dense(
    path: &Path,
    layer: usize,
    entries: &[DenseEntry<'_>],
    heldout: bool,
    meta: &HessianMeta,
) -> Result<(), Error> {
    let has_mlp = entries.iter().any(|e| e.name == "mlp_in");
    if !has_mlp {
        return Err(Error::Format("dense file requires mlp_in".into()));
    }
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

/// Writes `inputs-LLL.safetensors` matching `HESSIAN.md`.
#[allow(clippy::too_many_arguments)]
pub fn write_inputs(
    path: &Path,
    layer: usize,
    heldout_seqs: usize,
    x_bf16: &[u16],
    topk_idx: &[i32],
    topk_w: &[f32],
    hidden: usize,
    top_k: usize,
    meta: &HessianMeta,
) -> Result<(), Error> {
    if heldout_seqs == 0 || hidden == 0 || top_k == 0 {
        return Err(Error::Format("inputs require positive dimensions".into()));
    }
    if !x_bf16.len().is_multiple_of(hidden) {
        return Err(Error::Format("x length not divisible by hidden".into()));
    }
    let n_tok = x_bf16.len() / hidden;
    if n_tok == 0 || topk_idx.len() != n_tok * top_k || topk_w.len() != n_tok * top_k {
        return Err(Error::Format("inputs tensor shape mismatch".into()));
    }
    let mut m = meta.base(layer)?;
    m.insert("heldout_sequences".into(), heldout_seqs.to_string());
    let x_b = le_bytes(x_bf16, u16::to_le_bytes);
    let idx_b = le_bytes(topk_idx, i32::to_le_bytes);
    let w_b = le_bytes(topk_w, f32::to_le_bytes);
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

/// Writes optional `route-LLL.safetensors` matching `calib_forward.py`.
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
    let mut hasher = Sha256::new();
    hasher.update(config);
    hasher.update(index);
    hasher.finish_hex()
}

/// Minimal FIPS 180-4 SHA-256 implementation with zero dependencies.
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    len: usize,
    total: u64,
}

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            len: 0,
            total: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, slot) in w.iter_mut().enumerate().take(16) {
            let at = i * 4;
            *slot = u32::from_be_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        let add = [a, b, c, d, e, f, g, h];
        for (dst, src) in self.state.iter_mut().zip(add) {
            *dst = dst.wrapping_add(src);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.len).min(data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.len = 0;
            }
        }
    }

    fn finish_hex(mut self) -> String {
        let bits = self.total * 8;
        self.update(&[0x80]);
        if self.len > 56 {
            let pad = [0u8; 64];
            self.update(&pad[..64 - self.len]);
        }
        let pad = [0u8; 56];
        self.update(&pad[..56 - self.len]);
        self.update(&bits.to_be_bytes());
        let mut out = String::with_capacity(64);
        for word in self.state {
            use std::fmt::Write as _;
            let _ = write!(out, "{word:08x}");
        }
        out
    }
}
