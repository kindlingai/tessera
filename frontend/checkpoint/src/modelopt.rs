//! NVIDIA ModelOpt checkpoints: NVFP4, MXFP8, FP8 and fused experts.

use crate::file::{Error, File, Shards};
use crate::offer::{Offer, Slice, Stored};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, E4M3, F16, Filed, block};

/// Storage encoding of one quantized or virtually split tensor.
enum Encoding {
    Nvfp4 {
        weight: String,
        scale: String,
        scale2: String,
    },
    Mxfp8 {
        weight: String,
        scale: String,
    },
    Fp8 {
        weight: String,
        scale: String,
    },
    Plain {
        weight: String,
        dtype: String,
    },
}

/// Optional row half of a fused `[.., 2 * rows, cols]` matrix.
#[derive(Clone, Copy)]
enum Split {
    Whole,
    First,
    Second,
}

struct Entry {
    shape: Vec<usize>,
    stored: Stored,
    encoding: Encoding,
    split: Split,
}

/// Safetensors shards carrying ModelOpt NVFP4, MXFP8 or FP8 weights.
pub(crate) struct ModelOpt {
    shards: Shards,
    entries: BTreeMap<String, Entry>,
    consumed: HashSet<String>,
}

/// Companion scale names to hide from plain slice listings.
const AUX_SUFFIXES: [&str; 6] = [
    ".input_scale",
    "_input_scale",
    ".k_scale",
    ".v_scale",
    ".q_scale",
    ".kv_scale",
];

/// Whether `dtype` is a widening float scale type.
fn is_float_dtype(dtype: &str) -> bool {
    matches!(dtype, "F32" | "BF16" | "F16")
}

/// Whether `dtype` is an 8-bit E8M0 block scale type.
fn is_e8m0_dtype(dtype: &str) -> bool {
    matches!(dtype, "U8" | "F8_E8M0")
}

/// Whether `shape` matches `prefix` with `last` on the final axis.
fn same_grid(shape: &[usize], prefix: &[usize], last: usize) -> bool {
    shape.len() == prefix.len() + 1
        && shape[..prefix.len()] == *prefix
        && shape[prefix.len()] == last
}

/// Finds primary and optional secondary scale tensor names for `weight`.
fn find_scales(names: &HashSet<String>, weight: &str) -> (Option<String>, Option<String>) {
    if let Some(stem) = weight.strip_suffix(".weight") {
        let s1 = [
            format!("{stem}.weight_scale"),
            format!("{stem}.weight_scale_inv"),
        ]
        .into_iter()
        .find(|n| names.contains(n));
        let s2_name = format!("{stem}.weight_scale_2");
        let s2 = names.contains(&s2_name).then_some(s2_name);
        return (s1, s2);
    }
    let s1 = [
        format!("{weight}_scale"),
        format!("{weight}_scale_inv"),
        format!("{weight}_weight_scale"),
        format!("{weight}.weight_scale"),
    ]
    .into_iter()
    .find(|n| names.contains(n));
    let s2 = [
        format!("{weight}_scale_2"),
        format!("{weight}_weight_scale_2"),
        format!("{weight}.weight_scale_2"),
    ]
    .into_iter()
    .find(|n| names.contains(n));
    (s1, s2)
}

/// Classifies one raw tensor as NVFP4, MXFP8 or FP8.
fn classify(
    shards: &Shards,
    names: &HashSet<String>,
    weight: &str,
) -> Option<(Vec<usize>, Stored, Encoding, Vec<String>)> {
    let w_dtype = shards.dtype(weight)?;
    let w_shape = shards.shape(weight)?;
    if w_shape.len() < 2 {
        return None;
    }
    let (s1, s2) = find_scales(names, weight);
    let s1 = s1?;
    let s1_dtype = shards.dtype(&s1)?;
    let s1_shape = shards.shape(&s1)?;
    let prefix = &w_shape[..w_shape.len() - 1];
    let last = *w_shape.last()?;
    if w_dtype == "U8"
        && let Some(ref s2) = s2
    {
        let s2_dtype = shards.dtype(s2)?;
        let cols = last * 2;
        if matches!(s1_dtype, "F8_E4M3" | "U8")
            && is_float_dtype(s2_dtype)
            && cols.is_multiple_of(16)
            && same_grid(s1_shape, prefix, cols / 16)
        {
            let mut shape = prefix.to_vec();
            shape.push(cols);
            let used = vec![weight.into(), s1.clone(), s2.clone()];
            let enc = Encoding::Nvfp4 {
                weight: weight.into(),
                scale: s1,
                scale2: s2.clone(),
            };
            return Some((shape, Stored::Form(&block::NVFP4), enc, used));
        }
    }
    if w_dtype == "F8_E4M3" && s2.is_none() {
        if is_e8m0_dtype(s1_dtype)
            && last.is_multiple_of(32)
            && same_grid(s1_shape, prefix, last / 32)
        {
            let used = vec![weight.into(), s1.clone()];
            let enc = Encoding::Mxfp8 {
                weight: weight.into(),
                scale: s1,
            };
            return Some((w_shape.to_vec(), Stored::Form(&block::MXFP8), enc, used));
        }
        if is_float_dtype(s1_dtype) {
            let used = vec![weight.into(), s1.clone()];
            let enc = Encoding::Fp8 {
                weight: weight.into(),
                scale: s1,
            };
            return Some((w_shape.to_vec(), Stored::BF16, enc, used));
        }
    }
    None
}

/// Returns `(gate_name, up_name)` when `name` is fused `gate_up_proj`.
fn fused_names(name: &str) -> Option<(String, String)> {
    let base = name
        .strip_suffix(".gate_up_proj.weight")
        .or_else(|| name.strip_suffix(".gate_up_proj"))?;
    Some((
        format!("{base}.gate_proj.weight"),
        format!("{base}.up_proj.weight"),
    ))
}

/// Whether `name` is an un-suffixed 3D expert matrix name.
fn bare_expert(name: &str, rank: usize) -> bool {
    rank == 3 && !name.ends_with(".weight") && (name.ends_with("_proj") || name.ends_with(".down"))
}

/// Checks if any shard carries ModelOpt quantized or fused tensors.
pub(crate) fn has_quantized(shards: &Shards) -> bool {
    let names = all_names(shards);
    for name in &names {
        if classify(shards, &names, name).is_some() {
            return true;
        }
        if fused_names(name).is_some() {
            return true;
        }
    }
    false
}

fn all_names(shards: &Shards) -> HashSet<String> {
    let mut names = HashSet::new();
    for file in shards.files() {
        for name in file.entry_names() {
            names.insert(name.to_string());
        }
    }
    names
}

fn clone_encoding(enc: &Encoding) -> Encoding {
    match enc {
        Encoding::Nvfp4 {
            weight,
            scale,
            scale2,
        } => Encoding::Nvfp4 {
            weight: weight.clone(),
            scale: scale.clone(),
            scale2: scale2.clone(),
        },
        Encoding::Mxfp8 { weight, scale } => Encoding::Mxfp8 {
            weight: weight.clone(),
            scale: scale.clone(),
        },
        Encoding::Fp8 { weight, scale } => Encoding::Fp8 {
            weight: weight.clone(),
            scale: scale.clone(),
        },
        Encoding::Plain { weight, dtype } => Encoding::Plain {
            weight: weight.clone(),
            dtype: dtype.clone(),
        },
    }
}

/// Inserts virtual `gate_proj` and `up_proj` entries for fused matrices.
fn add_split(
    entries: &mut BTreeMap<String, Entry>,
    names: &HashSet<String>,
    raw: &str,
    shape: &[usize],
    stored: Stored,
    enc: &Encoding,
) {
    let Some((gate, up)) = fused_names(raw) else {
        return;
    };
    if shape.len() < 2 || names.contains(&gate) || names.contains(&up) {
        return;
    }
    let row_axis = shape.len() - 2;
    if !shape[row_axis].is_multiple_of(2) {
        return;
    }
    let mut half = shape.to_vec();
    half[row_axis] /= 2;
    entries.insert(
        gate,
        Entry {
            shape: half.clone(),
            stored,
            encoding: clone_encoding(enc),
            split: Split::First,
        },
    );
    entries.insert(
        up,
        Entry {
            shape: half,
            stored,
            encoding: clone_encoding(enc),
            split: Split::Second,
        },
    );
}

impl ModelOpt {
    /// Wraps `shards`, pairing quantized weights with their scale tensors.
    pub fn new(shards: Shards) -> Self {
        let names = all_names(&shards);
        let mut entries = BTreeMap::new();
        let mut consumed = HashSet::new();
        let mut sorted: Vec<String> = names.iter().cloned().collect();
        sorted.sort();
        for name in &sorted {
            if AUX_SUFFIXES.iter().any(|s| name.ends_with(s)) {
                consumed.insert(name.clone());
                continue;
            }
            if let Some((shape, stored, enc, used)) = classify(&shards, &names, name) {
                for u in used {
                    consumed.insert(u);
                }
                add_split(&mut entries, &names, name, &shape, stored, &enc);
                let out_name = if bare_expert(name, shape.len()) {
                    format!("{name}.weight")
                } else {
                    name.clone()
                };
                entries.insert(
                    out_name,
                    Entry {
                        shape,
                        stored,
                        encoding: enc,
                        split: Split::Whole,
                    },
                );
                continue;
            }
            let Some(shape) = shards.shape(name).map(|s| s.to_vec()) else {
                continue;
            };
            let Some(dtype) = shards.dtype(name).map(str::to_string) else {
                continue;
            };
            let stored = match dtype.as_str() {
                "F32" => Stored::F32,
                "BF16" => Stored::BF16,
                "F16" => Stored::F16,
                _ => Stored::Other,
            };
            if fused_names(name).is_some() {
                let enc = Encoding::Plain {
                    weight: name.clone(),
                    dtype: dtype.clone(),
                };
                add_split(&mut entries, &names, name, &shape, stored, &enc);
            }
            if bare_expert(name, shape.len()) && !names.contains(&format!("{name}.weight")) {
                consumed.insert(name.clone());
                entries.insert(
                    format!("{name}.weight"),
                    Entry {
                        shape,
                        stored,
                        encoding: Encoding::Plain {
                            weight: name.clone(),
                            dtype,
                        },
                        split: Split::Whole,
                    },
                );
            }
        }
        Self {
            shards,
            entries,
            consumed,
        }
    }

    /// Logical `[outer, rows, cols]`, raw row count and row offset.
    fn dims(e: &Entry) -> Result<([usize; 3], usize, usize), Error> {
        let [outer, rows, cols] = match *e.shape.as_slice() {
            [r, c] => [1, r, c],
            [o, r, c] => [o, r, c],
            ref other => {
                return Err(Error::Format(format!(
                    "unsupported quantized rank {}",
                    other.len()
                )));
            }
        };
        let (raw_rows, row_base) = match e.split {
            Split::Whole => (rows, 0),
            Split::First => (rows * 2, 0),
            Split::Second => (rows * 2, rows),
        };
        Ok(([outer, rows, cols], raw_rows, row_base))
    }

    /// Bounds for one axis slice.
    fn axis_box(e: &Entry, axis: u32, (first, count): (usize, usize)) -> Result<[usize; 6], Error> {
        let ([outer, rows, cols], _, _) = Self::dims(e)?;
        let rank = e.shape.len();
        let bounds = match (rank, axis) {
            (2, 0) => [0, 1, first, count, 0, cols],
            (2, 1) => [0, 1, 0, rows, first, count],
            (3, 0) => [first, count, 0, rows, 0, cols],
            (3, 1) => [0, outer, first, count, 0, cols],
            (3, 2) => [0, outer, 0, rows, first, count],
            _ => {
                return Err(Error::Format(format!(
                    "cannot slice rank-{rank} tensor on axis {axis}"
                )));
            }
        };
        if bounds[0] + bounds[1] > outer
            || bounds[2] + bounds[3] > rows
            || bounds[4] + bounds[5] > cols
        {
            return Err(Error::Format("slice exceeds tensor shape".into()));
        }
        Ok(bounds)
    }

    /// Reads an NVFP4 sub-box in `block::NVFP4` layout.
    fn read_nvfp4(
        &self,
        e: &Entry,
        [weight, scale, scale2]: [&str; 3],
        [o0, oc, r0, rc, c0, cc]: [usize; 6],
    ) -> Result<Vec<u8>, Error> {
        let ([outer, _, cols], raw_rows, row_base) = Self::dims(e)?;
        if !c0.is_multiple_of(16) || !cc.is_multiple_of(16) {
            return Err(Error::Format("nvfp4 slice is not 16-aligned".into()));
        }
        let s2 = self.shards.floats(scale2)?;
        if s2.len() != 1 && s2.len() != outer {
            return Err(Error::Format(format!(
                "{scale2}: {} scales for {outer} slices",
                s2.len()
            )));
        }
        let scale_of = |o: usize| if s2.len() == 1 { s2[0] } else { s2[o] };
        let first_s2 = if oc > 0 { scale_of(o0) } else { 1.0 };
        let uniform = (o0..o0 + oc).all(|o| scale_of(o).to_bits() == first_s2.to_bits());
        let head = if uniform {
            first_s2
        } else {
            let top = (o0..o0 + oc)
                .map(|o| scale_of(o).abs())
                .fold(0.0f32, f32::max);
            if top > 0.0 { top } else { 1.0 }
        };
        let (raw_wb, raw_sb) = (cols / 2, cols / 16);
        let (col_b0, blk0, blkc) = (c0 / 2, c0 / 16, cc / 16);
        let w_file = self.shards.holding(weight)?;
        let s_file = self.shards.holding(scale)?;
        let mut out = Vec::with_capacity(4 + oc * rc * blkc * 9);
        out.extend_from_slice(&head.to_le_bytes());
        for o in o0..o0 + oc {
            let row0 = o * raw_rows + row_base + r0;
            let w_bytes = read_rows(w_file, weight, row0, rc, raw_wb)?;
            let s_bytes = read_rows(s_file, scale, row0, rc, raw_sb)?;
            let so = scale_of(o);
            let exact = uniform || so.to_bits() == head.to_bits();
            let ratio = if exact { 1.0 } else { so / head };
            for r in 0..rc {
                let w_row = &w_bytes[r * raw_wb + col_b0..];
                let s_row = &s_bytes[r * raw_sb + blk0..];
                for b in 0..blkc {
                    let sb = if exact {
                        s_row[b]
                    } else {
                        E4M3::from(f32::from(E4M3(s_row[b])) * ratio).0
                    };
                    out.push(sb);
                    out.extend_from_slice(&w_row[b * 8..(b + 1) * 8]);
                }
            }
        }
        Ok(out)
    }

    /// Reads an MXFP8 sub-box in `block::MXFP8` layout.
    fn read_mxfp8(
        &self,
        e: &Entry,
        [weight, scale]: [&str; 2],
        [o0, oc, r0, rc, c0, cc]: [usize; 6],
    ) -> Result<Vec<u8>, Error> {
        let ([_, _, cols], raw_rows, row_base) = Self::dims(e)?;
        if !c0.is_multiple_of(32) || !cc.is_multiple_of(32) {
            return Err(Error::Format("mxfp8 slice is not 32-aligned".into()));
        }
        let (raw_sb, blk0, blkc) = (cols / 32, c0 / 32, cc / 32);
        let w_file = self.shards.holding(weight)?;
        let s_file = self.shards.holding(scale)?;
        let mut out = Vec::with_capacity(oc * rc * blkc * 33);
        for o in o0..o0 + oc {
            let row0 = o * raw_rows + row_base + r0;
            let w_bytes = read_rows(w_file, weight, row0, rc, cols)?;
            let s_bytes = read_rows(s_file, scale, row0, rc, raw_sb)?;
            for r in 0..rc {
                let w_row = &w_bytes[r * cols + c0..];
                let s_row = &s_bytes[r * raw_sb + blk0..];
                for b in 0..blkc {
                    out.push(s_row[b]);
                    out.extend_from_slice(&w_row[b * 32..(b + 1) * 32]);
                }
            }
        }
        Ok(out)
    }

    /// Widens an FP8 sub-box with per-tensor, per-row or block scales.
    fn read_fp8_floats(
        &self,
        e: &Entry,
        [weight, scale]: [&str; 2],
        [o0, oc, r0, rc, c0, cc]: [usize; 6],
    ) -> Result<Vec<f32>, Error> {
        let ([outer, _, cols], raw_rows, row_base) = Self::dims(e)?;
        let scales = self.shards.floats(scale)?;
        let s_shape = self
            .shards
            .shape(scale)
            .ok_or_else(|| Error::Missing(scale.into()))?;
        let grid = fp8_grid(s_shape, [outer, raw_rows, cols], scales.len(), scale)?;
        let w_file = self.shards.holding(weight)?;
        let mut out = Vec::with_capacity(oc * rc * cc);
        for o in o0..o0 + oc {
            let row0 = o * raw_rows + row_base + r0;
            let w_bytes = read_rows(w_file, weight, row0, rc, cols)?;
            for r in 0..rc {
                let raw_r = row_base + r0 + r;
                let w_row = &w_bytes[r * cols + c0..r * cols + c0 + cc];
                for (dc, &b) in w_row.iter().enumerate() {
                    let s = grid.get(&scales, o, raw_r, c0 + dc);
                    out.push(f32::from(E4M3(b)) * s);
                }
            }
        }
        Ok(out)
    }

    /// Reads a plain sub-box as raw bytes.
    fn read_plain_bytes(
        &self,
        e: &Entry,
        weight: &str,
        width: usize,
        [o0, oc, r0, rc, c0, cc]: [usize; 6],
    ) -> Result<Vec<u8>, Error> {
        let ([_, _, cols], raw_rows, row_base) = Self::dims(e)?;
        let w_file = self.shards.holding(weight)?;
        let mut out = Vec::with_capacity(oc * rc * cc * width);
        for o in o0..o0 + oc {
            let row0 = o * raw_rows + row_base + r0;
            let bytes = read_rows(w_file, weight, row0, rc, cols * width)?;
            for r in 0..rc {
                let from = (r * cols + c0) * width;
                out.extend_from_slice(&bytes[from..from + cc * width]);
            }
        }
        Ok(out)
    }

    /// Reads a quantized entry's sub-box as packed form bytes.
    fn read_box(&self, e: &Entry, bounds: [usize; 6]) -> Result<Vec<u8>, Error> {
        match &e.encoding {
            Encoding::Nvfp4 {
                weight,
                scale,
                scale2,
            } => self.read_nvfp4(e, [weight, scale, scale2], bounds),
            Encoding::Mxfp8 { weight, scale } => self.read_mxfp8(e, [weight, scale], bounds),
            Encoding::Fp8 { weight, scale } => {
                let vals = self.read_fp8_floats(e, [weight, scale], bounds)?;
                let mut out = Vec::with_capacity(vals.len() * 2);
                for v in vals {
                    out.extend_from_slice(&Bf16::from(v).0.to_le_bytes());
                }
                Ok(out)
            }
            Encoding::Plain { weight, dtype } if dtype == "BF16" => {
                self.read_plain_bytes(e, weight, 2, bounds)
            }
            Encoding::Plain { weight, .. } => {
                Err(Error::DType(format!("{weight} holds floats, not a form")))
            }
        }
    }

    /// Full bounds for `e`.
    fn whole(e: &Entry) -> Result<[usize; 6], Error> {
        let ([outer, rows, cols], _, _) = Self::dims(e)?;
        Ok([0, outer, 0, rows, 0, cols])
    }
}

/// Reads `count` contiguous rows of `row_bytes` bytes from `file`.
fn read_rows(
    file: &File,
    name: &str,
    row0: usize,
    count: usize,
    row_bytes: usize,
) -> Result<Vec<u8>, Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    file.bytes_at(
        name,
        (row0 * row_bytes) as u64,
        Some((count * row_bytes) as u64),
    )
}

/// Scale lookup grid for FP8 E4M3 weights.
struct Fp8Grid {
    per_outer: bool,
    sr: usize,
    sc: usize,
    br: usize,
    bc: usize,
}

impl Fp8Grid {
    fn get(&self, scales: &[f32], o: usize, r: usize, c: usize) -> f32 {
        let outer = if self.per_outer { o } else { 0 };
        scales[(outer * self.sr + r / self.br) * self.sc + c / self.bc]
    }
}

/// Resolves block dimensions for an FP8 scale tensor.
fn block_size(dim: usize, count: usize, name: &str) -> Result<usize, Error> {
    if count == 1 {
        Ok(dim.max(1))
    } else if dim.div_ceil(128) == count {
        Ok(128)
    } else if dim.is_multiple_of(count) {
        Ok(dim / count)
    } else {
        Err(Error::Format(format!(
            "{name}: {count} scales do not fit axis {dim}"
        )))
    }
}

fn fp8_grid(
    s_shape: &[usize],
    [outer, rows, cols]: [usize; 3],
    len: usize,
    name: &str,
) -> Result<Fp8Grid, Error> {
    if len == 1 {
        return Ok(Fp8Grid {
            per_outer: false,
            sr: 1,
            sc: 1,
            br: rows.max(1),
            bc: cols.max(1),
        });
    }
    if outer > 1 && len == outer {
        return Ok(Fp8Grid {
            per_outer: true,
            sr: 1,
            sc: 1,
            br: rows.max(1),
            bc: cols.max(1),
        });
    }
    let (per_outer, tail) = if outer > 1 && s_shape.first() == Some(&outer) {
        (true, &s_shape[1..])
    } else {
        (false, s_shape)
    };
    let (sr, sc) = match *tail {
        [r] => (r, 1),
        [r, c] => (r, c),
        _ => {
            return Err(Error::Format(format!(
                "{name}: unexpected FP8 scale shape {s_shape:?}"
            )));
        }
    };
    let br = block_size(rows, sr, name)?;
    let bc = block_size(cols, sc, name)?;
    Ok(Fp8Grid {
        per_outer,
        sr,
        sc,
        br,
        bc,
    })
}

impl Offer for ModelOpt {
    fn slices(&self) -> Vec<Slice> {
        let mut out = Vec::new();
        for (name, e) in &self.entries {
            out.push(Slice {
                name: name.clone(),
                shape: e.shape.clone(),
                stored: e.stored,
            });
        }
        for s in self.shards.slices() {
            if !self.consumed.contains(&s.name) {
                out.push(s);
            }
        }
        out
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let Some(e) = self.entries.get(name) else {
            return self.shards.floats(name);
        };
        let bounds = Self::whole(e)?;
        match &e.encoding {
            Encoding::Nvfp4 { .. } => Ok(block::NVFP4.unpack_all(&self.read_box(e, bounds)?)),
            Encoding::Mxfp8 { .. } => Ok(block::MXFP8.unpack_all(&self.read_box(e, bounds)?)),
            Encoding::Fp8 { weight, scale } => self.read_fp8_floats(e, [weight, scale], bounds),
            Encoding::Plain { weight, dtype } => {
                let width = match dtype.as_str() {
                    "F32" => 4,
                    "BF16" | "F16" => 2,
                    _ => return Err(Error::DType(format!("{weight} is {dtype}"))),
                };
                let raw = self.read_plain_bytes(e, weight, width, bounds)?;
                Ok(match dtype.as_str() {
                    "F32" => raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect(),
                    "BF16" => raw
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from(Bf16(u16::from_le_bytes(*b))))
                        .collect(),
                    _ => F16::floats(&raw),
                })
            }
        }
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        let Some(e) = self.entries.get(name) else {
            return self.shards.packed(name);
        };
        self.read_box(e, Self::whole(e)?)
    }

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let Some(e) = self.entries.get(name) else {
            return self.shards.packed_rows(name, rows);
        };
        self.read_box(e, Self::axis_box(e, 0, rows)?)
    }

    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let Some(e) = self.entries.get(name) else {
            return self.shards.packed_share(name, rows);
        };
        self.read_box(e, Self::axis_box(e, 0, rows)?)
    }

    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Result<Vec<u8>, Error> {
        let Some(e) = self.entries.get(name) else {
            return self.shards.packed_axis(name, axis, range);
        };
        self.read_box(e, Self::axis_box(e, axis, range)?)
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        self.shards.i64s(name)
    }

    fn filed(&self, name: &str) -> Option<Arc<dyn Filed>> {
        if self.entries.contains_key(name) {
            return None;
        }
        self.shards.filed(name)
    }
}
