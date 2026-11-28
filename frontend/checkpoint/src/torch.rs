//! PyTorch zip checkpoints, read without running their pickles.

use crate::Error;
use crate::offer::{Offer, Slice, Stored};
use crate::pickle::{self, DType, MAX_PICKLE, Value, View};
use crate::zip::{self, Entry};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, F16};

/// A `torch.save` file. Nested dict keys join with `.`.
pub struct Torch {
    file: fs::File,
    size: u64,
    views: BTreeMap<String, (View, Entry)>,
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Format(format!("torch: {what}"))
}

/// Tensors under `name`, found through string-keyed dicts.
fn flatten(name: &str, v: Value, out: &mut BTreeMap<String, View>) {
    match v {
        Value::Tensor(view) => {
            out.insert(name.to_string(), view);
        }
        Value::Dict(pairs) => {
            for (k, v) in pairs {
                let Value::Str(k) = k else { continue };
                let joined = if name.is_empty() {
                    k
                } else {
                    format!("{name}.{k}")
                };
                flatten(&joined, v, out);
            }
        }
        _ => {}
    }
}

fn stored(dtype: DType) -> Stored {
    match dtype {
        DType::F32 => Stored::F32,
        DType::F16 => Stored::F16,
        DType::Bf16 => Stored::BF16,
        _ => Stored::Other,
    }
}

/// The form of a two-byte float type.
fn half(dtype: DType) -> Option<&'static dyn Form> {
    match stored(dtype) {
        Stored::Form(f) if f.dtype().bytes() == 2 => Some(f),
        _ => None,
    }
}

/// Whether a view is contiguous in row-major order.
fn is_contiguous(view: &View) -> bool {
    if view.shape.len() != view.stride.len() {
        return false;
    }
    let mut step = 1u64;
    for (&dim, &stride) in view.shape.iter().zip(&view.stride).rev() {
        if dim > 1 {
            if stride != step {
                return false;
            }
            let Some(next) = step.checked_mul(dim as u64) else {
                return false;
            };
            step = next;
        }
    }
    true
}

/// Widens contiguous bytes of `dtype` into f32 values.
fn widen_contiguous(dtype: DType, bytes: &[u8]) -> Vec<f32> {
    match dtype {
        DType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F16 => F16::floats(bytes),
        DType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect(),
        DType::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        DType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        DType::I16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c) as f32)
            .collect(),
        DType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => bytes.iter().map(|&b| f32::from(b)).collect(),
    }
}

/// Reads little-endian f32 values directly from `file`.
#[cfg(all(unix, target_endian = "little"))]
fn read_f32_direct(file: &fs::File, at: u64, count: usize, size: u64) -> Result<Vec<f32>, Error> {
    use std::os::unix::fs::FileExt;
    let len = (count as u64)
        .checked_mul(4)
        .ok_or_else(|| bad("byte length overflow"))?;
    if at.checked_add(len).is_none_or(|end| end > size) {
        return Err(bad("a range runs past the end"));
    }
    let mut out = vec![0.0f32; count];
    // SAFETY: `out` is initialized memory valid for byte writes.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * 4) };
    file.read_exact_at(bytes, at)?;
    Ok(out)
}

/// Visits strided element offsets in row-major order.
fn visit_strided(view: &View, first: u64, count: usize, mut take: impl FnMut(usize)) {
    let mut index = vec![0usize; view.shape.len()];
    for _ in 0..count {
        let mut at = view.offset - first;
        for (&i, &s) in index.iter().zip(&view.stride) {
            at += i as u64 * s;
        }
        take(at as usize);
        for axis in (0..index.len()).rev() {
            index[axis] += 1;
            if index[axis] < view.shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// One element at the start of `b`, as f32.
fn widen(dtype: DType, b: &[u8]) -> f32 {
    let two = || u16::from_le_bytes([b[0], b[1]]);
    match dtype {
        DType::F64 => f64::from_le_bytes(b[..8].try_into().expect("8 bytes")) as f32,
        DType::F32 => f32::from_le_bytes(b[..4].try_into().expect("4 bytes")),
        DType::F16 => f32::from(F16(two())),
        DType::Bf16 => f32::from(Bf16(two())),
        DType::I64 => i64::from_le_bytes(b[..8].try_into().expect("8 bytes")) as f32,
        DType::I32 => i32::from_le_bytes(b[..4].try_into().expect("4 bytes")) as f32,
        DType::I16 => two() as i16 as f32,
        DType::I8 => b[0] as i8 as f32,
        DType::U8 | DType::Bool => b[0] as f32,
    }
}

impl Torch {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = fs::File::open(path)?;
        let size = file.metadata()?.len();
        let entries = zip::entries(&file)?;
        let mut pickles = entries.keys().filter_map(|n| n.strip_suffix("/data.pkl"));
        let (Some(root), None) = (pickles.next(), pickles.next()) else {
            return Err(bad("want one data.pkl"));
        };
        let root = root.to_string();
        if root.contains('/') || entries.keys().any(|n| !n.starts_with(&format!("{root}/"))) {
            return Err(bad("an entry lies outside the archive's directory"));
        }
        let read = |name: &str, most: usize| -> Result<Option<Vec<u8>>, Error> {
            let Some(e) = entries.get(&format!("{root}/{name}")) else {
                return Ok(None);
            };
            if e.len > most as u64 {
                return Err(bad(format_args!("{name} is too large")));
            }
            zip::read_at(&file, e.start, e.len, size).map(Some)
        };
        if let Some(order) = read("byteorder", 16)?
            && order != b"little"
        {
            return Err(bad("big-endian storages are unsupported"));
        }
        let pkl = read("data.pkl", MAX_PICKLE)?.expect("found above");
        let mut found = BTreeMap::new();
        flatten("", pickle::parse(&pkl)?, &mut found);
        let mut views = BTreeMap::new();
        for (name, view) in found {
            let s = &view.storage;
            let entry = entries.get(&format!("{root}/data/{}", s.key));
            let entry = *entry.ok_or_else(|| bad(format_args!("no storage {}", s.key)))?;
            if s.count.checked_mul(s.dtype.width()) != Some(entry.len) {
                return Err(bad(format_args!("storage {} has the wrong size", s.key)));
            }
            views.insert(name, (view, entry));
        }
        Ok(Self { file, size, views })
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.views.get(name).map(|(v, _)| v.shape.as_slice())
    }

    /// A tensor's values in row-major order, widened to f32.
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let (view, entry) = self
            .views
            .get(name)
            .ok_or_else(|| Error::Missing(name.into()))?;
        let count: usize = view.shape.iter().product();
        if count == 0 {
            return Ok(Vec::new());
        }
        let width = view.storage.dtype.width();
        let end = view.end().ok_or_else(|| bad("a view overflows"))?;
        let first = view.offset.min(end);
        let byte_start = entry.start + first * width;
        let byte_len = (end - first) * width;
        if is_contiguous(view) {
            #[cfg(all(unix, target_endian = "little"))]
            if view.storage.dtype == DType::F32 {
                return read_f32_direct(&self.file, byte_start, count, self.size);
            }
            let bytes = zip::read_at(&self.file, byte_start, byte_len, self.size)?;
            return Ok(widen_contiguous(view.storage.dtype, &bytes));
        }
        let bytes = zip::read_at(&self.file, byte_start, byte_len, self.size)?;
        let mut out = Vec::with_capacity(count);
        let elem_w = width as usize;
        visit_strided(view, first, count, |at| {
            out.push(widen(view.storage.dtype, &bytes[at * elem_w..]));
        });
        Ok(out)
    }
}

impl Offer for Torch {
    fn slices(&self) -> Vec<Slice> {
        (self.views.iter())
            .map(|(name, (v, _))| Slice {
                name: name.clone(),
                shape: v.shape.clone(),
                stored: stored(v.storage.dtype),
            })
            .collect()
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        Torch::floats(self, name)
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        let (view, entry) = self
            .views
            .get(name)
            .ok_or_else(|| Error::Missing(name.into()))?;
        if half(view.storage.dtype).is_none() {
            return Err(Error::DType(format!("{name} holds floats, not a form")));
        }
        let count: usize = view.shape.iter().product();
        if count == 0 {
            return Ok(Vec::new());
        }
        let end = view.end().ok_or_else(|| bad("a view overflows"))?;
        let first = view.offset.min(end);
        let byte_start = entry.start + first * 2;
        let byte_len = (end - first) * 2;
        if is_contiguous(view) {
            return zip::read_at(&self.file, byte_start, byte_len, self.size);
        }
        let bytes = zip::read_at(&self.file, byte_start, byte_len, self.size)?;
        let mut out = Vec::with_capacity(count * 2);
        visit_strided(view, first, count, |at| {
            out.extend_from_slice(&bytes[at * 2..at * 2 + 2]);
        });
        Ok(out)
    }

    fn packed_rows(&self, name: &str, (first, count): (usize, usize)) -> Result<Vec<u8>, Error> {
        let (view, entry) = self
            .views
            .get(name)
            .ok_or_else(|| Error::Missing(name.into()))?;
        if half(view.storage.dtype).is_none() {
            return Err(Error::DType(format!("{name} cannot be read in rows")));
        }
        let rows = *view.shape.first().unwrap_or(&0);
        if rows == 0 || count == 0 {
            return Ok(Vec::new());
        }
        if first.checked_add(count).is_none_or(|end| end > rows) {
            return Err(bad("row slice runs past the end"));
        }
        let row_elems: usize = view.shape[1..].iter().product();
        if is_contiguous(view) {
            let _ = view.end().ok_or_else(|| bad("a view overflows"))?;
            let off = entry.start + (view.offset + (first * row_elems) as u64) * 2;
            let len = (count * row_elems) as u64 * 2;
            return zip::read_at(&self.file, off, len, self.size);
        }
        let stride = 2 * row_elems;
        let all = self.packed(name)?;
        Ok(all[first * stride..(first + count) * stride].to_vec())
    }
}
