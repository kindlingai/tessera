//! PyTorch zip checkpoints, read without running their pickles.

use crate::Error;
use crate::offer::{Offer, Slice, Stored};
use crate::pickle::{self, DType, MAX_PICKLE, Value, View};
use crate::zip::{self, Entry};
use sensai_engine::{Bf16, F16};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

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
        DType::Bf16 => Stored::Bf16,
        _ => Stored::Other,
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
        let width = view.storage.dtype.width();
        let end = view.end().ok_or_else(|| bad("a view overflows"))?;
        let first = view.offset.min(end);
        let bytes = zip::read_at(
            &self.file,
            entry.start + first * width,
            (end - first) * width,
            self.size,
        )?;
        let count: usize = view.shape.iter().product();
        let mut out = Vec::with_capacity(count);
        let mut index = vec![0usize; view.shape.len()];
        for _ in 0..count {
            let mut at = view.offset - first;
            for (&i, &s) in index.iter().zip(&view.stride) {
                at += i as u64 * s;
            }
            out.push(widen(view.storage.dtype, &bytes[(at * width) as usize..]));
            for axis in (0..index.len()).rev() {
                index[axis] += 1;
                if index[axis] < view.shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
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
        Err(Error::DType(format!("{name} holds floats, not a form")))
    }
}
