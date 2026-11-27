//! A checkpoint's offer: named slices, read when claimed.

use crate::Error;
use std::sync::Arc;
use tessera_engine::Filed;
use tessera_engine::storage::{Bf16, F16, F32, Form};

/// How a slice's elements lie in the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stored {
    /// A registered form's bytes, in `Form::pack` layout.
    Form(&'static dyn Form),
    /// An element type without a widening reader here.
    Other,
}

impl Stored {
    pub const F32: Self = Self::Form(&F32);
    pub const BF16: Self = Self::Form(&Bf16);
    pub const F16: Self = Self::Form(&F16);

    /// The name manifests write.
    pub fn name(&self) -> &'static str {
        match self {
            Stored::Form(f) => f.name(),
            Stored::Other => "other",
        }
    }

    /// The type `name` stands for, in `Stored::name` spelling.
    pub fn parse(name: &str) -> Result<Stored, Error> {
        match name {
            "other" => Ok(Stored::Other),
            _ => match tessera_engine::storage::named(name) {
                Some(form) => Ok(Stored::Form(form)),
                None => Err(Error::Format(format!("unknown stored type {name:?}"))),
            },
        }
    }
}

/// One tensor on offer, by name, shape and storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Slice {
    pub name: String,
    /// Row-major, the slowest axis first.
    pub shape: Vec<usize>,
    pub stored: Stored,
}

impl Slice {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A checkpoint: slices listed up front, bytes read on demand.
pub trait Offer: Send + Sync {
    fn slices(&self) -> Vec<Slice>;
    /// Slice `name` widened to f32.
    fn floats(&self, name: &str) -> Result<Vec<f32>, Error>;
    /// Slice `name` as its form's bytes.
    fn packed(&self, name: &str) -> Result<Vec<u8>, Error>;
    /// An `I64` slice's values.
    fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        Err(Error::DType(format!("{name} cannot be read as I64")))
    }
    /// Slice `name` left on disk, as stored.
    fn filed(&self, name: &str) -> Option<Arc<dyn Filed>> {
        let _ = name;
        None
    }
    /// Rows `first..first + count` of a packed slice.
    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let _ = rows;
        Err(Error::DType(format!("{name} cannot be read in rows")))
    }
    /// A rank's rows of a packed slice, or the whole.
    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let _ = rows;
        self.packed(name)
    }
    /// One rank's slice of a packed tensor.
    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Result<Vec<u8>, Error> {
        if axis == 0 {
            return self.packed_share(name, range);
        }
        self.packed(name)
    }
}

/// A float's bf16 bytes.
pub(crate) fn narrow(x: &f32) -> [u8; 2] {
    tessera_engine::Bf16::from(*x).0.to_le_bytes()
}
