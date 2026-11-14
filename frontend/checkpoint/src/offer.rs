//! What a checkpoint offers: named slices, read when claimed.

use crate::Error;
use codegen::form::Form;

/// How a slice's elements lie in the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stored {
    F32,
    Bf16,
    F16,
    /// A registered form's bytes, as `Form::pack` lays them out.
    Form(&'static dyn Form),
    /// An element type no reader here widens.
    Other,
}

impl Stored {
    /// The name manifests write.
    pub fn name(&self) -> &'static str {
        match self {
            Stored::F32 => "f32",
            Stored::Bf16 => "bf16",
            Stored::F16 => "f16",
            Stored::Form(f) => f.name(),
            Stored::Other => "other",
        }
    }

    /// The type `name` stands for, as `Stored::name` writes it.
    pub fn parse(name: &str) -> Result<Stored, Error> {
        match name {
            "f32" => Ok(Stored::F32),
            "bf16" => Ok(Stored::Bf16),
            "f16" => Ok(Stored::F16),
            "other" => Ok(Stored::Other),
            _ => match codegen::form::named(name) {
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
pub trait Offer {
    fn slices(&self) -> Vec<Slice>;
    /// Slice `name` widened to f32.
    fn floats(&self, name: &str) -> Result<Vec<f32>, Error>;
    /// Slice `name` as its form's bytes. Floats have none.
    fn packed(&self, name: &str) -> Result<Vec<u8>, Error>;
    /// Rows `first..first + count` of a packed slice.
    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let _ = rows;
        Err(Error::DType(format!("{name} is not read by rows")))
    }
}
