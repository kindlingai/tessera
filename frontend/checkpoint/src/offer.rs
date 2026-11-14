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
}
