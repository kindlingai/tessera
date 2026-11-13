//! Accumulators that round to narrower floats after every add.

use crate::tensor::Error;
use graph::{Bf16, DType, F16};

/// An accumulator that rounds after every add.
#[derive(Clone, Copy)]
pub(crate) struct Acc(DType);

impl Acc {
    pub(crate) fn new(dtype: DType) -> Result<Acc, Error> {
        match dtype {
            DType::F32 | DType::F16 | DType::Bf16 => Ok(Acc(dtype)),
            other => Err(Error::Unsupported(format!("{other:?} accumulator"))),
        }
    }

    pub(crate) fn add(self, sum: f32, v: f32) -> f32 {
        let s = sum + v;
        match self.0 {
            DType::F16 => F16::from(s).into(),
            DType::Bf16 => Bf16::from(s).into(),
            _ => s,
        }
    }

    pub(crate) fn sum(self, values: impl Iterator<Item = f32>) -> f32 {
        values.fold(0.0, |s, v| self.add(s, v))
    }

    pub(crate) fn dot(self, a: &[f32], b: &[f32]) -> f32 {
        self.sum(a.iter().zip(b).map(|(x, y)| x * y))
    }
}
