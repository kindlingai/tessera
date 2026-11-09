//! Rounding f32 values to narrower floats, ties to even.

use crate::tensor::Error;
use graph::DType;

/// Largest finite f16.
const F16_MAX: f32 = 65504.0;
/// Smallest normal f16, 2^-14.
const F16_NORMAL: f32 = 6.103_515_6e-5;
/// Subnormal f16 steps per unit, 2^24.
const F16_STEPS: f32 = 16_777_216.0;

/// Keeps the top `keep` mantissa bits of a finite f32.
fn trim(x: f32, keep: u32) -> f32 {
    let drop = 23 - keep;
    let b = x.to_bits();
    let half = (1 << (drop - 1)) - 1 + ((b >> drop) & 1);
    f32::from_bits((b + half) & !((1 << drop) - 1))
}

pub(crate) fn to_bf16(x: f32) -> f32 {
    if x.is_nan() { x } else { trim(x, 7) }
}

pub(crate) fn to_f16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x.abs() < F16_NORMAL {
        return (x * F16_STEPS).round_ties_even() / F16_STEPS;
    }
    let r = trim(x, 10);
    if r.abs() > F16_MAX {
        f32::INFINITY.copysign(x)
    } else {
        r
    }
}

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
            DType::F16 => to_f16(s),
            DType::Bf16 => to_bf16(s),
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
