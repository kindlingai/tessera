use crate::round::{Acc, to_bf16, to_f16};
use graph::DType;

const ULP16: f32 = 1.0 / 1024.0;
const ULP_BF16: f32 = 1.0 / 128.0;
const TINY16: f32 = 1.0 / 16_777_216.0;
const NORMAL16: f32 = 1.0 / 16_384.0;

type Round = fn(f32) -> f32;

/// A rounding, its input and the expected result.
const CASES: &[(Round, f32, f32)] = &[
    (to_f16, 1.0 + ULP16 / 2.0, 1.0),
    (to_f16, 1.0 + 1.5 * ULP16, 1.0 + 2.0 * ULP16),
    (to_f16, 1.0 + 0.6 * ULP16, 1.0 + ULP16),
    (to_f16, -2049.0, -2048.0),
    (to_f16, 2051.0, 2052.0),
    (to_f16, 65519.0, 65504.0),
    (to_f16, 65520.0, f32::INFINITY),
    (to_f16, -1e6, f32::NEG_INFINITY),
    (to_f16, TINY16 / 2.0, 0.0),
    (to_f16, 1.5 * TINY16, 2.0 * TINY16),
    (to_f16, 0.75 * NORMAL16, 0.75 * NORMAL16),
    (to_bf16, 1.0 + ULP_BF16 / 2.0, 1.0),
    (to_bf16, 1.0 + 1.5 * ULP_BF16, 1.0 + 2.0 * ULP_BF16),
    (to_bf16, f32::MAX, f32::INFINITY),
];

/// An accumulator's sum of 2048 and three ones.
const SUMS: [(DType, f32); 3] = [
    (DType::F32, 2051.0),
    (DType::F16, 2048.0),
    (DType::Bf16, 2048.0),
];

fn sums(&(dtype, want): &(DType, f32)) {
    let xs = [2048.0, 1.0, 1.0, 1.0];
    assert_eq!(
        Acc::new(dtype).unwrap().sum(xs.into_iter()),
        want,
        "{dtype:?}"
    );
}

#[test]
fn rounds_ties_even() {
    for (i, &(round, x, want)) in CASES.iter().enumerate() {
        assert_eq!(round(x), want, "case {i}: {x}");
    }
    assert!(to_f16(f32::NAN).is_nan() && to_bf16(f32::NAN).is_nan());
    SUMS.iter().for_each(sums);
    assert!(Acc::new(DType::I32).is_err());
}
