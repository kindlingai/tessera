use crate::round::{Acc, to_bf16, to_f16};
use graph::DType;

const ULP16: f32 = 1.0 / 1024.0;

#[test]
fn f16_ties_even() {
    assert_eq!(to_f16(1.0 + ULP16 / 2.0), 1.0);
    assert_eq!(to_f16(1.0 + 1.5 * ULP16), 1.0 + 2.0 * ULP16);
    assert_eq!(to_f16(1.0 + 0.6 * ULP16), 1.0 + ULP16);
    assert_eq!(to_f16(-2049.0), -2048.0);
    assert_eq!(to_f16(2051.0), 2052.0);
}

#[test]
fn f16_edges() {
    assert_eq!(to_f16(65519.0), 65504.0);
    assert_eq!(to_f16(65520.0), f32::INFINITY);
    assert_eq!(to_f16(-1e6), f32::NEG_INFINITY);
    let tiny = 2f32.powi(-24);
    assert_eq!(to_f16(tiny / 2.0), 0.0);
    assert_eq!(to_f16(1.5 * tiny), 2.0 * tiny);
    assert_eq!(to_f16(2f32.powi(-14) * 0.75), 2f32.powi(-14) * 0.75);
    assert!(to_f16(f32::NAN).is_nan());
}

#[test]
fn bf16_rounds() {
    let ulp = 1.0 / 128.0;
    assert_eq!(to_bf16(1.0 + ulp / 2.0), 1.0);
    assert_eq!(to_bf16(1.0 + 1.5 * ulp), 1.0 + 2.0 * ulp);
    assert_eq!(to_bf16(f32::MAX), f32::INFINITY);
    assert!(to_bf16(f32::NAN).is_nan());
}

#[test]
fn accumulators_round_each_add() {
    let xs = [2048.0, 1.0, 1.0, 1.0];
    assert_eq!(Acc::new(DType::F32).unwrap().sum(xs.into_iter()), 2051.0);
    assert_eq!(Acc::new(DType::F16).unwrap().sum(xs.into_iter()), 2048.0);
    assert_eq!(Acc::new(DType::Bf16).unwrap().sum(xs.into_iter()), 2048.0);
    assert!(Acc::new(DType::I32).is_err());
}
