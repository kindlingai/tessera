use crate::scalar::{Bf16, E2M1, E4M3, F16};

const ULP16: f32 = 1.0 / 1024.0;
const ULP_BF16: f32 = 1.0 / 128.0;
const TINY16: f32 = 1.0 / 16_777_216.0;
const NORMAL16: f32 = 1.0 / 16_384.0;

fn f16(x: f32) -> f32 {
    F16::from(x).into()
}

fn bf16(x: f32) -> f32 {
    Bf16::from(x).into()
}

fn e4m3(x: f32) -> f32 {
    E4M3::from(x).into()
}

fn e2m1(x: f32) -> f32 {
    E2M1::from(x).into()
}

type Round = fn(f32) -> f32;

/// A rounding, its input and the expected result.
const CASES: &[(Round, f32, f32)] = &[
    (f16, 1.0 + ULP16 / 2.0, 1.0),
    (f16, 1.0 + 1.5 * ULP16, 1.0 + 2.0 * ULP16),
    (f16, 1.0 + 0.6 * ULP16, 1.0 + ULP16),
    (f16, -2049.0, -2048.0),
    (f16, 2051.0, 2052.0),
    (f16, 65519.0, 65504.0),
    (f16, 65520.0, f32::INFINITY),
    (f16, -1e6, f32::NEG_INFINITY),
    (f16, TINY16 / 2.0, 0.0),
    (f16, 1.5 * TINY16, 2.0 * TINY16),
    (f16, 0.75 * NORMAL16, 0.75 * NORMAL16),
    (bf16, 1.0 + ULP_BF16 / 2.0, 1.0),
    (bf16, 1.0 + 1.5 * ULP_BF16, 1.0 + 2.0 * ULP_BF16),
    (bf16, f32::MAX, f32::INFINITY),
    (e4m3, 1.0 + 1.0 / 16.0, 1.0),
    (e4m3, 1.0 + 3.0 / 16.0, 1.25),
    (e4m3, 448.0, 448.0),
    (e4m3, -1000.0, -448.0),
    (e4m3, 1.0 / 1024.0, 0.0),
    (e4m3, 3.0 / 1024.0, 1.0 / 256.0),
    (e2m1, 0.25, 0.0),
    (e2m1, 0.75, 1.0),
    (e2m1, 1.25, 1.0),
    (e2m1, 2.5, 2.0),
    (e2m1, 3.5, 4.0),
    (e2m1, -100.0, -6.0),
];

#[test]
fn rounds_ties_even() {
    for (round, x, want) in CASES {
        assert_eq!(round(*x), *want, "{x}");
    }
    assert!(f16(f32::NAN).is_nan() && bf16(f32::NAN).is_nan() && e4m3(f32::NAN).is_nan());
    assert_eq!(e2m1(f32::NAN), 0.0);
    assert_eq!(Bf16::from(1.0).0, 0x3f80);
    assert_eq!(F16::from(1.0).0, 0x3c00);
    assert_eq!(E4M3::from(1.0).0, 0x38);
    assert_eq!(E2M1::from(-6.0).0, 0xf);
    assert!(E2M1::from(1.0) < E2M1::from(1.5));
    assert_eq!(format!("{:?}", Bf16::from(2.0)), "Bf16(2)");
}
