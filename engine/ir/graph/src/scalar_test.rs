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
    assert_eq!(e2m1(f32::NAN), 6.0, "no NaN code, so it saturates");
    let wire = |x| crate::types::DType::F16.round(x);
    assert_eq!([wire(1e9), wire(f32::NEG_INFINITY)], [65504.0, -65504.0]);
    assert!(wire(f32::NAN).is_nan());
    assert_eq!(Bf16::from(1.0).0, 0x3f80);
    assert_eq!(F16::from(1.0).0, 0x3c00);
    assert_eq!(E4M3::from(1.0).0, 0x38);
    assert_eq!(E2M1::from(-6.0).0, 0xf);
    assert!(E2M1::from(1.0) < E2M1::from(1.5));
    assert_eq!(format!("{:?}", Bf16::from(2.0)), "Bf16(2)");
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
fn widen_matches_format() {
    use crate::scalar::{E8M0, Format};

    for bits in 0..=u16::MAX {
        let b = u32::from(bits);
        assert!(same_f32(
            Bf16(bits).to_f32(),
            Format::<8, 7, true, true>::widen(b)
        ));
        assert!(same_f32(
            F16(bits).to_f32(),
            Format::<5, 10, true, true>::widen(b)
        ));
    }
    for bits in 0..=u8::MAX {
        let b = u32::from(bits);
        assert!(same_f32(
            E4M3(bits).to_f32(),
            Format::<4, 3, false, true>::widen(b)
        ));
        assert!(same_f32(
            E2M1(bits).to_f32(),
            Format::<2, 1, false, false>::widen(b)
        ));
        let want = match bits {
            0xFF => f32::NAN,
            n => f64::from_bits(((1023 + i32::from(n) - 127) as u64) << 52) as f32,
        };
        assert!(same_f32(f32::from(E8M0(bits)), want));
    }
}

/// Checks ties-to-even optimality against adjacent magnitude codes.
fn check_ties_even(x: f32, code: u32, sign_bit: u32, top: u32, inf: bool, widen: fn(u32) -> f32) {
    let want_sign = (x.to_bits() >> 31) << sign_bit;
    assert_eq!(code & (1 << sign_bit), want_sign, "{x:?} sign");
    let mag = code & ((1 << sign_bit) - 1);
    if x.is_nan() {
        assert_ne!(mag, 0);
        return;
    }
    let a = f64::from(x.abs());
    let top_val = f64::from(widen(top));
    if a > top_val {
        let want = if inf && a > top_val + (top_val - f64::from(widen(top - 1))) * 0.5 {
            top + 1
        } else {
            top
        };
        assert_eq!(mag, want, "{x:?} overflow");
        return;
    }
    assert!(mag <= top, "{x:?} exceeded top");
    let cur = (a - f64::from(widen(mag))).abs();
    if mag > 0 {
        let prev = (a - f64::from(widen(mag - 1))).abs();
        assert!(
            cur < prev || (cur == prev && mag.is_multiple_of(2)),
            "{x:?} prev"
        );
    }
    if mag < top {
        let next = (a - f64::from(widen(mag + 1))).abs();
        assert!(
            cur < next || (cur == next && mag.is_multiple_of(2)),
            "{x:?} next"
        );
    }
}

#[test]
fn narrow_matches_ties_even() {
    let mantissas = [
        0, 1, 0x3fff, 0x7fff, 0x8000, 0x8001, 0x7_ffff, 0x8_0000, 0x10_0000, 0x20_0000, 0x3f_ffff,
        0x40_0000, 0x7f_ffff,
    ];
    for sign in [0u32, 0x8000_0000] {
        for exp in 0u32..=255 {
            for man in mantissas {
                let x = f32::from_bits(sign | (exp << 23) | man);
                check_ties_even(x, u32::from(Bf16::from(x).0), 15, 0x7f7f, true, |c| {
                    Bf16(c as u16).to_f32()
                });
                check_ties_even(x, u32::from(F16::from(x).0), 15, 0x7bff, true, |c| {
                    F16(c as u16).to_f32()
                });
                check_ties_even(x, u32::from(E4M3::from(x).0), 7, 0x7e, false, |c| {
                    E4M3(c as u8).to_f32()
                });
                check_ties_even(x, u32::from(E2M1::from(x).0), 3, 0x7, false, |c| {
                    E2M1(c as u8).to_f32()
                });
            }
        }
    }
}
