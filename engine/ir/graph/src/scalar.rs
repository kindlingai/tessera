//! Narrow floats as bit patterns. Widen to f32 to compute.

use std::fmt;

/// The top half of an f32.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bf16(pub u16);

/// IEEE binary16.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct F16(pub u16);

/// OCP float8 e4m3: no infinities, 448 at most.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct E4M3(pub u8);

/// A 4-bit float: sign, two exponent bits, one mantissa bit.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct E2M1(pub u8);

/// `x` rounded at mantissa bit `drop`, ties to even.
fn trim(x: f32, drop: u32) -> u32 {
    let b = x.to_bits();
    let half = (1 << (drop - 1)) - 1 + ((b >> drop) & 1);
    (b + half) & !((1 << drop) - 1)
}

impl From<f32> for Bf16 {
    fn from(x: f32) -> Self {
        if x.is_nan() {
            return Bf16((x.to_bits() >> 16) as u16 | 0x40);
        }
        Bf16((trim(x, 16) >> 16) as u16)
    }
}

impl From<Bf16> for f32 {
    fn from(h: Bf16) -> f32 {
        f32::from_bits(u32::from(h.0) << 16)
    }
}

/// Largest finite f16.
const F16_MAX: f32 = 65504.0;
/// Smallest normal f16, 2^-14.
const F16_NORMAL: f32 = 6.103_515_6e-5;
/// Subnormal f16 steps per unit, 2^24.
const F16_STEPS: f32 = 16_777_216.0;

impl From<f32> for F16 {
    fn from(x: f32) -> Self {
        let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
        if x.is_nan() {
            return F16(sign | 0x7e00);
        }
        if x.abs() < F16_NORMAL {
            let steps = (x.abs() * F16_STEPS).round_ties_even() as u16;
            return F16(sign | steps);
        }
        let r = f32::from_bits(trim(x, 13));
        if r.abs() > F16_MAX {
            return F16(sign | 0x7c00);
        }
        let bits = r.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        F16(sign | ((exp as u16) << 10) | ((bits >> 13) & 0x3ff) as u16)
    }
}

impl From<F16> for f32 {
    fn from(h: F16) -> f32 {
        let sign = if h.0 & 0x8000 != 0 { -1.0 } else { 1.0 };
        let (exp, man) = ((h.0 >> 10) & 0x1f, h.0 & 0x3ff);
        sign * match exp {
            0 => f32::from(man) / F16_STEPS,
            0x1f if man == 0 => f32::INFINITY,
            0x1f => f32::NAN,
            _ => (1.0 + f32::from(man) / 1024.0) * 2f32.powi(i32::from(exp) - 15),
        }
    }
}

/// Largest E4M3.
const E4M3_MAX: f32 = 448.0;
/// Smallest normal E4M3, 2^-6.
const E4M3_NORMAL: f32 = 0.015_625;
/// Subnormal E4M3 steps per unit, 2^9.
const E4M3_STEPS: f32 = 512.0;

impl From<f32> for E4M3 {
    fn from(x: f32) -> Self {
        let sign = ((x.to_bits() >> 24) & 0x80) as u8;
        if x.is_nan() {
            return E4M3(sign | 0x7f);
        }
        if x.abs() < E4M3_NORMAL {
            let steps = (x.abs() * E4M3_STEPS).round_ties_even() as u8;
            return E4M3(sign | steps);
        }
        let r = f32::from_bits(trim(x, 20));
        if r.abs() > E4M3_MAX {
            return E4M3(sign | 0x7e);
        }
        let bits = r.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 7;
        E4M3(sign | ((exp as u8) << 3) | ((bits >> 20) & 7) as u8)
    }
}

impl From<E4M3> for f32 {
    fn from(h: E4M3) -> f32 {
        let sign = if h.0 & 0x80 != 0 { -1.0 } else { 1.0 };
        let (exp, man) = ((h.0 >> 3) & 0xf, h.0 & 7);
        sign * match (exp, man) {
            (0, _) => f32::from(man) / E4M3_STEPS,
            (0xf, 7) => f32::NAN,
            _ => (1.0 + f32::from(man) / 8.0) * 2f32.powi(i32::from(exp) - 7),
        }
    }
}

/// The eight magnitudes an E2M1 can hold.
const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

impl From<f32> for E2M1 {
    /// NaN becomes zero. Past 6 saturates.
    fn from(x: f32) -> Self {
        let sign = ((x.to_bits() >> 28) & 8) as u8;
        if x.is_nan() {
            return E2M1(0);
        }
        let a = x.abs();
        // The nearest magnitude, the even code on a tie.
        let mut code = E2M1_VALUES.partition_point(|v| *v < a).min(7);
        if code > 0 {
            let (below, above) = (E2M1_VALUES[code - 1], E2M1_VALUES[code]);
            let d = (a - below, above - a);
            if d.0 < d.1 || (d.0 == d.1 && code % 2 == 1) {
                code -= 1;
            }
        }
        E2M1(sign | code as u8)
    }
}

impl From<E2M1> for f32 {
    fn from(h: E2M1) -> f32 {
        let v = E2M1_VALUES[usize::from(h.0 & 7)];
        if h.0 & 8 != 0 { -v } else { v }
    }
}

macro_rules! compare_as_f32 {
    ($($t:ident),*) => {$(
        impl PartialOrd for $t {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                f32::from(*self).partial_cmp(&f32::from(*other))
            }
        }

        impl fmt::Debug for $t {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($t), f32::from(*self))
            }
        }
    )*};
}

compare_as_f32!(Bf16, F16, E4M3, E2M1);
