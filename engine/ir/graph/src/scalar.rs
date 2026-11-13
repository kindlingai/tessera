//! Narrow floats as bit patterns. Widen to f32 to compute.

use std::fmt;

/// Encodes `E` exponent bits and `M` mantissa bits.
struct Format<const E: u32, const M: u32, const INF: bool, const NAN: bool>;

impl<const E: u32, const M: u32, const INF: bool, const NAN: bool> Format<E, M, INF, NAN> {
    const BIAS: i32 = (1 << (E - 1)) - 1;
    /// Every magnitude bit set.
    const ONES: u32 = (1 << (E + M)) - 1;
    /// The largest finite code. `INF` reserves the top exponent.
    const TOP: u32 = match (INF, NAN) {
        (true, _) => Self::ONES - (1 << M),
        (false, true) => Self::ONES - 1,
        (false, false) => Self::ONES,
    };
    /// Subnormal steps per unit.
    const STEPS: f64 = pow2(Self::BIAS - 1 + M as i32);

    fn narrow(x: f32) -> u32 {
        let sign = (x.to_bits() >> 31) << (E + M);
        // `NAN` without `INF` keeps one NaN code, all ones.
        if x.is_nan() {
            let quiet = Self::ONES & !((1 << M) - 1) | (1 << (M - 1));
            return sign | if INF { quiet } else { Self::ONES };
        }
        let a = x.abs();
        if a < 2f32.powi(1 - Self::BIAS) {
            return sign | (f64::from(a) * Self::STEPS).round_ties_even() as u32;
        }
        let drop = 23 - M;
        let bits = a.to_bits();
        let half = (1 << (drop - 1)) - 1 + ((bits >> drop) & 1);
        let r = f32::from_bits((bits + half) & !((1 << drop) - 1));
        // Past the top, a type without `INF` saturates.
        if r > Self::widen(Self::TOP) {
            let over = if INF { Self::TOP + 1 } else { Self::TOP };
            return sign | over;
        }
        let exp = ((r.to_bits() >> 23) as i32 - 127 + Self::BIAS) as u32;
        sign | (exp << M) | ((r.to_bits() >> drop) & ((1 << M) - 1))
    }

    fn widen(bits: u32) -> f32 {
        let sign = if bits >> (E + M) & 1 == 1 { -1.0 } else { 1.0 };
        let (exp, man) = ((bits >> M) & ((1 << E) - 1), bits & ((1 << M) - 1));
        let all = (1 << E) - 1;
        sign * match exp {
            0 => (f64::from(man) / Self::STEPS) as f32,
            _ if exp == all && INF => match man {
                0 => f32::INFINITY,
                _ => f32::NAN,
            },
            _ if NAN && !INF && bits & Self::ONES == Self::ONES => f32::NAN,
            _ => (1.0 + man as f32 / (1 << M) as f32) * 2f32.powi(exp as i32 - Self::BIAS),
        }
    }
}

/// Two to the `n`, for a `const`.
const fn pow2(n: i32) -> f64 {
    f64::from_bits(((1023 + n) as u64) << 52)
}

/// A narrow float type over a `Format`, converting through f32.
macro_rules! minifloat {
    ($(#[$doc:meta])* $name:ident($repr:ty): $e:literal, $m:literal, inf = $inf:literal, nan = $nan:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub $repr);

        impl From<f32> for $name {
            fn from(x: f32) -> Self {
                $name(Format::<$e, $m, $inf, $nan>::narrow(x) as $repr)
            }
        }

        impl From<$name> for f32 {
            fn from(h: $name) -> f32 {
                Format::<$e, $m, $inf, $nan>::widen(u32::from(h.0))
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                f32::from(*self).partial_cmp(&f32::from(*other))
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), f32::from(*self))
            }
        }
    };
}

minifloat! {
    /// The top half of an f32.
    Bf16(u16): 8, 7, inf = true, nan = true
}

minifloat! {
    /// IEEE binary16.
    F16(u16): 5, 10, inf = true, nan = true
}

minifloat! {
    /// OCP float8 e4m3: no infinities, 448 at most.
    E4M3(u8): 4, 3, inf = false, nan = true
}

minifloat! {
    /// A 4-bit float, 6 at most. NaN saturates.
    E2M1(u8): 2, 1, inf = false, nan = false
}
