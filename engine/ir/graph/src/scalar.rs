//! Narrow floats as bit patterns. Widen to f32 to compute.

use std::fmt;

/// Encodes `E` exponent bits and `M` mantissa bits.
pub(crate) struct Format<const E: u32, const M: u32, const INF: bool, const NAN: bool>;

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
    /// Subnormal steps as f32, for formats narrower than f32.
    const STEPS_F32: f32 = Self::STEPS as f32;
    /// Smallest normal f32 bit pattern for this format.
    const MIN_NORM_BITS: u32 = ((128 - Self::BIAS) as u32) << 23;
    /// Largest finite f32 bit pattern for this format.
    const TOP_BITS: u32 = Self::widen(Self::TOP).to_bits();
    /// Payload bits for NaN.
    const NAN_CODE: u32 = if INF {
        Self::ONES & !((1 << M) - 1) | (1 << (M - 1))
    } else {
        Self::ONES
    };
    /// Code used on overflow: infinity or saturation.
    const OVER_CODE: u32 = if INF { Self::TOP + 1 } else { Self::TOP };
    /// Exponent bias adjustment shifted to the mantissa boundary.
    const EXP_OFFSET: u32 = ((127 - Self::BIAS) as u32) << M;

    #[inline(always)]
    fn narrow(x: f32) -> u32 {
        let bits = x.to_bits();
        let sign = (bits >> 31) << (E + M);
        let mag = bits & 0x7fff_ffff;
        let drop = 23 - M;
        let half = (1 << (drop - 1)) - 1 + ((mag >> drop) & 1);
        let r = mag.wrapping_add(half) & !((1 << drop) - 1);
        let norm = (r >> drop).wrapping_sub(Self::EXP_OFFSET);
        let clamped = if r > Self::TOP_BITS {
            Self::OVER_CODE
        } else {
            norm
        };
        // Exact power-of-two scaling with magic-number ties-to-even rounding.
        let scaled = x.abs() * Self::STEPS_F32;
        let sub = (scaled + 8_388_608.0f32)
            .to_bits()
            .wrapping_sub(0x4b00_0000);
        let body = if mag > 0x7f80_0000 {
            Self::NAN_CODE
        } else if mag < Self::MIN_NORM_BITS {
            sub
        } else {
            clamped
        };
        sign | body
    }

    pub(crate) const fn widen(bits: u32) -> f32 {
        let sign = if bits >> (E + M) & 1 == 1 { -1.0 } else { 1.0 };
        let (exp, man) = ((bits >> M) & ((1 << E) - 1), bits & ((1 << M) - 1));
        let all = (1 << E) - 1;
        sign * match exp {
            0 => (man as f64 / Self::STEPS) as f32,
            _ if exp == all && INF => match man {
                0 => f32::INFINITY,
                _ => f32::NAN,
            },
            _ if NAN && !INF && bits & Self::ONES == Self::ONES => f32::NAN,
            _ => (1.0 + man as f32 / (1 << M) as f32) * pow2(exp as i32 - Self::BIAS) as f32,
        }
    }
}

/// Two to the `n`, for a `const`.
const fn pow2(n: i32) -> f64 {
    f64::from_bits(((1023 + n) as u64) << 52)
}

/// Widens an IEEE binary16 bit pattern to f32.
#[inline(always)]
const fn widen_f16(bits: u32) -> f32 {
    let sign = (bits & 0x8000) << 16;
    let mag = bits & 0x7fff;
    let inf_nan = if mag >= 0x7c00 { 0x7f80_0000 } else { 0 };
    f32::from_bits(sign | (mag << 13) | inf_nan) * f32::from_bits(0x7780_0000)
}

/// Rounds f32 to bf16, ties to even.
#[inline(always)]
fn narrow_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16;
    let nan = ((bits >> 16) & 0x8000) | 0x7fc0;
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        nan as u16
    } else {
        round as u16
    }
}

/// Widens a bf16 bit pattern to f32.
#[inline(always)]
const fn widen_bf16(bits: u32) -> f32 {
    f32::from_bits(bits << 16)
}

/// Lookup table for OCP float8 e4m3 decoding.
const E4M3_F32: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = Format::<4, 3, false, true>::widen(i as u32);
        i += 1;
    }
    table
};

/// Widens an OCP float8 e4m3 byte via table.
#[inline(always)]
const fn widen_e4m3(bits: u32) -> f32 {
    E4M3_F32[(bits & 0xff) as usize]
}

/// Widens a 4-bit e2m1 nibble to f32.
#[inline(always)]
const fn widen_e2m1(bits: u32) -> f32 {
    let sign = (bits & 0x8) << 28;
    let mag = bits & 0x7;
    let body = if mag == 0 {
        0
    } else if mag == 1 {
        0x3f00_0000
    } else {
        (mag + 252) << 22
    };
    f32::from_bits(sign | body)
}

/// A narrow float type over a `Format`, converting through f32.
macro_rules! minifloat {
    ($(#[$doc:meta])* $name:ident($repr:ty): $e:literal, $m:literal, inf = $inf:literal, nan = $nan:literal, widen = $widen:path, narrow = $narrow:path) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(pub $repr);

        impl $name {
            pub const EXPONENT: u32 = $e;
            pub const MANTISSA: u32 = $m;

            /// The exact value as an f32.
            #[inline(always)]
            pub const fn to_f32(self) -> f32 {
                $widen(self.0 as u32)
            }
        }

        impl From<f32> for $name {
            #[inline(always)]
            fn from(x: f32) -> Self {
                $name($narrow(x) as $repr)
            }
        }

        impl From<$name> for f32 {
            #[inline(always)]
            fn from(h: $name) -> f32 {
                h.to_f32()
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
    Bf16(u16): 8, 7, inf = true, nan = true, widen = widen_bf16, narrow = narrow_bf16
}

minifloat! {
    /// IEEE binary16.
    F16(u16): 5, 10, inf = true, nan = true, widen = widen_f16, narrow = Format::<5, 10, true, true>::narrow
}

impl F16 {
    /// Little-endian halves as floats.
    pub fn floats(bytes: &[u8]) -> Vec<f32> {
        let half = |b: &[u8; 2]| f32::from(F16(u16::from_le_bytes(*b)));
        bytes.as_chunks::<2>().0.iter().map(half).collect()
    }
}

minifloat! {
    /// OCP float8 e4m3: no infinities, 448 at most.
    E4M3(u8): 4, 3, inf = false, nan = true, widen = widen_e4m3, narrow = Format::<4, 3, false, true>::narrow
}

minifloat! {
    /// A 4-bit float, 6 at most. NaN saturates.
    E2M1(u8): 2, 1, inf = false, nan = false, widen = widen_e2m1, narrow = Format::<2, 1, false, false>::narrow
}

/// OCP e8m0: `2^(n - 127)`, and 0xFF is NaN.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct E8M0(pub u8);

/// Rounds up, so a block scale never clips.
impl From<f32> for E8M0 {
    #[inline(always)]
    fn from(x: f32) -> Self {
        if x.is_nan() {
            return E8M0(0xFF);
        }
        let x = x.abs();
        let (exp, man) = ((x.to_bits() >> 23) as i32, x.to_bits() & 0x7F_FFFF);
        let up = exp + i32::from(man != 0);
        E8M0(up.clamp(0, 254) as u8)
    }
}

impl From<E8M0> for f32 {
    #[inline(always)]
    fn from(e: E8M0) -> f32 {
        let bits = match e.0 {
            0 => 0x0040_0000,
            0xFF => 0x7fc0_0000,
            n => u32::from(n) << 23,
        };
        f32::from_bits(bits)
    }
}
