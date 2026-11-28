//! Block forms: a scale, then codes it multiplies.

use crate::form::{Form, Read, form};
use crate::ops::{Bits, host_fns};
use tessera_graph::{DType, E2M1, E4M3, E8M0, F16};

/// A code: what one element stores, before its block's scale.
pub trait Code: Sync {
    const BITS: u32;
    /// A code's largest magnitude.
    const MAX: f32;
    /// The code at bit `low` of `w`.
    fn decode(w: u32, low: u32) -> f32;
}

/// How a code stores a value.
pub trait Encode: Code {
    fn encode(scaled: f32) -> u32;

    /// The scale for a block, from its values.
    fn scale(block: &[f32]) -> f32 {
        block.iter().fold(0.0f32, |m, x| m.max(x.abs())) / Self::MAX
    }
}

/// A scale: a narrow float before each block's codes.
pub trait Scale: Sync {
    const BYTES: usize;
    /// The type's largest scale.
    const MAX: f32;
    fn put(scale: f32, into: &mut [u8]);
    fn get(from: &[u8]) -> f32;
}

/// A signed byte.
pub struct I8;

/// Symmetric int4: a nibble 8 below its value.
pub struct Nibble;

/// Symmetric int2: 2-bit levels {-1.5, -0.5, +0.5, +1.5}.
pub struct Crumb;

/// Expands `$each` over the shared host and kernel block reads.
// Bodies use only kernel operations, inlined in kernel text.
#[macro_export]
macro_rules! block_fns {
    ($each:ident) => {
        $each! {
            /// An f16 scale: two bytes, low first.
            fn f16_scale(p: Buf<U8>) -> f32 {
                (p[0] as u32 | ((p[1] as u32) << 8)).f16(0)
            }

            /// An e4m3 scale: one byte.
            fn e4m3_scale(p: Buf<U8>) -> f32 {
                (p[0] as u32).e4m3(0)
            }

            /// An e8m0 scale: one byte, the f32 exponent.
            fn e8m0_scale(p: Buf<U8>) -> f32 {
                f32::from_bits((p[0] as u32) << 23)
            }

            /// An e4m3 byte.
            fn e4m3_field(w: u32, low: u32) -> f32 {
                w.e4m3(low)
            }

            /// An e2m1 nibble.
            fn e2m1_field(w: u32, low: u32) -> f32 {
                w.e2m1(low)
            }

            /// A signed byte.
            fn i8_field(w: u32, low: u32) -> f32 {
                w.signed(low, 8)
            }

            /// A nibble 8 below its value.
            fn nibble_field(w: u32, low: u32) -> f32 {
                w.bits(low, 4) as f32 - 8.0
            }

            /// A 2-bit crumb centered at 1.5.
            fn crumb_field(w: u32, low: u32) -> f32 {
                w.bits(low, 2) as f32 - 1.5
            }

            /// Code `j`'s byte, paired by halves under `halves`.
            fn code_byte(#[build] bits: u32, #[build] b: u64, #[build] halves: bool, j: u64) -> u64 {
                if bits == 8 {
                    j
                } else if bits == 2 {
                    j / 4
                } else if halves {
                    j % (b / 2)
                } else {
                    j / 2
                }
            }

            /// Code `j`'s shift within its byte.
            fn code_shift(#[build] bits: u32, #[build] b: u64, #[build] halves: bool, j: u64) -> u32 {
                if bits == 8 {
                    0
                } else if bits == 2 {
                    (2 * (j % 4)) as u32
                } else if halves {
                    (4 * (j / (b / 2))) as u32
                } else {
                    (4 * (j % 2)) as u32
                }
            }
        }
    };
}

crate::block_fns!(host_fns);

/// Expands `$each` over block scales, then codes.
// Row: type, bits type or width, largest value, read.
#[macro_export]
macro_rules! block_reads {
    ($each:ident) => {
        $each! {
            scales: (F16, u16, 65504.0, f16_scale), (E4M3, u8, 448.0, e4m3_scale),
                (E8M0, u8, 1.7014118e38, e8m0_scale);
            codes: (E2M1, 4, 6.0, e2m1_field), (E4M3, 8, 448.0, e4m3_field),
                (I8, 8, 127.0, i8_field), (Nibble, 4, 8.0, nibble_field),
                (Crumb, 2, 1.5, crumb_field);
        }
    };
}

macro_rules! define_reads {
    (scales: $(($s:ident, $repr:ty, $smax:literal, $get:ident)),*;
     codes: $(($c:ident, $bits:literal, $cmax:literal, $field:ident)),*;) => {
        $(
            impl Scale for $s {
                const BYTES: usize = size_of::<$repr>();
                const MAX: f32 = $smax;
                #[inline(always)]
                fn put(scale: f32, into: &mut [u8]) {
                    into[..Self::BYTES].copy_from_slice(&$s::from(scale).0.to_le_bytes());
                }
                #[inline(always)]
                fn get(from: &[u8]) -> f32 {
                    $get(from)
                }
            }
        )*
        $(
            impl Code for $c {
                const BITS: u32 = $bits;
                const MAX: f32 = $cmax;
                #[inline(always)]
                fn decode(w: u32, low: u32) -> f32 {
                    $field(w, low)
                }
            }
        )*
    };
}

crate::block_reads!(define_reads);

impl Encode for E2M1 {
    fn encode(scaled: f32) -> u32 {
        u32::from(E2M1::from(scaled).0)
    }
}

impl Encode for E4M3 {
    fn encode(scaled: f32) -> u32 {
        u32::from(E4M3::from(scaled).0)
    }
}

impl Encode for I8 {
    fn encode(scaled: f32) -> u32 {
        (scaled.round_ties_even().clamp(-128.0, 127.0) as i8) as u8 as u32
    }
}

impl Encode for Nibble {
    fn encode(scaled: f32) -> u32 {
        (scaled + 8.5).clamp(0.0, 15.0) as u32
    }

    /// Keeps the sign: the widest value lands on -8.
    fn scale(block: &[f32]) -> f32 {
        let (mut hi, mut lo) = (0.0f32, 0.0f32);
        for &x in block {
            hi = hi.max(x);
            lo = lo.min(x);
        }
        let neg = -lo;
        let top = if hi > neg {
            hi
        } else if neg > hi {
            lo
        } else if hi == 0.0 {
            0.0
        } else {
            // First occurrence breaks +hi and -hi ties.
            block.iter().copied().find(|x| x.abs() == hi).unwrap_or(hi)
        };
        top / -8.0
    }
}

impl Encode for Crumb {
    fn encode(scaled: f32) -> u32 {
        (scaled + 2.0).clamp(0.0, 3.0) as u32
    }
}

/// `B` codes after one scale, after an optional `HEAD` f32.
pub struct Block<S, C, const B: usize, const HALVES: bool, const HEAD: bool> {
    pub name: &'static str,
    _types: std::marker::PhantomData<(S, C)>,
}

impl<S: Scale, C: Encode, const B: usize, const HALVES: bool, const HEAD: bool>
    Block<S, C, B, HALVES, HEAD>
{
    pub const fn named(name: &'static str) -> Self {
        Self {
            name,
            _types: std::marker::PhantomData,
        }
    }

    const BLOCK_BYTES: usize = S::BYTES + B * C::BITS as usize / 8;
    const HEAD_BYTES: usize = if HEAD { 4 } else { 0 };

    /// Code `j`'s byte and shift.
    fn place(j: usize) -> (usize, u32) {
        let (b, j) = (B as u64, j as u64);
        let byte = code_byte(C::BITS, b, HALVES, j) as usize;
        (byte, code_shift(C::BITS, b, HALVES, j))
    }

    fn head(bytes: &[u8]) -> f32 {
        match HEAD {
            true => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            false => 1.0,
        }
    }

    /// The head: the widest block scale lands on `S::MAX`.
    fn tensor_scale(data: &[f32]) -> f32 {
        let widest = data
            .chunks(B)
            .map(C::scale)
            .fold(0.0f32, |m, s| m.max(s.abs()));
        match HEAD && widest > 0.0 {
            true => widest / S::MAX,
            false => 1.0,
        }
    }
    /// Packs one full block using its bit width and layout.
    #[inline(always)]
    fn pack_full(block: &[f32], codes: &mut [u8], mut enc: impl FnMut(f32) -> u32) {
        if C::BITS == 8 {
            for (dst, &x) in codes.iter_mut().zip(block) {
                *dst = enc(x) as u8;
            }
        } else if C::BITS == 4 && HALVES {
            let (lo, hi) = block.split_at(B / 2);
            for ((dst, &a), &b) in codes.iter_mut().zip(lo).zip(hi) {
                *dst = (enc(a) | (enc(b) << 4)) as u8;
            }
        } else if C::BITS == 4 {
            for (dst, pair) in codes.iter_mut().zip(block.as_chunks::<2>().0) {
                *dst = (enc(pair[0]) | (enc(pair[1]) << 4)) as u8;
            }
        } else if C::BITS == 2 {
            for (dst, quad) in codes.iter_mut().zip(block.as_chunks::<4>().0) {
                let b =
                    enc(quad[0]) | (enc(quad[1]) << 2) | (enc(quad[2]) << 4) | (enc(quad[3]) << 6);
                *dst = b as u8;
            }
        } else {
            for (j, &x) in block.iter().enumerate() {
                let (byte, shift) = Self::place(j);
                codes[byte] |= (enc(x) << shift) as u8;
            }
        }
    }

    /// Sums one full block against `xs`.
    #[inline(always)]
    fn dot_full(xs: &[f32], codes: &[u8]) -> f32 {
        if C::BITS == 8 {
            let mut acc = [0.0f32; 4];
            let (xc, xr) = xs.as_chunks::<4>();
            let (cc, cr) = codes.as_chunks::<4>();
            for (xv, cv) in xc.iter().zip(cc) {
                for i in 0..4 {
                    acc[i] += xv[i] * C::decode(u32::from(cv[i]), 0);
                }
            }
            let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for (&xj, &b) in xr.iter().zip(cr) {
                sum += xj * C::decode(u32::from(b), 0);
            }
            sum
        } else if C::BITS == 4 && HALVES {
            let (lo, hi) = xs.split_at(B / 2);
            let mut acc = [0.0f32; 4];
            let (lc, lr) = lo.as_chunks::<4>();
            let (hc, hr) = hi.as_chunks::<4>();
            let (cc, cr) = codes.as_chunks::<4>();
            for ((lv, hv), cv) in lc.iter().zip(hc).zip(cc) {
                for i in 0..4 {
                    let w = u32::from(cv[i]);
                    acc[i] += lv[i] * C::decode(w, 0) + hv[i] * C::decode(w, 4);
                }
            }
            let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for ((&a, &b), &byte) in lr.iter().zip(hr).zip(cr) {
                let w = u32::from(byte);
                sum += a * C::decode(w, 0) + b * C::decode(w, 4);
            }
            sum
        } else if C::BITS == 4 {
            let mut acc = [0.0f32; 4];
            let (xc, xr) = xs.as_chunks::<4>();
            let (cc, cr) = codes.as_chunks::<2>();
            for (xv, cv) in xc.iter().zip(cc) {
                let (w0, w1) = (u32::from(cv[0]), u32::from(cv[1]));
                acc[0] += xv[0] * C::decode(w0, 0);
                acc[1] += xv[1] * C::decode(w0, 4);
                acc[2] += xv[2] * C::decode(w1, 0);
                acc[3] += xv[3] * C::decode(w1, 4);
            }
            let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for (pair, &byte) in xr.as_chunks::<2>().0.iter().zip(cr) {
                let w = u32::from(byte);
                sum += pair[0] * C::decode(w, 0) + pair[1] * C::decode(w, 4);
            }
            sum
        } else if C::BITS == 2 {
            let mut acc = [0.0f32; 4];
            for (quad, &byte) in xs.as_chunks::<4>().0.iter().zip(codes) {
                let w = u32::from(byte);
                acc[0] += quad[0] * C::decode(w, 0);
                acc[1] += quad[1] * C::decode(w, 2);
                acc[2] += quad[2] * C::decode(w, 4);
                acc[3] += quad[3] * C::decode(w, 6);
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        } else {
            let mut part = 0.0;
            for (j, &xj) in xs.iter().enumerate() {
                let (byte, shift) = Self::place(j);
                part += xj * C::decode(u32::from(codes[byte]), shift);
            }
            part
        }
    }

    /// True when `s` has an exact power-of-two reciprocal.
    #[inline(always)]
    fn exact_pow2(s: f32) -> bool {
        let bits = s.to_bits();
        let exp = (bits >> 23) & 0xff;
        (bits & 0x7f_ffff) == 0 && (1..=253).contains(&exp)
    }
}

impl<S: Scale, C: Encode, const B: usize, const HALVES: bool, const HEAD: bool> Form
    for Block<S, C, B, HALVES, HEAD>
{
    fn name(&self) -> &'static str {
        self.name
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn read(&self) -> Read {
        Read::Decode
    }
    fn narrow(&self) -> bool {
        true
    }
    fn block(&self) -> usize {
        B
    }
    fn bytes(&self, elements: usize) -> usize {
        Self::HEAD_BYTES + elements.div_ceil(B) * Self::BLOCK_BYTES
    }
    fn offset(&self, index: usize) -> usize {
        match index {
            0 => 0,
            _ => Self::HEAD_BYTES + index / B * Self::BLOCK_BYTES,
        }
    }
    fn elements(&self, offset: usize) -> usize {
        match offset {
            0 => 0,
            _ => (offset - Self::HEAD_BYTES) / Self::BLOCK_BYTES * B,
        }
    }
    fn element_bytes(&self) -> f64 {
        Self::BLOCK_BYTES as f64 / B as f64
    }

    /// Each block's codes against `x`, under the block's scale.
    fn dot(&self, x: &[f32], bytes: &[u8], at: usize) -> f32 {
        assert!(at.is_multiple_of(B), "a dot starts on a block");
        let head = Self::head(bytes);
        let first = Self::HEAD_BYTES + (at / B) * Self::BLOCK_BYTES;
        let blocks = bytes[first..].chunks(Self::BLOCK_BYTES);
        let mut sum = 0.0;
        for (xs, slot) in x.chunks(B).zip(blocks) {
            let codes = &slot[S::BYTES..];
            let part = if xs.len() == B {
                Self::dot_full(xs, codes)
            } else {
                let mut part = 0.0;
                for (j, xj) in xs.iter().enumerate() {
                    let (byte, shift) = Self::place(j);
                    part += xj * C::decode(u32::from(codes[byte]), shift);
                }
                part
            };
            sum += S::get(slot) * part;
        }
        head * sum
    }

    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        let (block, j) = (index / B, index % B);
        let at = Self::HEAD_BYTES + block * Self::BLOCK_BYTES;
        let (byte, shift) = Self::place(j);
        let code = C::decode(u32::from(bytes[at + S::BYTES + byte]), shift);
        Self::head(bytes) * S::get(&bytes[at..]) * code
    }

    fn pack(&self, data: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes(data.len())];
        let tensor = Self::tensor_scale(data);
        if HEAD {
            out[..4].copy_from_slice(&tensor.to_le_bytes());
        }
        let body = &mut out[Self::HEAD_BYTES..];
        for (block, slot) in data.chunks(B).zip(body.chunks_mut(Self::BLOCK_BYTES)) {
            let scale = C::scale(block) / tensor;
            S::put(scale, slot);
            // Codes see the scale as it was stored.
            let stored = S::get(slot) * tensor;
            let codes = &mut slot[S::BYTES..];
            if block.len() == B {
                if stored == 0.0 {
                    let zero = C::encode(0.0);
                    if zero != 0 {
                        Self::pack_full(block, codes, |_| zero);
                    }
                } else if Self::exact_pow2(stored) {
                    let inv = 1.0 / stored;
                    Self::pack_full(block, codes, |x| C::encode(x * inv));
                } else {
                    Self::pack_full(block, codes, |x| C::encode(x / stored));
                }
            } else {
                for (j, x) in block.iter().enumerate() {
                    let scaled = if stored == 0.0 { 0.0 } else { x / stored };
                    let (byte, shift) = Self::place(j);
                    codes[byte] |= (C::encode(scaled) << shift) as u8;
                }
            }
        }
        out
    }
}

/// All block forms for `$each`, each with its `form_kind`.
#[macro_export]
macro_rules! block_forms {
    ($each:ident) => {
        $each! {
            (Q8_0, "q8_0", 1, F16, I8, 32, true, false, "GGUF Q8_0: an f16 scale, then 32 signed bytes."),
            (Q4_0, "q4_0", 2, F16, Nibble, 32, true, false, "GGUF Q4_0: an f16 scale, then 32 nibbles."),
            (NVFP4, "nvfp4", 3, E4M3, E2M1, 16, false, true, "nvfp4: an f32 head, then e4m3 scales over e2m1 codes."),
            (MXFP8, "mxfp8", 4, E8M0, E4M3, 32, false, false, "OCP mxfp8: an e8m0 scale, then 32 e4m3 codes."),
            (MXFP4, "mxfp4", 5, E8M0, E2M1, 32, false, false, "OCP mxfp4: an e8m0 scale, then 32 e2m1 codes."),
            (INT4_G32, "int4_g32", 6, F16, Nibble, 32, false, false, "Group-wise int4: an f16 scale, then 32 nibbles."),
            (INT4_G128, "int4_g128", 7, F16, Nibble, 128, false, false, "`INT4_G32` with groups of 128."),
            (INT2_G32, "int2_g32", 17, F16, Crumb, 32, false, false, "Group-wise int2: an f16 scale, then 32 2-bit codes."),
            (INT2_G128, "int2_g128", 18, F16, Crumb, 128, false, false, "`INT2_G32` with groups of 128."),
        }
    };
}

macro_rules! define_blocks {
    ($(($name:ident, $text:literal, $kind:literal, $scale:ty, $code:ty, $b:literal, $halves:literal, $head:literal, $doc:literal),)*) => {$(
        #[doc = $doc]
        pub const $name: Block<$scale, $code, $b, $halves, $head> = Block::named($text);
        form!($name);
    )*};
}

crate::block_forms!(define_blocks);
