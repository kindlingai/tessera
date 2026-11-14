//! Block forms: a scale, then codes it multiplies.

use crate as codegen;
use crate::form::Form;
use crate::inline::{Inline, Pass};
use crate::ir::{Expr, Helper};
use crate::prelude::{Buf, U8, device, helper, kernel};
use graph::{DType, E2M1, E4M3, F16};

/// A code: what one element stores, before its block's scale.
pub trait Code: Sync {
    const BITS: u32;
    /// The largest magnitude a code holds.
    const MAX: f32;
    fn encode(scaled: f32) -> u32;
    fn decode(code: u32) -> f32;

    /// Code `c` as a float, in a kernel.
    fn float(_: u32) -> f32 {
        device()
    }

    /// `float` as a tree, for the block's helper.
    fn float_inline(c: impl Pass<Expr>) -> Inline;

    /// The scale for a block, from its values.
    fn scale(block: &[f32]) -> f32 {
        block.iter().fold(0.0f32, |m, x| m.max(x.abs())) / Self::MAX
    }
}

/// A scale: a narrow float before each block's codes.
pub trait Scale: Sync {
    const BYTES: usize;
    /// The largest scale the type holds.
    const MAX: f32;
    fn put(scale: f32, into: &mut [u8]);
    fn get(from: &[u8]) -> f32;

    /// The scale starting at byte `p`, in a kernel.
    fn load(_: Buf<U8>) -> f32 {
        device()
    }

    /// `load` as a tree, for the block's helper.
    fn load_inline(p: impl Pass<Expr>) -> Inline;
}

/// A float from `x`: `E` exponent and `M` mantissa bits.
#[helper("f_mini{E}_{M}")]
pub(crate) fn minifloat<const E: u32, const M: u32>(x: u32) -> f32 {
    let ex = (x >> M) & ((1u32 << E) - 1);
    let mn = x & ((1u32 << M) - 1);
    // A subnormal counts units of the least normal.
    let v = if ex == 0 {
        mn as f32 * f32::from_bits((129 - (1u32 << (E - 1)) - M) << 23)
    } else {
        f32::from_bits(((ex + (128 - (1u32 << (E - 1)))) << 23) | (mn << (23 - M)))
    };
    if ((x >> (E + M)) & 1) != 0 { -v } else { v }
}

/// An f16 scale: two bytes, low first.
#[kernel]
fn f16_scale(p: Buf<U8>) -> f32 {
    minifloat::<5, 10>(p[0] as u32 | ((p[1] as u32) << 8))
}

/// An e4m3 scale: one byte.
#[kernel]
fn e4m3_scale(p: Buf<U8>) -> f32 {
    minifloat::<4, 3>(p[0] as u32)
}

macro_rules! scale {
    ($t:ident($repr:ty), $bytes:literal, $max:literal, $load:ident) => {
        impl Scale for $t {
            const BYTES: usize = $bytes;
            const MAX: f32 = $max;
            fn put(scale: f32, into: &mut [u8]) {
                into[..$bytes].copy_from_slice(&$t::from(scale).0.to_le_bytes());
            }
            fn get(from: &[u8]) -> f32 {
                let mut raw = [0u8; $bytes];
                raw.copy_from_slice(&from[..$bytes]);
                $t(<$repr>::from_le_bytes(raw)).into()
            }
            fn load_inline(p: impl Pass<Expr>) -> Inline {
                $load(p)
            }
        }
    };
}

scale!(F16(u16), 2, 65504.0, f16_scale_inline);
scale!(E4M3(u8), 1, 448.0, e4m3_scale_inline);

impl Code for E2M1 {
    const BITS: u32 = 4;
    const MAX: f32 = 6.0;
    fn encode(scaled: f32) -> u32 {
        u32::from(E2M1::from(scaled).0)
    }
    fn decode(code: u32) -> f32 {
        E2M1(code as u8).into()
    }
    fn float_inline(c: impl Pass<Expr>) -> Inline {
        minifloat_inline::<2, 1>(c)
    }
}

/// A signed byte, from its two's complement.
#[kernel]
fn i8_code(c: u32) -> f32 {
    ((c ^ 128) as f32) - 128.0
}

/// A nibble 8 below its value.
#[kernel]
fn nibble_code(c: u32) -> f32 {
    c as f32 - 8.0
}

/// A signed byte.
pub struct I8;

impl Code for I8 {
    const BITS: u32 = 8;
    const MAX: f32 = 127.0;
    fn encode(scaled: f32) -> u32 {
        (scaled.round_ties_even().clamp(-128.0, 127.0) as i8) as u8 as u32
    }
    fn decode(code: u32) -> f32 {
        f32::from(code as u8 as i8)
    }
    fn float_inline(c: impl Pass<Expr>) -> Inline {
        i8_code_inline(c)
    }
}

/// A nibble 8 below its value, as in GGUF Q4_0.
pub struct Nibble;

impl Code for Nibble {
    const BITS: u32 = 4;
    const MAX: f32 = 8.0;
    fn encode(scaled: f32) -> u32 {
        (scaled + 8.5).clamp(0.0, 15.0) as u32
    }
    fn decode(code: u32) -> f32 {
        code as f32 - 8.0
    }
    fn float_inline(c: impl Pass<Expr>) -> Inline {
        nibble_code_inline(c)
    }

    /// Keeps the sign: the widest value lands on -8.
    fn scale(block: &[f32]) -> f32 {
        let top = block
            .iter()
            .fold(0.0f32, |m, x| if x.abs() > m.abs() { *x } else { m });
        top / -8.0
    }
}

/// Element `i` of `p`: head, block scale and code.
#[helper("form_{name}")]
fn element<S: Scale, C: Code, const B: usize, const HALVES: bool, const HEAD: bool>(
    #[build] name: &'static str,
    p: Buf<U8>,
    i: u64,
) -> f32 {
    let j = i % B as u64;
    let b = p + (HEAD as u64 * 4 + (i / B as u64) * (S::BYTES + B * C::BITS as usize / 8) as u64);
    // Code `j`'s byte and shift, paired by halves under `HALVES`.
    let byte = if C::BITS == 8 {
        j
    } else if HALVES {
        j % (B / 2) as u64
    } else {
        j / 2
    };
    let shift = if C::BITS == 8 {
        0
    } else if HALVES {
        (4 * (j / (B / 2) as u64)) as u32
    } else {
        (4 * (j % 2)) as u32
    };
    let c = ((b[S::BYTES as u64 + byte] as u32) >> shift) & ((1u32 << C::BITS) - 1);
    let head = if HEAD {
        f32::from_bits(
            p[0] as u32 | ((p[1] as u32) << 8) | ((p[2] as u32) << 16) | ((p[3] as u32) << 24),
        )
    } else {
        1.0
    };
    head * S::load(b) * C::float(c)
}

/// `B` codes after one scale, after an optional `HEAD` f32.
pub struct Block<S, C, const B: usize, const HALVES: bool, const HEAD: bool> {
    pub name: &'static str,
    _types: std::marker::PhantomData<(S, C)>,
}

impl<S: Scale, C: Code, const B: usize, const HALVES: bool, const HEAD: bool>
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

    /// Code `j`'s byte and shift, paired by halves under `HALVES`.
    fn place(j: usize) -> (usize, u32) {
        match (C::BITS, HALVES) {
            (8, _) => (j, 0),
            (_, true) => (j % (B / 2), 4 * (j / (B / 2)) as u32),
            _ => (j / 2, 4 * (j % 2) as u32),
        }
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
}

impl<S: Scale, C: Code, const B: usize, const HALVES: bool, const HEAD: bool> Form
    for Block<S, C, B, HALVES, HEAD>
{
    fn name(&self) -> &'static str {
        self.name
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn bits(&self) -> u32 {
        C::BITS
    }
    fn ty(&self) -> &'static str {
        "u8"
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

    fn load(&self, name: &str, index: &str) -> String {
        format!("form_{}({name}, {index})", self.name)
    }

    fn header(&self) -> Vec<Helper> {
        vec![element_ir::<S, C, B, HALVES, HEAD>(self.name)]
    }

    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        let (block, j) = (index / B, index % B);
        let at = Self::HEAD_BYTES + block * Self::BLOCK_BYTES;
        let (byte, shift) = Self::place(j);
        let code = (u32::from(bytes[at + S::BYTES + byte]) >> shift) & ((1 << C::BITS) - 1);
        Self::head(bytes) * S::get(&bytes[at..]) * C::decode(code)
    }

    fn pack(&self, data: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes(data.len())];
        let tensor = Self::tensor_scale(data);
        if HEAD {
            out[..4].copy_from_slice(&tensor.to_le_bytes());
        }
        for (b, block) in data.chunks(B).enumerate() {
            let at = Self::HEAD_BYTES + b * Self::BLOCK_BYTES;
            let scale = C::scale(block) / tensor;
            S::put(scale, &mut out[at..]);
            // Codes see the scale as it was stored.
            let stored = S::get(&out[at..]) * tensor;
            for (j, x) in block.iter().enumerate() {
                let scaled = if stored == 0.0 { 0.0 } else { x / stored };
                let (byte, shift) = Self::place(j);
                out[at + S::BYTES + byte] |= (C::encode(scaled) << shift) as u8;
            }
        }
        out
    }
}

/// A block form as a registered constant.
macro_rules! block {
    ($(#[$doc:meta])* $name:ident = $text:literal, $scale:ty, $code:ty, $b:literal, halves = $halves:literal, head = $head:literal) => {
        $(#[$doc])*
        pub const $name: Block<$scale, $code, $b, $halves, $head> = Block::named($text);
        crate::form!($name);
    };
}

block! {
    /// GGUF Q8_0: an f16 scale, then 32 signed bytes.
    Q8_0 = "q8_0", F16, I8, 32, halves = true, head = false
}

block! {
    /// GGUF Q4_0: an f16 scale, then 32 nibbles.
    Q4_0 = "q4_0", F16, Nibble, 32, halves = true, head = false
}

block! {
    /// nvfp4: an f32 head, then e4m3 scales over e2m1 codes.
    NVFP4 = "nvfp4", E4M3, E2M1, 16, halves = false, head = true
}
