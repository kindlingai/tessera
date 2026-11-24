//! Block forms: a scale, then codes it multiplies.

use crate::form::{Form, Read, form};
use tessera_graph::{DType, E2M1, E4M3, E8M0, F16};

/// A code: what one element stores, before its block's scale.
pub trait Code: Sync {
    const BITS: u32;
    /// A code's largest magnitude.
    const MAX: f32;
    fn encode(scaled: f32) -> u32;
    fn decode(code: u32) -> f32;

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

macro_rules! scale {
    ($t:ident($repr:ty), $bytes:literal, $max:literal) => {
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
        }
    };
}

scale!(F16(u16), 2, 65504.0);
scale!(E4M3(u8), 1, 448.0);
scale!(E8M0(u8), 1, 1.7014118e38);

/// Every e2m1 value, by code.
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

impl Code for E2M1 {
    const BITS: u32 = 4;
    const MAX: f32 = 6.0;
    fn encode(scaled: f32) -> u32 {
        u32::from(E2M1::from(scaled).0)
    }
    fn decode(code: u32) -> f32 {
        E2M1_VALUES[code as usize & 15]
    }
}

impl Code for E4M3 {
    const BITS: u32 = 8;
    const MAX: f32 = 448.0;
    fn encode(scaled: f32) -> u32 {
        u32::from(E4M3::from(scaled).0)
    }
    fn decode(code: u32) -> f32 {
        E4M3(code as u8).into()
    }
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
}

/// Symmetric int4: a nibble 8 below its value.
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

    /// Keeps the sign: the widest value lands on -8.
    fn scale(block: &[f32]) -> f32 {
        let top = block
            .iter()
            .fold(0.0f32, |m, x| if x.abs() > m.abs() { *x } else { m });
        top / -8.0
    }
}

/// Symmetric int2: 2-bit levels {-1.5, -0.5, +0.5, +1.5}.
pub struct Crumb;

impl Code for Crumb {
    const BITS: u32 = 2;
    const MAX: f32 = 1.5;
    fn encode(scaled: f32) -> u32 {
        (scaled + 2.0).clamp(0.0, 3.0) as u32
    }
    fn decode(code: u32) -> f32 {
        code as f32 - 1.5
    }
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
            (2, _) => (j / 4, 2 * (j % 4) as u32),
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
    fn read(&self) -> Read {
        Read::Decode
    }
    fn bits(&self) -> u32 {
        C::BITS
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
        let (head, mask) = (Self::head(bytes), (1u32 << C::BITS) - 1);
        let mut sum = 0.0;
        for (b, xs) in x.chunks(B).enumerate() {
            let start = Self::HEAD_BYTES + (at / B + b) * Self::BLOCK_BYTES;
            let codes = &bytes[start + S::BYTES..start + Self::BLOCK_BYTES];
            let mut part = 0.0;
            for (j, xj) in xs.iter().enumerate() {
                let (byte, shift) = Self::place(j);
                part += xj * C::decode((u32::from(codes[byte]) >> shift) & mask);
            }
            sum += S::get(&bytes[start..]) * part;
        }
        head * sum
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

/// Expands `$each` over all block storage forms.
#[macro_export]
macro_rules! block_forms {
    ($each:ident) => {
        $each! {
            (Q8_0, "q8_0", F16, I8, 32, true, false, "GGUF Q8_0: an f16 scale, then 32 signed bytes."),
            (Q4_0, "q4_0", F16, Nibble, 32, true, false, "GGUF Q4_0: an f16 scale, then 32 nibbles."),
            (NVFP4, "nvfp4", E4M3, E2M1, 16, false, true, "nvfp4: an f32 head, then e4m3 scales over e2m1 codes."),
            (MXFP8, "mxfp8", E8M0, E4M3, 32, false, false, "OCP mxfp8: an e8m0 scale, then 32 e4m3 codes."),
            (MXFP4, "mxfp4", E8M0, E2M1, 32, false, false, "OCP mxfp4: an e8m0 scale, then 32 e2m1 codes."),
            (INT4_G32, "int4_g32", F16, Nibble, 32, false, false, "Group-wise int4: an f16 scale, then 32 nibbles."),
            (INT4_G128, "int4_g128", F16, Nibble, 128, false, false, "`INT4_G32` with groups of 128."),
            (INT2_G32, "int2_g32", F16, Crumb, 32, false, false, "Group-wise int2: an f16 scale, then 32 2-bit codes."),
            (INT2_G128, "int2_g128", F16, Crumb, 128, false, false, "`INT2_G32` with groups of 128."),
        }
    };
}

macro_rules! define_blocks {
    ($(($name:ident, $text:literal, $scale:ty, $code:ty, $b:literal, $halves:literal, $head:literal, $doc:literal),)*) => {$(
        #[doc = $doc]
        pub const $name: Block<$scale, $code, $b, $halves, $head> = Block::named($text);
        form!($name);
    )*};
}

crate::block_forms!(define_blocks);
