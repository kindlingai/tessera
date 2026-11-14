//! Block forms: a scale, then codes it multiplies.

use crate::form::Form;
use graph::{DType, E2M1, E4M3, F16};

/// A code: what one element stores, before its block's scale.
pub trait Code: Sync {
    const BITS: u32;
    /// The largest magnitude a code holds.
    const MAX: f32;
    fn encode(scaled: f32) -> u32;
    fn decode(code: u32) -> f32;
    /// Kernel text: code `c` as a float.
    fn text(c: &str) -> String;

    /// The kernel function `text` calls, if any.
    fn header() -> Option<String> {
        None
    }

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
    /// Kernel text: the scale whose first byte `p` points at.
    fn text(p: &str) -> String;
    /// The kernel function `text` calls.
    fn header() -> String;
}

/// The kernel function decoding `e` and `m` bits.
pub(crate) fn minifloat_name(e: u32, m: u32) -> String {
    format!("f_mini{e}_{m}")
}

/// The function's text: a float from a word.
pub(crate) fn minifloat(e: u32, m: u32) -> String {
    let bias = (1 << (e - 1)) - 1;
    let step = 2f32.powi(1 - bias - m as i32);
    let normal = format!("f_bits(((e + {}u) << 23) | (mn << {}))", 127 - bias, 23 - m);
    format!(
        "f_fn float {}(u32 x) {{\n    u32 e = (x >> {m}) & {}u;\n    u32 mn = x & {}u;\n    float v = e == 0u ? (float)mn * {step:?}f : {normal};\n    return ((x >> {}) & 1u) ? -v : v;\n}}\n",
        minifloat_name(e, m),
        (1 << e) - 1,
        (1 << m) - 1,
        e + m,
    )
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
            fn text(p: &str) -> String {
                let word = (0..$bytes)
                    .map(|i| format!("((u32)({p})[{i}] << {})", 8 * i))
                    .collect::<Vec<_>>()
                    .join(" | ");
                format!("{}({word})", minifloat_name($t::EXPONENT, $t::MANTISSA))
            }
            fn header() -> String {
                minifloat($t::EXPONENT, $t::MANTISSA)
            }
        }
    };
}

scale!(F16(u16), 2, 65504.0);
scale!(E4M3(u8), 1, 448.0);

impl Code for E2M1 {
    const BITS: u32 = 4;
    const MAX: f32 = 6.0;
    fn encode(scaled: f32) -> u32 {
        u32::from(E2M1::from(scaled).0)
    }
    fn decode(code: u32) -> f32 {
        E2M1(code as u8).into()
    }
    fn text(c: &str) -> String {
        format!("{}({c})", minifloat_name(2, 1))
    }
    fn header() -> Option<String> {
        Some(minifloat(2, 1))
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
    fn text(c: &str) -> String {
        format!("(float)(((int)({c}) ^ 128) - 128)")
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
    fn text(c: &str) -> String {
        format!("((float)({c}) - 8.0f)")
    }

    /// Keeps the sign: the widest value lands on -8.
    fn scale(block: &[f32]) -> f32 {
        let top = block
            .iter()
            .fold(0.0f32, |m, x| if x.abs() > m.abs() { *x } else { m });
        top / -8.0
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

    /// An inline function decoding element `i` of `p`.
    fn header(&self) -> Vec<String> {
        let (head, scale, code) = (Self::HEAD_BYTES, S::BYTES, C::BITS);
        let at = format!("p + {head} + (i / {B}) * {}", Self::BLOCK_BYTES);
        let (byte, shift) = match (code, HALVES) {
            (8, _) => ("j".to_string(), "0".to_string()),
            (_, true) => (format!("j % {}", B / 2), format!("4 * (j / {})", B / 2)),
            _ => ("j / 2".into(), "4 * (j % 2)".into()),
        };
        let mask = (1u32 << code) - 1;
        let tensor = match HEAD {
            true => "f_bits((u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24))",
            false => "1.0f",
        };
        let own = format!(
            "f_fn float form_{}(f_const(u8) p, u64 i) {{\n    u64 j = i % {B};\n    f_const(u8) b = {at};\n    u32 c = ((u32)b[{scale} + {byte}] >> ({shift})) & {mask}u;\n    return {tensor} * {} * {};\n}}\n",
            self.name,
            S::text("b"),
            C::text("c"),
        );
        let mut parts = vec![S::header()];
        parts.extend(C::header());
        parts.push(own);
        parts
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
