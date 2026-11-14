//! The exlr8 trellis form: K-bit codes in tiles.

use crate::block::{minifloat, minifloat_name};
use crate::form::{Form, Layout};
use graph::{DType, E4M3, F16};

/// The codebook multiplier.
const MCG: u32 = 0xCBAC_1FED;

/// Header bytes: `in`, `out`, the sharded side, a pad.
const HEAD: usize = 16;

/// Record field 4: entries hold 128 input channels.
pub const INPUT_SIDE: u32 = 2;

/// Channels one entry holds on its sharded side.
const ENTRY: usize = 128;

/// A matrix of `K`-bit trellis codes, stored `[out][in]`.
pub struct Trellis<const K: u32> {
    pub name: &'static str,
}

/// The matrix's shape and side, from the header.
struct Shape {
    inp: usize,
    out: usize,
    side: u32,
}

impl Shape {
    fn read(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        Self {
            inp: word(0) as usize,
            out: word(1) as usize,
            side: word(2),
        }
    }

    fn write(&self) -> [u8; HEAD] {
        let mut head = [0u8; HEAD];
        for (i, v) in [self.inp as u32, self.out as u32, self.side]
            .iter()
            .enumerate()
        {
            head[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        head
    }

    /// The tile holding `T[row][col]`, counted in the entries.
    fn tile(&self, row: usize, col: usize) -> usize {
        // An entry: 128 channels by all of the other side.
        match self.side {
            INPUT_SIDE => {
                let (entry, r) = (row / ENTRY, row % ENTRY / 16);
                entry * (self.out / 2) + 64 * (col / ENTRY) + 8 * r + col % ENTRY / 16
            }
            _ => {
                let (entry, c) = (col / ENTRY, col % ENTRY / 16);
                entry * (self.inp / 2) + 8 * (row / 16) + c
            }
        }
    }
}

/// Stream position of tile element `(r, c)`.
fn position(r: usize, c: usize) -> usize {
    let t = 4 * (c % 8) + (r % 8) / 2;
    let s = 4 * usize::from(c >= 8) + 2 * usize::from(r >= 8) + r % 2;
    8 * t + s
}

/// Bit `b` of a tile: word pairs swapped, high first.
fn bit(tile: &[u8], b: usize) -> u32 {
    let word = (b / 16) ^ 1;
    let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
    u32::from(w >> (15 - b % 16)) & 1
}

/// A state's weight: the mcg hash, then E4M3 rounding.
pub fn codebook(state: u32) -> f32 {
    let x = (state & 0xFFFF).wrapping_mul(MCG);
    let x = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let low = f32::from(F16((x & 0xFFFF) as u16));
    let high = f32::from(F16((x >> 16) as u16));
    let sum = f32::from(F16::from(low + high));
    f32::from(E4M3::from(sum.clamp(-448.0, 448.0)))
}

impl<const K: u32> Trellis<K> {
    pub const fn named(name: &'static str) -> Self {
        Self { name }
    }

    /// Bytes one tile of 256 codes takes.
    const TILE: usize = 32 * K as usize;

    /// The code at stream position `j` of a tile.
    fn code(tile: &[u8], j: usize) -> u32 {
        (0..K as usize).fold(0, |v, b| (v << 1) | bit(tile, j * K as usize + b))
    }

    /// Element `index` of a `[out][in]` matrix in `bytes`.
    fn decode(bytes: &[u8], index: usize) -> f32 {
        let shape = Shape::read(bytes);
        let (col, row) = (index / shape.inp, index % shape.inp);
        let at = HEAD + shape.tile(row, col) * Self::TILE;
        let tile = &bytes[at..at + Self::TILE];
        let j = position(row % 16, col % 16);
        // A state: the 16 bits ending at its value, wrapping.
        let mut state = 0;
        for t in 0..16usize.div_ceil(K as usize) {
            state |= Self::code(tile, (j + 256 - t) % 256) << (t * K as usize);
        }
        codebook(state)
    }

    /// Kernel text of `decode`, as one function.
    fn text(&self) -> String {
        let (k, tile, head) = (K, Self::TILE, HEAD);
        let steps = 16u32.div_ceil(K);
        let f16 = minifloat_name(5, 10);
        format!(
            "f_fn u32 form_{name}_bit(f_const(u8) t, u32 b) {{\n    u32 w = (b >> 4) ^ 1u;\n    u32 word = (u32)t[2 * w] | ((u32)t[2 * w + 1] << 8);\n    return (word >> (15 - (b & 15))) & 1u;\n}}\n\
f_fn u32 form_{name}_code(f_const(u8) t, u32 j) {{\n    u32 v = 0;\n    for (u32 b = 0; b < {k}u; b++) {{\n        v = (v << 1) | form_{name}_bit(t, j * {k}u + b);\n    }}\n    return v;\n}}\n\
f_fn float form_{name}(f_const(u8) p, u64 i) {{\n    u32 inp = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);\n    u32 out = (u32)p[4] | ((u32)p[5] << 8) | ((u32)p[6] << 16) | ((u32)p[7] << 24);\n    u32 side = (u32)p[8];\n    u32 col = (u32)(i / inp);\n    u32 row = (u32)(i % inp);\n    u32 tile;\n    if (side == {INPUT_SIDE}u) {{\n        tile = (row / 128u) * (out / 2u) + 64u * (col / 128u) + 8u * ((row % 128u) / 16u) + (col % 128u) / 16u;\n    }} else {{\n        tile = (col / 128u) * (inp / 2u) + 8u * (row / 16u) + (col % 128u) / 16u;\n    }}\n    f_const(u8) t = p + {head} + (u64)tile * {tile}u;\n    u32 r = row % 16u;\n    u32 c = col % 16u;\n    u32 j = 8u * (4u * (c % 8u) + (r % 8u) / 2u) + 4u * (c >= 8u) + 2u * (r >= 8u) + (r % 2u);\n    u32 state = 0;\n    for (u32 s = 0; s < {steps}u; s++) {{\n        state |= form_{name}_code(t, (j + 256u - s) % 256u) << (s * {k}u);\n    }}\n    u32 x = (state & 0xFFFFu) * {MCG}u;\n    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;\n    float v = f_round_f16({f16}(x & 0xFFFFu) + {f16}(x >> 16));\n    return f_round_e4m3(v);\n}}\n",
            name = self.name,
        )
    }
}

/// Kernel text: round to nearest even in a float format.
fn rounding(name: &str, e: u32, m: u32, top: f32) -> String {
    let bias = (1i32 << (e - 1)) - 1;
    let least = 2f32.powi(1 - bias);
    let steps = 2f32.powi(bias - 1 + m as i32);
    let drop = 23 - m;
    format!(
        "f_fn float {name}(float x) {{\n    float a = x < 0.0f ? -x : x;\n    float sign = x < 0.0f ? -1.0f : 1.0f;\n    if (a < {least:?}f) {{\n        float r = (a * {steps:?}f + 8388608.0f) - 8388608.0f;\n        return sign * (r / {steps:?}f);\n    }}\n    u32 bits = f_asbits(a);\n    u32 nudge = ({half}u) + ((bits >> {drop}u) & 1u);\n    float r = f_bits((bits + nudge) & ~{low}u);\n    return sign * (r > {top:?}f ? {top:?}f : r);\n}}\n",
        half = (1u32 << (drop - 1)) - 1,
        low = (1u32 << drop) - 1,
    )
}

impl<const K: u32> Form for Trellis<K> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn bits(&self) -> u32 {
        K
    }
    fn ty(&self) -> &'static str {
        "u8"
    }
    fn narrow(&self) -> bool {
        true
    }
    fn layout(&self) -> Layout {
        Layout::Tiles { rows: 16, cols: 16 }
    }
    /// A tile, so no float row copies in as is.
    fn block(&self) -> usize {
        256
    }
    fn align(&self) -> usize {
        ENTRY
    }
    fn bytes(&self, elements: usize) -> usize {
        HEAD + elements * K as usize / 8
    }
    fn offset(&self, index: usize) -> usize {
        match index {
            0 => 0,
            _ => HEAD + index * K as usize / 8,
        }
    }
    fn elements(&self, offset: usize) -> usize {
        match offset {
            0 => 0,
            _ => (offset - HEAD) * 8 / K as usize,
        }
    }
    fn element_bytes(&self) -> f64 {
        f64::from(K) / 8.0
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("form_{}({name}, {index})", self.name)
    }
    fn header(&self) -> Vec<String> {
        vec![
            minifloat(5, 10),
            rounding("f_round_f16", 5, 10, 65504.0),
            rounding("f_round_e4m3", 4, 3, 448.0),
            self.text(),
        ]
    }
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        Self::decode(bytes, index)
    }
    /// Codes come from the release. Nothing encodes here.
    fn pack(&self, _: &[f32]) -> Vec<u8> {
        panic!("{} codes are read from a release, never packed", self.name)
    }
}

/// The header, then the entries as the release stores them.
pub fn packed(inp: usize, out: usize, side: u32, entries: &[u8]) -> Vec<u8> {
    let shape = Shape { inp, out, side };
    let mut bytes = Vec::with_capacity(HEAD + entries.len());
    bytes.extend_from_slice(&shape.write());
    bytes.extend_from_slice(entries);
    bytes
}

/// The form for `k` bits a weight.
pub fn form_for(k: u32) -> Option<&'static dyn Form> {
    match k {
        2 => Some(&TRELLIS_K2),
        3 => Some(&TRELLIS_K3),
        4 => Some(&TRELLIS_K4),
        _ => None,
    }
}

/// A trellis form as a registered constant.
macro_rules! trellis {
    ($(#[$doc:meta])* $name:ident = $text:literal, $k:literal) => {
        $(#[$doc])*
        pub const $name: Trellis<$k> = Trellis::named($text);
        crate::form!($name);
    };
}

trellis! {
    /// exlr8 at two bits a weight.
    TRELLIS_K2 = "trellis_k2", 2
}

trellis! {
    /// exlr8 at three bits a weight.
    TRELLIS_K3 = "trellis_k3", 3
}

trellis! {
    /// exlr8 at four bits a weight.
    TRELLIS_K4 = "trellis_k4", 4
}
