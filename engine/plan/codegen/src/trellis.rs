//! The exlr8 trellis form: K-bit codes in tiles.

use crate as codegen;
use crate::form::Form;
use crate::ir::Helper;
use crate::prelude::{Bits, Buf, F32x4, U8, helper};
use graph::{DType, E4M3, F16};

/// The codebook multiplier.
const MCG: u32 = 0xCBAC_1FED;

/// Header words: `in`, `out`, the sharded side, `1 / in`.
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
        let inverse = (1.0 / self.inp as f32).to_bits();
        for (i, v) in [self.inp as u32, self.out as u32, self.side, inverse]
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
fn codebook(state: u32) -> f32 {
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
        let (k, b) = (K as usize, j * K as usize);
        if 16 % k != 0 {
            return (0..k).fold(0, |v, i| (v << 1) | bit(tile, b + i));
        }
        // Codes that divide a word sit in one, high first.
        let word = (b / 16) ^ 1;
        let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
        u32::from(w >> (16 - k - b % 16)) & ((1 << k) - 1)
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
}

/// `x` rounded to nearest even, `e` and `m` bits.
#[helper("{name}")]
fn rounding(
    #[build] name: &'static str,
    #[build] e: u32,
    #[build] m: u32,
    #[build] top: f32,
    x: f32,
) -> f32 {
    let a = if x < 0.0 { -x } else { x };
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    // A subnormal: in units of its step, rounded through 2^23.
    let steps = f32::from_bits((125 + (1u32 << (e - 1)) + m) << 23);
    let small = (a * steps + 8388608.0) - 8388608.0;
    let bits = a.to_bits();
    let nudge = ((1u32 << (22 - m)) - 1) + ((bits >> (23 - m)) & 1);
    let r = f32::from_bits((bits + nudge) & !((1u32 << (23 - m)) - 1));
    let big = if r > top { top } else { r };
    let least = f32::from_bits((129 - (1u32 << (e - 1))) << 23);
    sign * if a < least { small / steps } else { big }
}

/// A swapped word pair as 32 stream bits.
#[helper("form_{name}_pair")]
fn pair<const K: u32>(#[build] name: &'static str, t: Buf<U8>, q: u32) -> u32 {
    t.word(4 * (q % (8 * K)))
}

/// 32 stream bits from bit `s`, wrapping at the end.
#[helper("form_{name}_win")]
fn win<const K: u32>(#[build] name: &'static str, t: Buf<U8>, s: u32) -> u32 {
    let q = s >> 5;
    let sh = s & 31;
    let hi = pair::<K>(name, t, q);
    if sh == 0 {
        hi
    } else {
        (hi << sh) | (pair::<K>(name, t, q + 1) >> (32 - sh))
    }
}

/// A state's first bit, for the code at `j`.
#[helper("form_{name}_start")]
fn start<const K: u32>(#[build] name: &'static str, j: u32) -> u32 {
    // A state spans `16 / K` codes, wrapping at 256.
    ((j + (257 - (16 + K - 1) / K)) % 256) * K + ((16 + K - 1) / K * K - 16)
}

/// A state's weight: hashed, then rounded to f16 and e4m3.
#[helper("form_{name}_value")]
fn value(#[build] name: &'static str, state: u32) -> f32 {
    let x = (state & 0xFFFF) * MCG;
    let y = (x & 0x8FFF8FFF) ^ 0x3B603B60;
    let v = rounding("f_round_f16", 5, 10, 65504.0, y.f16(0) + y.f16(16));
    rounding("f_round_e4m3", 4, 3, 448.0, v)
}

/// Element `n`'s column: a float estimate, corrected by one.
#[helper("form_{name}_col")]
fn col(#[build] name: &'static str, p: Buf<U8>, n: u32, inp: u32) -> u32 {
    let c = (n as f32 * f32::from_bits(p.word(12))) as u32;
    let row = n - c * inp;
    // The top bit set: the row wrapped below zero.
    if (row >> 31) != 0 {
        c - 1
    } else if row >= inp {
        c + 1
    } else {
        c
    }
}

/// The tile holding `T[row][col]`, counted in the entries.
#[helper("form_{name}_tile")]
fn tile<const K: u32>(#[build] name: &'static str, p: Buf<U8>, row: u32, col: u32) -> Buf<U8> {
    let inp = p.word(0);
    let out = p.word(4);
    let t = if p.word(8) == INPUT_SIDE {
        (row / 128) * (out / 2) + 64 * (col / 128) + 8 * ((row % 128) / 16) + (col % 128) / 16
    } else {
        (col / 128) * (inp / 2) + 8 * (row / 16) + (col % 128) / 16
    };
    p + (HEAD as u64 + t as u64 * (32 * K) as u64)
}

/// Stream position of tile element `(r, c)`.
#[helper("form_{name}_j")]
fn slot(#[build] name: &'static str, r: u32, c: u32) -> u32 {
    8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) as u32 + 2 * (r >= 8) as u32 + (r % 2)
}

/// Element `i` of a `[out][in]` matrix at `p`.
#[helper("form_{name}")]
fn one<const K: u32>(#[build] name: &'static str, p: Buf<U8>, i: u64) -> f32 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    value(name, win::<K>(name, t, start::<K>(name, j)) >> 16)
}

/// Four along a row: codes `j`, `j+1`, `j+8`, `j+9`.
#[helper("form_{name}_x4")]
fn four<const K: u32>(#[build] name: &'static str, p: Buf<U8>, i: u64) -> F32x4 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    let a = win::<K>(name, t, start::<K>(name, j));
    let b = win::<K>(name, t, start::<K>(name, j + 8));
    F32x4::new(
        value(name, a >> 16),
        value(name, a >> (16 - K)),
        value(name, b >> 16),
        value(name, b >> (16 - K)),
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
    /// A tile, so no float row copies in as is.
    fn block(&self) -> usize {
        256
    }
    fn packs(&self) -> bool {
        false
    }
    fn heavy(&self) -> bool {
        true
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
    /// Four along a row share one tile.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("form_{}_x4({name}, {index})", self.name)
    }
    fn header(&self) -> Vec<Helper> {
        vec![one_ir::<K>(self.name), four_ir::<K>(self.name)]
    }
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        Self::decode(bytes, index)
    }
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
