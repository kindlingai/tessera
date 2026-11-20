//! The exlr8 trellis form in kernel text: tiles and codebook.

use crate as sensai_codegen;
use crate::form::{Load, form, named};
use crate::ir::Helper;
use crate::prelude::{Bits, Buf, F32x4, U8, helper};
use sensai_mem::trellis::*;

pub use sensai_mem::trellis::packed;

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
fn value(#[build] name: &'static str, #[build] mul1: u32, state: u32) -> f32 {
    let s = state & 0xFFFF;
    let x = s * MCG;
    let y = (x & 0x8FFF8FFF) ^ 0x3B603B60;
    let z = s * MUL1;
    let sum = (z & 0xFF) + ((z >> 8) & 0xFF) + ((z >> 16) & 0xFF) + (z >> 24);
    let scaled = (1024 + sum) as f32 * MUL1_SCALE + MUL1_OFFSET;
    let raw = if mul1 != 0 {
        scaled
    } else {
        y.f16(0) + y.f16(16)
    };
    let v = rounding("f_round_f16", 5, 10, 65504.0, raw);
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
fn tile<const K: u32>(#[build] name: &'static str, p: Buf<U8>, row: u32, at: u32) -> Buf<U8> {
    let inp = p.word(0);
    let out = p.word(4);
    // Columns past `out` lie in later matrices of a set.
    let col = at % out;
    let first = (at / out) as u64 * (inp / 16) as u64 * (out / 16) as u64;
    let t = if p.word(8) == INPUT_SIDE {
        (row / 128) * (out / 2) + 64 * (col / 128) + 8 * ((row % 128) / 16) + (col % 128) / 16
    } else {
        (col / 128) * (inp / 2) + 8 * (row / 16) + (col % 128) / 16
    };
    p + (HEAD as u64 + (first + t as u64) * (32 * K) as u64)
}

/// Stream position of tile element `(r, c)`.
#[helper("form_{name}_j")]
fn slot(#[build] name: &'static str, r: u32, c: u32) -> u32 {
    8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) as u32 + 2 * (r >= 8) as u32 + (r % 2)
}

/// Element `i` of a `[out][in]` matrix at `p`.
#[helper("form_{name}")]
fn one<const K: u32>(#[build] name: &'static str, #[build] mul1: u32, p: Buf<U8>, i: u64) -> f32 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    value(name, mul1, win::<K>(name, t, start::<K>(name, j)) >> 16)
}

/// Four along a row: codes `j`, `j+1`, `j+8`, `j+9`.
#[helper("form_{name}_x4")]
fn four<const K: u32>(
    #[build] name: &'static str,
    #[build] mul1: u32,
    p: Buf<U8>,
    i: u64,
) -> F32x4 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    let a = win::<K>(name, t, start::<K>(name, j));
    let b = win::<K>(name, t, start::<K>(name, j + 8));
    F32x4::new(
        value(name, mul1, a >> 16),
        value(name, mul1, a >> (16 - K)),
        value(name, mul1, b >> 16),
        value(name, mul1, b >> (16 - K)),
    )
}

impl<const K: u32> Load for Trellis<K> {
    fn codes(&self) -> Option<&'static dyn Load> {
        form_for(K, self.codebook).filter(|_| self.rotated)
    }
    fn ty(&self) -> &'static str {
        "u8"
    }
    fn heavy(&self) -> bool {
        true
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("form_{}({name}, {index})", self.name)
    }
    /// Four along a row share one tile.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("form_{}_x4({name}, {index})", self.name)
    }
    fn header(&self) -> Vec<Helper> {
        let mul1 = u32::from(self.codebook == Codebook::Mul1);
        vec![one_ir::<K>(self.name, mul1), four_ir::<K>(self.name, mul1)]
    }
}

/// The form for `k` bits a weight in `codebook`.
pub fn form_for(k: u32, codebook: Codebook) -> Option<&'static dyn Load> {
    named(sensai_mem::trellis::form_for(k, codebook)?.name())
}

form!(TRELLIS_K2);
form!(TRELLIS_K3);
form!(TRELLIS_K4);
form!(TRELLIS_K2_ROT);
form!(TRELLIS_K3_ROT);
form!(TRELLIS_K4_ROT);
form!(TRELLIS_K2_MUL1);
form!(TRELLIS_K3_MUL1);
form!(TRELLIS_K4_MUL1);
form!(TRELLIS_K2_MUL1_ROT);
form!(TRELLIS_K3_MUL1_ROT);
form!(TRELLIS_K4_MUL1_ROT);
