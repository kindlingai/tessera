//! The exlr8 trellis form in kernel text: tiles and codebook.

use crate as tessera_codegen;
use crate::form::{Load, form, named};
use crate::ir::Helper;
use crate::prelude::{Bits, Buf, F32x4, U8, helper, round_e4m3, round_f16};
use tessera_mem::trellis::*;

pub use tessera_mem::trellis::{INPUT_SIDE, packed};

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

/// Build code of the mul1 codebook.
const MUL1_BOOK: u32 = 1;
/// Build code of pmcgx. Its states hold pairs.
const PAIRS: u32 = 2;

/// A codebook's build code.
fn book(codebook: Codebook) -> u32 {
    match codebook {
        Codebook::Mcg => 0,
        Codebook::Mul1 => MUL1_BOOK,
        Codebook::Pmcgx => PAIRS,
    }
}

/// The first bit of code `j`'s state.
#[helper("form_{name}_start")]
fn first<const K: u32>(#[build] name: &'static str, #[build] book: u32, j: u32) -> u32 {
    if book == PAIRS {
        // A pair's 20 bits end at bit 2K(p+1), wrapping.
        (2 * K * (j / 2 + 1) + 256 * K - 20) % (256 * K)
    } else {
        // A state spans `16 / K` codes, wrapping at 256.
        ((j + (257 - (16 + K - 1) / K)) % 256) * K + ((16 + K - 1) / K * K - 16)
    }
}

/// Weight `odd` of a pair state: two hashes, then rounded.
#[helper("form_{name}_pair_value")]
fn pair_value(#[build] name: &'static str, state: u32, odd: u32) -> f32 {
    let w = state & 0xFFFFF;
    let x = ((w * MCG) & 0x8FFF8FFF) ^ 0x3B603B60;
    let y = ((w * PMCG2) & 0x8FFF8FFF) ^ 0x3B603B60;
    let raw = if odd != 0 {
        x.f16(16) + y.f16(0)
    } else {
        x.f16(0) + y.f16(16)
    };
    round_e4m3(round_f16(raw))
}

/// Weight `odd` of window `w`: pair half, or next code.
#[helper("form_{name}_lane")]
fn lane<const K: u32>(#[build] name: &'static str, #[build] book: u32, w: u32, odd: u32) -> f32 {
    if book == PAIRS {
        pair_value(name, w >> 12, odd)
    } else {
        value(name, book, w >> (16 - K * odd))
    }
}

/// A state's weight: hashed, then rounded to f16 and e4m3.
#[helper("form_{name}_value")]
fn value(#[build] name: &'static str, #[build] book: u32, state: u32) -> f32 {
    let s = state & 0xFFFF;
    let x = s * MCG;
    let y = (x & 0x8FFF8FFF) ^ 0x3B603B60;
    let z = s * MUL1;
    let sum = (z & 0xFF) + ((z >> 8) & 0xFF) + ((z >> 16) & 0xFF) + (z >> 24);
    let scaled = (1024 + sum) as f32 * MUL1_SCALE + MUL1_OFFSET;
    let raw = if book == MUL1_BOOK {
        scaled
    } else {
        y.f16(0) + y.f16(16)
    };
    round_e4m3(round_f16(raw))
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
    let side = p.word(8);
    // Columns past `out` lie in later matrices of a set.
    let col = at % out;
    let m = at / out;
    let t = if (side & 0xFFFF) == INPUT_SIDE {
        (row / 128) * (out / 2) + 64 * (col / 128) + 8 * ((row % 128) / 16) + (col % 128) / 16
    } else {
        (col / 128) * (inp / 2) + 8 * (row / 16) + (col % 128) / 16
    };
    let base = if (side & 0x8000_0000) != 0 {
        (p.word(24 + m * 8) as u64) | ((p.word(28 + m * 8) as u64) << 32)
    } else {
        HEAD as u64 + m as u64 * (inp / 16) as u64 * (out / 16) as u64 * (32 * K) as u64
    };
    p + (base + t as u64 * (32 * K) as u64)
}

/// Stream position of tile element `(r, c)`.
#[helper("form_{name}_j")]
fn slot(#[build] name: &'static str, r: u32, c: u32) -> u32 {
    8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) as u32 + 2 * (r >= 8) as u32 + (r % 2)
}

/// Element `i` of a `[out][in]` matrix at `p`.
#[helper("form_{name}")]
fn one<const K: u32>(#[build] name: &'static str, #[build] book: u32, p: Buf<U8>, i: u64) -> f32 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    let odd = if book == PAIRS { j % 2 } else { 0 };
    lane::<K>(
        name,
        book,
        win::<K>(name, t, first::<K>(name, book, j)),
        odd,
    )
}

/// Four along a row: codes `j`, `j+1`, `j+8`, `j+9`.
#[helper("form_{name}_x4")]
fn four<const K: u32>(
    #[build] name: &'static str,
    #[build] book: u32,
    p: Buf<U8>,
    i: u64,
) -> F32x4 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot(name, row % 16, c % 16);
    let a = win::<K>(name, t, first::<K>(name, book, j));
    let b = win::<K>(name, t, first::<K>(name, book, j + 8));
    F32x4::new(
        lane::<K>(name, book, a, 0),
        lane::<K>(name, book, a, 1),
        lane::<K>(name, book, b, 0),
        lane::<K>(name, book, b, 1),
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
    fn decoder(&self, wide: bool) -> Option<Helper> {
        let book = book(self.codebook);
        Some(match wide {
            true => four_ir::<K>(self.name, book),
            false => one_ir::<K>(self.name, book),
        })
    }
    fn header(&self) -> Vec<Helper> {
        let book = book(self.codebook);
        vec![one_ir::<K>(self.name, book), four_ir::<K>(self.name, book)]
    }
}

/// The form for `k` bits a weight in `codebook`.
pub fn form_for(k: u32, codebook: Codebook) -> Option<&'static dyn Load> {
    named(tessera_mem::trellis::form_for(k, codebook)?.name())
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
form!(TRELLIS_K2_PMCGX);
form!(TRELLIS_K3_PMCGX);
form!(TRELLIS_K4_PMCGX);
form!(TRELLIS_K2_PMCGX_ROT);
form!(TRELLIS_K3_PMCGX_ROT);
form!(TRELLIS_K4_PMCGX_ROT);
