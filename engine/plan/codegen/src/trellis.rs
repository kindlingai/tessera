//! The exlr8 trellis form in kernel text: tiles and codebook.

use crate as tessera_codegen;
use crate::form::{Load, form, named};
use crate::inline::{Inline, Pass};
use crate::ir::{Expr, Helper};
use crate::prelude::{Bits, Buf, F32x4, U8, device, helper, round_e4m3, round_f16};
use tessera_mem::trellis::*;

pub use tessera_mem::trellis::{INPUT_SIDE, packed};

/// Each shared trellis function as a helper, `name::body`.
macro_rules! kernel_fns {
    ($($(#[$doc:meta])* $kernel:literal
        fn $name:ident $(<const $c:ident: $ct:ty>)?
            ($($(#[$build:meta])* $arg:ident: $ty:tt),*) -> $ret:tt $body:block)*) => {$(
        $(#[$doc])*
        pub mod $name {
            use super::*;
            #[helper($kernel)]
            pub fn body $(<const $c: $ct>)? ($($(#[$build])* $arg: $ty),*) -> $ret $body
        }
    )*};
}

tessera_mem::trellis_fns!(kernel_fns);

/// A codebook, typed for kernel text.
pub trait LoadBook {
    /// Bits in a state.
    const BITS: u32;
    /// Whether a state holds two weights.
    const PAIRS: bool;

    /// Weight `odd` of `state`, in a kernel.
    fn weight(_: u32, _: u32) -> f32 {
        device()
    }

    /// `weight` as a tree.
    fn weight_inline(state: impl Pass<Expr>, odd: impl Pass<Expr>) -> Inline;
}

/// Codebook types, and a form's helpers in each.
macro_rules! kernel_books {
    ($(($book:ident, $name:literal, $suffix:literal, $bits:literal, $pairs:literal,
        $weight:ident),)*) => {
        /// Codebooks as types.
        pub mod book {
            $(
                #[doc = concat!("The ", $name, " codebook.")]
                pub struct $book;
            )*
        }

        $(
            impl LoadBook for book::$book {
                const BITS: u32 = $bits;
                const PAIRS: bool = $pairs;

                fn weight_inline(state: impl Pass<Expr>, odd: impl Pass<Expr>) -> Inline {
                    $weight::body_inline(state, odd)
                }
            }
        )*

        /// The one-element and four-element helpers of a form.
        fn decoders<const K: u32>(name: &'static str, codebook: Codebook) -> [Helper; 2] {
            match codebook {
                $(Codebook::$book => [
                    one_ir::<book::$book, K>(name),
                    four_ir::<book::$book, K>(name),
                ],)*
            }
        }
    };
}

tessera_mem::codebooks!(kernel_books);

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

/// Weight `odd` of window `w`: pair half, or next code.
#[helper("form_{name}_lane")]
fn lane<B: LoadBook, const K: u32>(#[build] name: &'static str, w: u32, odd: u32) -> f32 {
    if B::PAIRS {
        B::weight(w >> (32 - B::BITS), odd)
    } else {
        B::weight(w >> (32 - B::BITS - K * odd), 0)
    }
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
    let t = local::body(inp, out, side, row, col);
    let base = if (side & 0x8000_0000) != 0 {
        (p.word(24 + m * 8) as u64) | ((p.word(28 + m * 8) as u64) << 32)
    } else {
        HEAD as u64 + m as u64 * (inp / 16) as u64 * (out / 16) as u64 * (32 * K) as u64
    };
    p + (base + t as u64 * (32 * K) as u64)
}

/// Element `i` of a `[out][in]` matrix at `p`.
#[helper("form_{name}")]
fn one<B: LoadBook, const K: u32>(#[build] name: &'static str, p: Buf<U8>, i: u64) -> f32 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot::body(row % 16, c % 16);
    let odd = if B::PAIRS { j % 2 } else { 0 };
    lane::<B, K>(
        name,
        win::<K>(name, t, start::body::<K>(B::BITS, B::PAIRS, j)),
        odd,
    )
}

/// Four along a row: codes `j`, `j+1`, `j+8`, `j+9`.
#[helper("form_{name}_x4")]
fn four<B: LoadBook, const K: u32>(#[build] name: &'static str, p: Buf<U8>, i: u64) -> F32x4 {
    let inp = p.word(0);
    let n = i as u32;
    let c = col(name, p, n, inp);
    let row = n - c * inp;
    let t = tile::<K>(name, p, row, c);
    let j = slot::body(row % 16, c % 16);
    let a = win::<K>(name, t, start::body::<K>(B::BITS, B::PAIRS, j));
    let b = win::<K>(name, t, start::body::<K>(B::BITS, B::PAIRS, j + 8));
    F32x4::new(
        lane::<B, K>(name, a, 0),
        lane::<B, K>(name, a, 1),
        lane::<B, K>(name, b, 0),
        lane::<B, K>(name, b, 1),
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
        let [one, four] = decoders::<K>(self.name, self.codebook);
        Some(if wide { four } else { one })
    }
    fn header(&self) -> Vec<Helper> {
        decoders::<K>(self.name, self.codebook).into()
    }
}

/// The form for `k` bits a weight in `codebook`.
pub fn form_for(k: u32, codebook: Codebook) -> Option<&'static dyn Load> {
    named(tessera_mem::trellis::form_for(k, codebook)?.name())
}

macro_rules! register_trellis {
    ($(($book:ident, $suffix:literal, $k:literal, $plain:ident, $turned:ident, $kind:literal),)*) => {
        $(
            form!(tessera_mem::trellis::$plain);
            form!(tessera_mem::trellis::$turned);
        )*
    };
}

tessera_mem::trellis_forms!(register_trellis);
