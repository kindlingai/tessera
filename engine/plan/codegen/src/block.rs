//! Block forms in kernel text: helpers that decode a block.

use crate as sensai_codegen;
use crate::form::{Load, form};
use crate::inline::{Inline, Pass};
use crate::ir::{Expr, Helper};
use crate::prelude::{Bits, Buf, F32x4, U8, device, helper, kernel};
use sensai_graph::{E2M1, E4M3, E8M0, F16};
use sensai_mem::block::{Block, Code, I8, Nibble, Scale};
use sensai_mem::block::{INT4_G32, INT4_G128, MXFP4, MXFP8, NVFP4, Q4_0, Q8_0};

/// A code, typed for kernel text.
pub trait LoadCode: Code {
    /// Code at bit `low` of `w`, as a float.
    fn field(_: u32, _: u32) -> f32 {
        device()
    }

    /// `field` as a tree, for the block's helper.
    fn field_inline(w: impl Pass<Expr>, low: impl Pass<Expr>) -> Inline;
}

/// A scale, typed for kernel text.
pub trait LoadScale: Scale {
    /// The scale starting at byte `p`, in a kernel.
    fn load(_: Buf<U8>) -> f32 {
        device()
    }

    /// `load` as a tree, for the block's helper.
    fn load_inline(p: impl Pass<Expr>) -> Inline;
}

/// An f16 scale: two bytes, low first.
#[kernel]
fn f16_scale(p: Buf<U8>) -> f32 {
    (p[0] as u32 | ((p[1] as u32) << 8)).f16(0)
}

/// An e4m3 scale: one byte.
#[kernel]
fn e4m3_scale(p: Buf<U8>) -> f32 {
    (p[0] as u32).e4m3(0)
}

/// An e8m0 scale: one byte, the f32 exponent.
#[kernel]
fn e8m0_scale(p: Buf<U8>) -> f32 {
    f32::from_bits((p[0] as u32) << 23)
}

/// An e4m3 byte.
#[kernel]
fn e4m3_field(w: u32, low: u32) -> f32 {
    w.e4m3(low)
}

/// An e2m1 nibble.
#[kernel]
fn e2m1_field(w: u32, low: u32) -> f32 {
    w.e2m1(low)
}

/// A signed byte.
#[kernel]
fn i8_field(w: u32, low: u32) -> f32 {
    w.signed(low, 8)
}

/// A nibble 8 below its value.
#[kernel]
fn nibble_field(w: u32, low: u32) -> f32 {
    w.bits(low, 4) as f32 - 8.0
}

/// Each type's kernel reader, by its decode helper.
macro_rules! reads {
    ($($trait:ident :: $method:ident for $t:ident = $helper:ident($($arg:ident),*);)*) => {$(
        impl $trait for $t {
            fn $method($($arg: impl Pass<Expr>),*) -> Inline {
                $helper($($arg),*)
            }
        }
    )*};
}

reads! {
    LoadScale::load_inline for F16 = f16_scale_inline(p);
    LoadScale::load_inline for E4M3 = e4m3_scale_inline(p);
    LoadScale::load_inline for E8M0 = e8m0_scale_inline(p);
    LoadCode::field_inline for E4M3 = e4m3_field_inline(w, low);
    LoadCode::field_inline for E2M1 = e2m1_field_inline(w, low);
    LoadCode::field_inline for I8 = i8_field_inline(w, low);
    LoadCode::field_inline for Nibble = nibble_field_inline(w, low);
}

/// Element `i` of `p`: head, block scale and code.
#[helper("form_{name}")]
fn element<S: LoadScale, C: LoadCode, const B: usize, const HALVES: bool, const HEAD: bool>(
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
    let head = if HEAD {
        f32::from_bits(
            p[0] as u32 | ((p[1] as u32) << 8) | ((p[2] as u32) << 16) | ((p[3] as u32) << 24),
        )
    } else {
        1.0
    };
    head * S::load(b) * C::field(b[S::BYTES as u64 + byte] as u32, shift)
}

/// Elements `i..i+4`: head, scale and four codes.
#[helper("form_{name}_x4")]
fn four<S: LoadScale, C: LoadCode, const B: usize, const HALVES: bool, const HEAD: bool>(
    #[build] name: &'static str,
    p: Buf<U8>,
    i: u64,
) -> F32x4 {
    let j = i % B as u64;
    let b = p + (HEAD as u64 * 4 + (i / B as u64) * (S::BYTES + B * C::BITS as usize / 8) as u64);
    // First byte of the codes, then each shift.
    let at = if C::BITS == 8 {
        j
    } else if HALVES {
        j % (B / 2) as u64
    } else {
        j / 2
    };
    let step = if C::BITS == 8 || HALVES { 8 } else { 4 };
    let shift = if C::BITS == 8 || !HALVES {
        0
    } else {
        (4 * (j / (B / 2) as u64)) as u32
    };
    let c = b + (S::BYTES as u64 + at);
    // Scales of two bytes keep the codes two-aligned.
    let w = if S::BYTES == 2 {
        c.half(0) | (c.half(2) << 16)
    } else {
        c[0] as u32 | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
    };
    let head = if HEAD { f32::from_bits(p.word(0)) } else { 1.0 };
    let scale = head * S::load(b);
    F32x4::new(
        scale * C::field(w, shift),
        scale * C::field(w, shift + step),
        scale * C::field(w, shift + 2 * step),
        scale * C::field(w, shift + 3 * step),
    )
}

impl<S: LoadScale, C: LoadCode, const B: usize, const HALVES: bool, const HEAD: bool> Load
    for Block<S, C, B, HALVES, HEAD>
{
    fn ty(&self) -> &'static str {
        "u8"
    }
    /// Nibbles: one word holds four codes of each half.
    fn pairs(&self) -> usize {
        match (C::BITS, HALVES) {
            (4, true) => B / 2,
            (4, false) => 4,
            _ => 0,
        }
    }

    fn load(&self, name: &str, index: &str) -> String {
        format!("form_{}({name}, {index})", self.name)
    }

    /// Four codes share their block's scale and bytes.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("form_{}_x4({name}, {index})", self.name)
    }

    fn header(&self) -> Vec<Helper> {
        vec![
            element_ir::<S, C, B, HALVES, HEAD>(self.name),
            four_ir::<S, C, B, HALVES, HEAD>(self.name),
        ]
    }
}

form!(Q8_0);
form!(Q4_0);
form!(NVFP4);
form!(MXFP8);
form!(MXFP4);
form!(INT4_G32);
form!(INT4_G128);
