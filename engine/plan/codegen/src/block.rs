//! Block forms in kernel text: helpers that decode a block.

use crate as tessera_codegen;
use crate::form::{Load, form, kernel_fns};
use crate::inline::{Inline, Pass};
use crate::ir::{Expr, Helper};
use crate::prelude::{Bits, Buf, F32x4, U8, device, helper, kernel};
use tessera_graph::{E2M1, E4M3, E8M0, F16};
use tessera_mem::block::{Block, Crumb, Encode, I8, Nibble, Scale};

/// A code, typed for kernel text.
pub trait LoadCode: Encode {
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

tessera_mem::block_fns!(kernel_fns);

/// Each scale and code's kernel reader, by its shared body.
macro_rules! kernel_reads {
    (scales: $(($s:ident, $repr:ty, $smax:literal, $get:ident)),*;
     codes: $(($c:ident, $bits:literal, $cmax:literal, $field:ident)),*;) => {
        $(
            impl LoadScale for $s {
                fn load_inline(p: impl Pass<Expr>) -> Inline {
                    $get::body_inline(p)
                }
            }
        )*
        $(
            impl LoadCode for $c {
                fn field_inline(w: impl Pass<Expr>, low: impl Pass<Expr>) -> Inline {
                    $field::body_inline(w, low)
                }
            }
        )*
    };
}

tessera_mem::block_reads!(kernel_reads);

/// Element `i` of `p`: head, block scale and code.
#[helper("form_{name}")]
fn element<S: LoadScale, C: LoadCode, const B: usize, const HALVES: bool, const HEAD: bool>(
    #[build] name: &'static str,
    p: Buf<U8>,
    i: u64,
) -> f32 {
    let j = i % B as u64;
    let b = p + (HEAD as u64 * 4 + (i / B as u64) * (S::BYTES + B * C::BITS as usize / 8) as u64);
    let byte = code_byte::body(C::BITS, B as u64, HALVES, j);
    let shift = code_shift::body(C::BITS, B as u64, HALVES, j);
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
    let at = code_byte::body(C::BITS, B as u64, HALVES, j);
    let step = if C::BITS == 2 {
        2
    } else if C::BITS == 8 || HALVES {
        8
    } else {
        4
    };
    let shift = if C::BITS == 8 || C::BITS == 2 || !HALVES {
        0
    } else {
        (4 * (j / (B / 2) as u64)) as u32
    };
    let c = b + (S::BYTES as u64 + at);
    // Four nibbles fill two bytes, four crumbs fill one.
    let wide = C::BITS == 8 || HALVES;
    // Scales of two bytes keep the codes two-aligned.
    let w = if C::BITS == 2 {
        c[0] as u32
    } else if S::BYTES == 2 && wide {
        c.half(0) | (c.half(2) << 16)
    } else if S::BYTES == 2 {
        c.half(0)
    } else if wide {
        c[0] as u32 | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
    } else {
        c[0] as u32 | ((c[1] as u32) << 8)
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

macro_rules! register_blocks {
    ($(($name:ident, $($rest:tt)*),)*) => {
        $(form!(tessera_mem::block::$name);)*
    };
}

tessera_mem::block_forms!(register_blocks);
