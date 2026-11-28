//! Masked entries set to -inf, then packed edits.

use crate::WIDE;
use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::form::{F32_FORM, Load as Stored, plain};
use tessera_lower::{Draft, Kernel, Source, T, Tree, sites};

/// Masks vocab logits, four entries per thread.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn allow(
    #[form(x_form)] x: Buf<Dyn>,
    bits: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    width: u32,
    words: u32,
    stride: u32,
    count: u32,
    steps: u32,
    #[build] wide: bool,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if wide {
        let quads = width / WIDE;
        if gid() >= len / WIDE {
            return;
        }
        let col = (gid() % quads) * WIDE;
        let row = (gid() / quads) * stride;
        let word = bits[(row + col / 32) as u64];
        let nibble = (word >> (col % 32)) & 15u32;
        let base = gid() as u64 * WIDE as u64;
        let mut o = F32x4::zero();
        if nibble == 0u32 {
            o.x = f32::NEG_INFINITY;
            o.y = f32::NEG_INFINITY;
            o.z = f32::NEG_INFINITY;
            o.w = f32::NEG_INFINITY;
            out.store4(base, o);
        } else if nibble == 15u32 {
            let v = x.load4(base);
            o.x = v.x;
            o.y = v.y;
            o.z = v.z;
            o.w = v.w;
            out.store4(base, o);
        } else {
            let v = x.load4(base);
            o.x = if (nibble & 1u32) == 0u32 {
                f32::NEG_INFINITY
            } else {
                v.x
            };
            o.y = if (nibble & 2u32) == 0u32 {
                f32::NEG_INFINITY
            } else {
                v.y
            };
            o.z = if (nibble & 4u32) == 0u32 {
                f32::NEG_INFINITY
            } else {
                v.z
            };
            o.w = if (nibble & 8u32) == 0u32 {
                f32::NEG_INFINITY
            } else {
                v.w
            };
            out.store4(base, o);
        }
    } else {
        if gid() >= len {
            return;
        }
        let col = gid() % width;
        let row = (gid() / width) * stride;
        let word = bits[(row + col / 32) as u64];
        let mut v = if (word >> (col % 32)) & 1 == 0 {
            f32::NEG_INFINITY
        } else {
            x[gid() as u64]
        };
        if count > 0 {
            // A binary search for the entry's edit.
            let base = row + words;
            let mut lo = 0u32;
            let mut n = count;
            for _round in 0..steps {
                if n > 0 {
                    let step = n / 2;
                    if bits[(base + lo + step) as u64] < col {
                        lo = lo + step + 1;
                        n = n - step - 1;
                    } else {
                        n = step;
                    }
                }
            }
            if lo < count && bits[(base + lo) as u64] == col {
                let rep = f32::from_bits(bits[(base + 2 * count + lo) as u64]);
                if v > 0.0 {
                    v = v / rep;
                } else {
                    v = v * rep;
                }
                v = v + f32::from_bits(bits[(base + count + lo) as u64]);
            }
        }
        out[gid() as u64] = v;
    }
}

/// Rows of `width`, a mask row and `count` edits each.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    allow(self.wide, self.x_form, self.out_form),
    name = "allow",
    threads = if self.wide { self.len / WIDE as usize } else { self.len },
    work = self.len,
    accepts = allow_accepts,
    emits = allow_emits,
    source = allow_source,
)]
pub(crate) struct Allow {
    pub len: usize,
    pub width: usize,
    pub words: usize,
    pub stride: usize,
    pub count: usize,
    pub steps: usize,
    pub wide: bool,
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn allow_accepts(k: &Allow, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        _ => !form.narrow(),
    }
}

fn allow_emits(k: &Allow, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn allow_source(k: &Allow) -> Source {
    let body = allow_ir(k.wide, k.x_form, k.out_form);
    let mode = if k.wide { "w" } else { "s" };
    let entry = format!(
        "plan_allow_{mode}_{}_{}",
        k.x_form.name(),
        k.out_form.name(),
    );
    Source::named(body, entry)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (Stage::Allow, Some((_, all))) = (&site.stage, site.operand) else {
        return None;
    };
    let words = site.width.div_ceil(32);
    let stride = all / (site.len / site.width).max(1);
    let count = stride.saturating_sub(words) / 3;
    let kernel = Allow {
        len: site.len,
        width: site.width,
        words,
        stride,
        count,
        steps: (usize::BITS - count.leading_zeros()) as usize,
        wide: count == 0 && site.width.is_multiple_of(WIDE as usize),
        x_form: plain(tree.node(site.x).form),
        out_form: match tree.node(at).form.name() {
            "bf16" | "f16" if tree.node(at).form.stores() => tree.node(at).form,
            _ => F32_FORM,
        },
    };
    Some(vec![Draft::new(kernel, site.reads(), site.len)])
}

tessera_lower::rule!("allow", propose);
