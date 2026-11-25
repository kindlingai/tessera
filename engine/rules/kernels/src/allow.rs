//! Masked entries set to -inf, then packed edits.

use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::{Kernel, stage_rule};

/// One thread per entry. Rows hold mask words, then edits.
#[kernel]
fn allow(
    x: Buf<F32>,
    bits: Buf<U32>,
    mut out: BufMut<F32>,
    len: u32,
    width: u32,
    words: u32,
    stride: u32,
    count: u32,
    steps: u32,
) {
    if gid() >= len {
        return;
    }
    let col = gid() % width;
    let row = (gid() / width) * stride;
    let word = bits[(row + col / 32) as u64];
    let mut v = x[gid() as u64];
    if (word >> (col % 32)) & 1 == 0 {
        v = f32::NEG_INFINITY;
    }
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

/// Rows of `width`, a mask row and `count` edits each.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(allow, threads = self.len)]
pub(crate) struct Allow {
    pub len: usize,
    pub width: usize,
    pub words: usize,
    pub stride: usize,
    pub count: usize,
    pub steps: usize,
}

stage_rule!("allow", |site| {
    let (Stage::Allow, Some((_, all))) = (&site.stage, site.operand) else {
        return None;
    };
    let words = site.width.div_ceil(32);
    let stride = all / (site.len / site.width).max(1);
    let count = stride.saturating_sub(words) / 3;
    Some(Allow {
        len: site.len,
        width: site.width,
        words,
        stride,
        count,
        steps: (usize::BITS - count.leading_zeros()) as usize,
    })
});
