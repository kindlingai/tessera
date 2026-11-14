//! Joins rows of two inputs along the last axis.

use codegen::prelude::*;
use lower::{Extent, Kernel};

/// One thread per output element, from whichever side holds it.
#[kernel]
fn join(a: Buf<F32>, b: Buf<F32>, mut out: BufMut<F32>, left: u32, right: u32, rows: u32) {
    let width = left + right;
    let row = gid() / width;
    let col = gid() % width;
    if row >= rows {
        return;
    }
    let v = if col < left {
        a[(row * left + col) as u64]
    } else {
        b[(row * right + col - left) as u64]
    };
    out[gid() as u64] = v;
}

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(join, threads = self.rows.times(self.left + self.right), sizes = [self.left, self.right])]
pub struct Join {
    pub left: usize,
    pub right: usize,
    /// The row count. Decode reads it at launch.
    pub rows: Extent,
}
