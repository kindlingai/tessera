//! Rows longer than a threshold, shrunk to it. Generated.

use crate::per_row;
use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::{Kernel, stage_rule};

/// One thread per row. Summing in order keeps batches alike.
#[kernel]
fn l2_clip(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32, threshold: f32, eps: f32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64] * x[at + i as u64];
    }
    // A zero row is kept, its root never taken.
    let mut by = 1.0f32;
    if sum > 0.0 {
        // The epsilon is added to the root, not under it.
        by = threshold / (1.0 / rsqrt(sum) + eps);
        if by > 1.0 {
            by = 1.0;
        }
    }
    for i in 0..width {
        out[at + i as u64] = x[at + i as u64] * by;
    }
}

/// One thread clips each row's length to `threshold`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    l2_clip,
    threads = self.rows,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub struct L2Clip {
    pub rows: usize,
    pub width: usize,
    pub threshold: f32,
    pub eps: f32,
}

stage_rule!("l2_clip", |site| {
    let Stage::L2Clip { threshold, eps } = site.stage else {
        return None;
    };
    Some(L2Clip {
        rows: site.len / site.width,
        width: site.width,
        threshold,
        eps,
    })
});
