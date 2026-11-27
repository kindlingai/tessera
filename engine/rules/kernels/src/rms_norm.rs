//! RMS norm over rows, generated.

use crate::per_row;
use tessera_codegen::prelude::*;
use tessera_lower::{Kernel, stage_rule};

/// One thread per row. Summing in order keeps batches alike.
#[kernel]
fn rms_norm<const WEIGHTED: bool>(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    eps: f32,
) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64] * x[at + i as u64];
    }
    let inv = rsqrt(sum / width as f32 + eps);
    for i in 0..width {
        let gain = if WEIGHTED { w[i as u64] } else { 1.0 };
        out[at + i as u64] = x[at + i as u64] * inv * gain;
    }
}

/// One thread scales each row by its root mean square.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    rms_norm,
    variant = self.weighted,
    name = match self.weighted {
        true => "rms_norm",
        false => "rms",
    },
    threads = self.rows,
    // The gains are read once per row.
    load = Some(per_row(
        self.rows,
        self.width,
        body.trip(),
        4.0 * self.width as f64 * f64::from(self.weighted),
    )),
    load_key = tessera_lower::load_key((self.rows, self.width, self.eps.to_bits(), self.weighted)),
)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

stage_rule!(
    "rms_norm",
    |site| {
        let (eps, weighted) = site.rms()?;
        Some(RmsNorm {
            rows: site.len / site.width,
            width: site.width,
            eps,
            weighted,
        })
    },
    |site| site.padded()
);
