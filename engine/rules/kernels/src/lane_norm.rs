//! RMS norm, a SIMD group per row, summed by shuffles.

use crate::{AHEAD, WIDE, per_lane_row};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::{Kernel, Needs, stage_rule};

/// Sums a row's squares, then scales the row.
#[kernel]
fn norm_lanes<const WIDE: u32, const AHEAD: u32, const WEIGHTED: bool>(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    eps: f32,
) {
    let row = gid() / LANES as u32;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let mut acc = 0.0f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        acc += v.x * v.x;
        acc += v.y * v.y;
        acc += v.z * v.z;
        acc += v.w * v.w;
    }
    acc = sum32(acc);
    let inv = rsqrt(acc / width as f32 + eps);
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        let mut g = F32x4::zero();
        if WEIGHTED {
            g = w.load4(i as u64);
        } else {
            g.x = 1.0;
            g.y = 1.0;
            g.z = 1.0;
            g.w = 1.0;
        }
        let mut o = F32x4::zero();
        o.x = v.x * inv * g.x;
        o.y = v.y * inv * g.y;
        o.z = v.z * inv * g.z;
        o.w = v.w * inv * g.w;
        out.store4(at + i as u64, o);
    }
}

/// An RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    norm_lanes::<WIDE, AHEAD>,
    variant = self.weighted,
    name = match self.weighted {
        true => "lane_norm",
        false => "lane_rms",
    },
    threads = self.rows * LANES,
    work = self.rows * self.width,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    // The gains are read once.
    load = Some(per_lane_row(
        self.rows,
        self.width,
        body.trip(),
        4.0 * (self.rows * self.width) as f64 * f64::from(self.weighted),
    )),
    unproven,
)]
pub(crate) struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

stage_rule!(
    "lane_norm",
    |site| {
        let (eps, weighted) = site.rms()?;
        // Wide loads need whole groups of four along a row.
        site.width.is_multiple_of(WIDE as usize).then(|| LaneNorm {
            rows: site.len / site.width,
            width: site.width,
            eps,
            weighted,
        })
    },
    |site| site.padded()
);
