//! Rows longer than a threshold, shrunk to it. Generated.

use crate::{AHEAD, WIDE, per_row};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::{Draft, Kernel, Load, Needs, T, Tree, sites, stage_rule};

/// The narrowest row for a SIMD group.
const LANE_MIN: usize = 16;

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
    // The epsilon is added to the root, not under it.
    let by = clip_by(sum, threshold, eps);
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
struct L2Clip {
    rows: usize,
    width: usize,
    threshold: f32,
    eps: f32,
}

/// The scale that clips a row of squared length `sum`.
#[kernel]
fn clip_by(sum: f32, threshold: f32, eps: f32) -> f32 {
    // A zero row is kept, its root never taken.
    let mut by = 1.0f32;
    if sum > 0.0 {
        by = threshold / (1.0 / rsqrt(sum) + eps);
        if by > 1.0 {
            by = 1.0;
        }
    }
    by
}

/// A SIMD group per row, summed by shuffles.
#[kernel]
fn l2_clip_lanes<const WIDE: u32, const CACHED: bool>(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    threshold: f32,
    eps: f32,
    #[build] ahead: u32,
) {
    let row = gid() / LANES as u32;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let col = lane() * WIDE;
    // A short row stays in registers between passes.
    let mut v = F32x4::new(0.0, 0.0, 0.0, 0.0);
    let mut acc = 0.0f32;
    if CACHED {
        if col < width {
            v = x.load4(at + col as u64);
        }
        acc = (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    } else {
        for i in (col..width).by(LANES as u32 * WIDE).unroll(ahead) {
            let s = x.load4(at + i as u64);
            acc += (s.x * s.x + s.y * s.y) + (s.z * s.z + s.w * s.w);
        }
    }
    acc = sum32(acc);
    let by = clip_by(acc, threshold, eps);
    if CACHED {
        if col < width {
            out.store4(
                at + col as u64,
                F32x4::new(v.x * by, v.y * by, v.z * by, v.w * by),
            );
        }
    } else {
        for i in (col..width).by(LANES as u32 * WIDE).unroll(ahead) {
            let s = x.load4(at + i as u64);
            out.store4(
                at + i as u64,
                F32x4::new(s.x * by, s.y * by, s.z * by, s.w * by),
            );
        }
    }
}

/// `L2Clip` with a SIMD group per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    l2_clip_lanes::<WIDE>(AHEAD),
    variant = self.cached(),
    name = "lane_l2_clip",
    threads = self.rows * LANES,
    work = self.rows * self.width,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    load = Some(lane_load(self, body.trip())),
    unproven,
)]
struct LaneL2Clip {
    rows: usize,
    width: usize,
    threshold: f32,
    eps: f32,
}

impl LaneL2Clip {
    /// Whether one load per lane covers a row.
    fn cached(&self) -> bool {
        self.width <= LANES * WIDE as usize
    }
}

fn lane_load(k: &LaneL2Clip, trip: tessera_lower::Ops) -> Load {
    let all = 4.0 * (k.rows * k.width) as f64;
    let reads = match k.cached() {
        true => vec![all],
        false => vec![all, all],
    };
    Load {
        threads: k.rows * LANES,
        group: 256,
        trips: k.width.div_ceil(LANES * WIDE as usize) as f64,
        trip,
        reads,
        written: all,
        flight: (4 * WIDE * AHEAD) as f64,
        ..Load::default()
    }
}

/// The clip `site` names, one thread per row.
fn clip(site: &sites::Stage) -> Option<L2Clip> {
    let Stage::L2Clip { threshold, eps } = site.stage else {
        return None;
    };
    Some(L2Clip {
        rows: site.len / site.width,
        width: site.width,
        threshold,
        eps,
    })
}

stage_rule!("l2_clip", clip);

mod lanes {
    use super::*;

    fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let site: sites::Stage = tree.site(at)?;
        let L2Clip {
            rows,
            width,
            threshold,
            eps,
        } = clip(&site)?;
        if width < LANE_MIN || !width.is_multiple_of(WIDE as usize) {
            return None;
        }
        let kernel = LaneL2Clip {
            rows,
            width,
            threshold,
            eps,
        };
        Some(vec![Draft::new(kernel, site.reads(), site.len)])
    }

    tessera_lower::rule!("lane_l2_clip", propose);
}
