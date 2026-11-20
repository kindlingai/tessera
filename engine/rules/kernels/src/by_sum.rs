//! Rows less their mean, or over their sum, generated.

use crate::{AHEAD, WIDE, per_lane_row, per_row};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::sites;
use tessera_lower::{Kernel, Needs, stage_rule};

/// One thread per row: the sum, then each element.
#[kernel]
fn by_sum<const DIVIDE: bool>(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += x[at + i as u64];
    }
    let mean = sum / width as f32;
    for i in 0..width {
        if DIVIDE {
            out[at + i as u64] = x[at + i as u64] / sum;
        } else {
            out[at + i as u64] = x[at + i as u64] - mean;
        }
    }
}

/// A SIMD group per row, summed by shuffles.
#[kernel]
fn sum_lanes<const WIDE: u32, const AHEAD: u32, const DIVIDE: bool>(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
) {
    let row = gid() / LANES as u32;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        sum += (v.x + v.y) + (v.z + v.w);
    }
    sum = sum32(sum);
    let mean = sum / width as f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        let mut o = F32x4::zero();
        if DIVIDE {
            o.x = v.x / sum;
            o.y = v.y / sum;
            o.z = v.z / sum;
            o.w = v.w / sum;
        } else {
            o.x = v.x - mean;
            o.y = v.y - mean;
            o.z = v.z - mean;
            o.w = v.w - mean;
        }
        out.store4(at + i as u64, o);
    }
}

/// One thread per row: centered, or scaled to unit sum.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    by_sum,
    variant = self.divide,
    name = match self.divide {
        true => "unit_sum",
        false => "center",
    },
    threads = self.rows,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub(crate) struct BySum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
}

/// `BySum` with a SIMD group per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    sum_lanes::<WIDE, AHEAD>,
    variant = self.divide,
    name = match self.divide {
        true => "lane_unit_sum",
        false => "lane_center",
    },
    threads = self.rows * LANES,
    work = self.rows * self.width,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    load = Some(per_lane_row(self.rows, self.width, body.trip(), 0.0)),
    unproven,
)]
pub(crate) struct LaneSum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
}

/// The row stage `site` names, if it sums its rows.
fn summed(site: &sites::Stage) -> Option<BySum> {
    let divide = match site.stage {
        Stage::Center => false,
        Stage::UnitSum => true,
        _ => return None,
    };
    Some(BySum {
        rows: site.len / site.width,
        width: site.width,
        divide,
    })
}

stage_rule!("by_sum", summed);

mod lanes {
    use super::*;

    // Wide loads need whole groups of four along a row.
    stage_rule!(
        "lane_sum",
        |site| summed(site)
            .filter(|s| s.width.is_multiple_of(WIDE as usize))
            .map(
                |BySum {
                     rows,
                     width,
                     divide,
                 }| LaneSum {
                    rows,
                    width,
                    divide
                }
            ),
        |site| site.padded()
    );
}
