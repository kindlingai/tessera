//! Rows less their mean, or over their sum, generated.

use crate::{AHEAD, WIDE, io_name, io_pair, per_lane_row, per_row};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Draft, Kernel, Needs, Source, T, Tree, knob, sites};

knob!("lane_sum", "ahead", 1, 16);

/// One thread per row: the sum, then each element.
#[kernel]
fn by_sum<const DIVIDE: bool>(
    #[form(in_form)] x: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    #[build] in_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
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
fn sum_lanes<const WIDE: u32, const DIVIDE: bool>(
    #[form(in_form)] x: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    #[build] ahead: u32,
    #[build] in_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let row = gid() / LANES as u32;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let mut sum = 0.0f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(ahead) {
        let v = x.load4(at + i as u64);
        sum += (v.x + v.y) + (v.z + v.w);
    }
    sum = sum32(sum);
    let mean = sum / width as f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(ahead) {
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
    by_sum(self.in_form, self.out_form),
    variant = self.divide,
    name = match self.divide {
        true => "unit_sum",
        false => "center",
    },
    threads = self.rows,
    accepts = by_sum_accepts,
    emits = by_sum_emits,
    source = by_sum_source,
    load = Some(per_row(self.rows, self.width, body.trip(), 0.0)),
)]
pub(crate) struct BySum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
    pub in_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn by_sum_accepts(k: &BySum, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.in_form.name()
}

fn by_sum_emits(k: &BySum, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn by_sum_source(k: &BySum) -> Source {
    let base = if k.divide { "unit_sum" } else { "center" };
    let body = if k.divide {
        by_sum_ir::<true>(k.in_form, k.out_form)
    } else {
        by_sum_ir::<false>(k.in_form, k.out_form)
    };
    Source::named(body, io_name(base, k.in_form, k.out_form))
}

/// `BySum` with a SIMD group per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    sum_lanes::<WIDE>(self.ahead, self.in_form, self.out_form),
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
    accepts = lane_sum_accepts,
    emits = lane_sum_emits,
    source = lane_sum_source,
    load = Some(per_lane_row(self.rows, self.width, body.trip(), 0.0)),
    unproven,
)]
pub(crate) struct LaneSum {
    pub rows: usize,
    pub width: usize,
    pub divide: bool,
    pub ahead: u32,
    pub in_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn lane_sum_accepts(k: &LaneSum, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.in_form.name()
}

fn lane_sum_emits(k: &LaneSum, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn lane_sum_source(k: &LaneSum) -> Source {
    let base = if k.divide {
        "lane_unit_sum"
    } else {
        "lane_center"
    };
    let body = if k.divide {
        sum_lanes_ir::<WIDE, true>(k.ahead, k.in_form, k.out_form)
    } else {
        sum_lanes_ir::<WIDE, false>(k.ahead, k.in_form, k.out_form)
    };
    Source::named(body, io_name(base, k.in_form, k.out_form))
}

/// The row stage `site` names, if it sums its rows.
fn summed(tree: &Tree, at: T, site: &sites::Stage) -> Option<BySum> {
    let divide = match site.stage {
        Stage::Center => false,
        Stage::UnitSum => true,
        _ => return None,
    };
    let (in_form, out_form) = io_pair(tree.node(site.x).form, tree.node(at).form);
    Some(BySum {
        rows: site.len / site.width,
        width: site.width,
        divide,
        in_form,
        out_form,
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let kernel = summed(tree, at, &site)?;
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("by_sum", propose);

mod lanes {
    use super::*;

    fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let site: sites::Stage = tree.site(at)?;
        let BySum {
            rows,
            width,
            divide,
            in_form,
            out_form,
        } = summed(tree, at, &site)?;
        if !width.is_multiple_of(WIDE as usize) {
            return None;
        }
        let ahead = tree
            .knobs
            .get("lane_sum", "ahead")
            .map_or(AHEAD, |v| v as u32);
        let kernel = LaneSum {
            rows,
            width,
            divide,
            ahead,
            in_form,
            out_form,
        };
        Some(vec![Draft::new(kernel, site.padded(), site.len)])
    }

    tessera_lower::rule!("lane_sum", propose);
}
