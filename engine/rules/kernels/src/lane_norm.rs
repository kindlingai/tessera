//! RMS norm, a SIMD group per row, summed by shuffles.

use codegen::prelude::*;
use graph::Order;
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree};

/// Elements one wide load brings a lane.
const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: u32 = 4;

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
    name = lane_name(self),
    work = self.rows * self.width,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    source = Source::Made(lane_body(self)),
    launches = lane_launches,
    load = lane_load,
    unproven,
)]
pub struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

fn lane_name(k: &LaneNorm) -> &'static str {
    match k.weighted {
        true => "lane_norm",
        false => "lane_rms",
    }
}

fn lane_body(k: &LaneNorm) -> codegen::ir::Body {
    let mut body = match k.weighted {
        true => norm_lanes_ir::<WIDE, AHEAD, true>(),
        false => norm_lanes_ir::<WIDE, AHEAD, false>(),
    };
    body.entry = k.name().into();
    body
}

fn lane_load(k: &LaneNorm, len: usize) -> Option<Load> {
    let elements = (k.rows * k.width) as f64;
    Some(Load {
        threads: k.rows * LANES,
        group: 256,
        trips: (k.width / (LANES * WIDE as usize)) as f64,
        trip: lane_body(k).trip(),
        // The row is read twice, the gains once.
        reads: vec![8.0 * elements, 4.0 * elements * f64::from(k.weighted)],
        written: 4.0 * len as f64,
        flight: (4 * WIDE * AHEAD) as f64,
    })
}

/// Without a weight, the row stands in for it, unread.
fn lane_launches(k: &LaneNorm, step: &Step) -> Vec<Launch> {
    let w = step.read(usize::from(k.weighted));
    let mut args = vec![step.read(0), w, step.out()];
    args.extend([k.rows, k.width].map(Arg::count));
    args.push(Arg::value(&k.eps));
    vec![Launch {
        args,
        shape: Shape::Line(k.rows * LANES),
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    let (rows, width) = (site.len / site.width, site.width);
    let norm = LaneNorm {
        rows,
        width,
        eps,
        weighted,
    };
    // Wide loads need whole groups of four along a row.
    let fits = width.is_multiple_of(WIDE as usize);
    fits.then(|| vec![Draft::new(norm, site.reads(), site.len)])
}

lower::rule!("lane_norm", propose);
