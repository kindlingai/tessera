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
#[derive(Clone, Copy, Debug)]
pub struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

impl LaneNorm {
    fn body(&self) -> codegen::ir::Body {
        let mut body = match self.weighted {
            true => norm_lanes_ir::<WIDE, AHEAD, true>(),
            false => norm_lanes_ir::<WIDE, AHEAD, false>(),
        };
        body.entry = self.name().into();
        body
    }
}

impl Kernel for LaneNorm {
    fn name(&self) -> &'static str {
        match self.weighted {
            true => "lane_norm",
            false => "lane_rms",
        }
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    /// Rows sum alike at any count. Other kernels differ.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, len: usize) -> Option<Load> {
        let elements = (self.rows * self.width) as f64;
        Some(Load {
            threads: self.rows * LANES,
            group: 256,
            trips: (self.width / (LANES * WIDE as usize)) as f64,
            trip: self.body().trip(),
            // The row is read twice, the gains once.
            reads: vec![8.0 * elements, 4.0 * elements * f64::from(self.weighted)],
            written: 4.0 * len as f64,
            flight: (4 * WIDE * AHEAD) as f64,
        })
    }

    fn source(&self) -> Source {
        Source::Made(self.body())
    }

    /// Without a weight, the row stands in for it, unread.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let w = step.read(usize::from(self.weighted));
        let mut args = vec![step.read(0), w, step.out()];
        args.extend([self.rows, self.width].map(Arg::count));
        args.push(Arg::value(&self.eps));
        vec![Launch {
            args,
            shape: Shape::Line(self.rows * LANES),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.rows * self.width) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }

    fn needs(&self) -> Needs {
        Needs {
            lanes: LANES,
            group_threads: 256,
            ..Needs::default()
        }
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    // Wide loads need whole groups of four along a row.
    if !site.width.is_multiple_of(WIDE as usize) {
        return None;
    }
    let norm = LaneNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
    };
    Some(vec![Draft::new(norm, site.reads(), site.len)])
}

lower::rule!("lane_norm", propose);
