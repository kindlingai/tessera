//! RMS norm with one warp per row, summed by shuffles.

use codegen::{Gen, LANES};
use graph::{DType, Order, Stage};
use lower::{Arg, Draft, Kernel, Launch, Load, Shape, Site, Source, Step, T, Tree};

/// Elements one wide load brings a lane.
const WIDE: usize = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: usize = 4;

/// An RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug)]
pub struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
}

impl LaneNorm {
    /// Sums a row's squares, then scales the row.
    fn write(&self) -> Gen {
        let mut g = Gen::new("norm_lanes");
        g.floats("x", false);
        g.floats("w", false);
        g.floats("out", true);
        g.scalar("rows", "u32");
        g.scalar("width", "u32");
        g.scalar("eps", "float");
        let lane = g.lane();
        g.line(format!("u32 row = gid / {LANES};"));
        g.guard("row >= rows");
        g.line("u64 at = (u64)row * width;");
        g.line("float acc = 0.0f;");
        let (from, step) = (format!("{lane} * {WIDE}"), (LANES * WIDE).to_string());
        g.repeat("i", &from, "width", &step, AHEAD, |g| {
            let x = g.four(DType::F32, "x + at + i");
            for x in &x {
                g.fma("acc", x, x);
            }
        });
        g.sum32("acc");
        let inv = g.rsqrt("acc / (float)width + eps");
        g.line(format!("float inv = {inv};"));
        g.repeat("i", &from, "width", &step, AHEAD, |g| {
            let x = g.four(DType::F32, "x + at + i");
            let w = g.four(DType::F32, "w + i");
            g.line("float4 o;");
            for (h, (x, w)) in ["x", "y", "z", "w"].iter().zip(x.iter().zip(&w)) {
                g.arith(format!("o.{h} = {x} * inv * {w};"), 2.0);
            }
            g.store4("out + at + i", "o");
        });
        g
    }
}

impl Kernel for LaneNorm {
    fn name(&self) -> &'static str {
        "lane_norm"
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
            trips: (self.width / (LANES * WIDE)) as f64,
            trip: self.write().trip(),
            // The row is read twice, the gains once.
            reads: vec![8.0 * elements, 4.0 * elements],
            written: 4.0 * len as f64,
            flight: (4 * WIDE * AHEAD) as f64,
        })
    }

    fn source(&self) -> Source {
        Source::Made(self.write().finish())
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.rows, self.width].map(Arg::count));
        args.push(Arg::F32(self.eps));
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
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Stage {
        stage: Stage::RmsNorm { eps },
        x,
        operand: Some((w, _)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    // Wide loads need whole groups of four along a row.
    if !width.is_multiple_of(WIDE) {
        return None;
    }
    let norm = LaneNorm {
        rows: len / width,
        width,
        eps: *eps,
    };
    Some(vec![Draft::new(norm, [x, w], len)])
}

lower::rule!("lane_norm", propose);
