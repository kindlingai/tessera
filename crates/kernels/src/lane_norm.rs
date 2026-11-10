//! RMS norm in three short launches, summing in lanes.

use graph::{Order, Stage};
use lower::{Arg, Draft, Kernel, Launch, Read, Shape, Site, Source, Step, T, Text, Tree, float};
use sim::{Load, Ops};

/// Elements one lane squares and adds by itself.
const CHUNK: usize = 16;

/// Which of the norm's launches a kernel is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// Each lane's sum of squares over part of a row.
    Squares,
    /// Each row's lanes added, then inverted to a factor.
    Root,
    /// Every element times its row's factor and its column's gain.
    Scale,
}

/// One launch of an RMS norm over `rows` of `width`.
#[derive(Clone, Copy, Debug)]
pub struct LaneNorm {
    pub pass: Pass,
    pub rows: usize,
    pub width: usize,
    pub lanes: usize,
    pub eps: f32,
}

// Lanes are runs of a row, so rows sum alike.
const SQUARES: &str = "    if (gid >= rows * lanes) {
        return;
    }
    u32 lane = gid % lanes;
    u64 at = (u64)(gid / lanes) * width;
    u32 from = (u32)((u64)width * lane / lanes);
    u32 to = (u32)((u64)width * (lane + 1) / lanes);
    float acc = 0.0f;
    for (u32 i = from; i < to; i++) {
        acc += x[at + i] * x[at + i];
    }
    out[gid] = acc;
";

const ROOT: &str = "    if (gid >= rows) {
        return;
    }
    u64 at = (u64)gid * lanes;
    float acc = 0.0f;
    for (u32 l = 0; l < lanes; l++) {
        acc += parts[at + l];
    }
    out[gid] = f_rsqrt(acc / (float)width + eps);
";

const SCALE: &str = "    if (gid >= rows * width) {
        return;
    }
    out[gid] = x[gid] * inv[gid / width] * w[gid % width];
";

impl LaneNorm {
    /// Its name, its source, and the buffers it reads.
    fn parts(&self) -> (&'static str, &'static str, &'static [&'static str]) {
        match self.pass {
            Pass::Squares => ("norm_squares", SQUARES, &["x"]),
            Pass::Root => ("norm_root", ROOT, &["parts"]),
            Pass::Scale => ("norm_scale", SCALE, &["x", "inv", "w"]),
        }
    }

    /// How many threads it starts, and each one's loop trips.
    fn threads(&self) -> (usize, usize) {
        match self.pass {
            Pass::Squares => (self.rows * self.lanes, self.width / self.lanes),
            Pass::Root => (self.rows, self.lanes),
            Pass::Scale => (self.rows * self.width, 1),
        }
    }
}

impl Kernel for LaneNorm {
    fn name(&self) -> &'static str {
        self.parts().0
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    /// Rows sum alike at any count. Other kernels differ.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, len: usize) -> Option<Load> {
        let (threads, trips) = self.threads();
        let scale = self.pass == Pass::Scale;
        let loads = if scale { 3.0 } else { 1.0 };
        let trip = Ops {
            float: if scale { 2.0 } else { 1.0 },
            loads,
            stores: f64::from(u8::from(scale)),
            fresh: if scale { 1.0 } else { 1.0 / 16.0 },
            ..Ops::default()
        };
        Some(Load {
            threads,
            group: 256,
            trips: trips as f64,
            trip,
            reads: vec![4.0 * (threads * trips) as f64],
            written: 4.0 * len as f64,
            flight: if scale { 12.0 } else { 64.0 },
        })
    }

    fn source(&self) -> Source {
        let (entry, body, reads) = self.parts();
        let mut buffers: Vec<_> = reads.iter().map(|name| float(name, false)).collect();
        buffers.push(float("out", true));
        let scalars = [("rows", "u32"), ("width", "u32"), ("lanes", "u32")];
        Source::Made(Text {
            entry: entry.into(),
            buffers,
            scalars: scalars.into_iter().chain([("eps", "float")]).collect(),
            body: body.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.rows, self.width, self.lanes].map(Arg::count));
        args.push(Arg::F32(self.eps));
        vec![Launch {
            args,
            shape: Shape::Line(self.threads().0),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        let (threads, trips) = self.threads();
        (threads * trips) as f64
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
    let (rows, lanes) = (len / width, (width / CHUNK).clamp(1, 256));
    let pass = |pass| LaneNorm {
        pass,
        rows,
        width,
        lanes,
        eps: *eps,
    };
    let scaled = [Read::Node(x), Read::Draft(1), Read::Node(w)];
    Some(vec![
        Draft::new(pass(Pass::Squares), [x], rows * lanes),
        Draft::new(pass(Pass::Root), [Read::Draft(0)], rows),
        Draft::new(pass(Pass::Scale), scaled, len),
    ])
}

lower::rule!("lane_norm", propose);
