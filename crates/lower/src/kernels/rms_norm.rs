//! RMS norm over rows, generated.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::kernels::per_row;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use crate::tree::{T, Tree};
use graph::Stage;
use sim::{Load, Ops};

/// One thread scales each row by its root mean square.
#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
}

// Summed in order, so batch size never changes a row.
const BODY: &str = "    if (gid >= rows) {
        return;
    }
    u64 at = (u64)gid * width;
    float sum = 0.0f;
    for (u32 i = 0; i < width; i++) {
        sum += x[at + i] * x[at + i];
    }
    float inv = f_rsqrt(sum / (float)width + eps);
    for (u32 i = 0; i < width; i++) {
        out[at + i] = x[at + i] * inv * w[i];
    }
";

impl Kernel for RmsNorm {
    fn name(&self) -> &'static str {
        "rms_norm"
    }

    /// Sums each row, then scales it.
    fn load(&self, _: usize) -> Option<Load> {
        let trip = Ops {
            float: 3.0,
            loads: 3.0,
            stores: 1.0,
            fresh: 1.0 / 16.0,
            ..Ops::default()
        };
        let gain = 4.0 * self.width as f64;
        Some(per_row(self.rows, self.width, trip, gain))
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::Made(Text {
            entry: "plan_rms_norm".into(),
            buffers: vec![float("x", false), float("w", false), float("out", true)],
            scalars: vec![("rows", "u32"), ("width", "u32"), ("eps", "float")],
            body: BODY.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [
            Arg::count(self.rows),
            Arg::count(self.width),
            Arg::F32(self.eps),
        ];
        let buffers = [step.read(0), step.read(1), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.rows),
        }]
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
    let kernel = RmsNorm {
        rows: len / width,
        width,
        eps: *eps,
    };
    Some(vec![Draft::new(kernel, [x, w], len)])
}

crate::rule!("rms_norm", propose);
