//! RMS norm over rows, generated.

use crate::per_row;
use codegen::prelude::*;
use graph::Stage;
use lower::{Arg, Draft, Kernel, Launch, Shape, Site, Source, Step, T, Tree};
use sim::{Load, Ops};

/// One thread per row. Summing in order keeps batches alike.
#[kernel]
fn rms_norm(x: Buf<F32>, w: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32, eps: f32) {
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
        out[at + i as u64] = x[at + i as u64] * inv * w[i as u64];
    }
}

/// One thread scales each row by its root mean square.
#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
}

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
        let mut text = rms_norm_ir().text();
        text.entry = "plan_rms_norm".into();
        Source::Made(text)
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
        eps,
    };
    Some(vec![Draft::new(kernel, [x, w], len)])
}

lower::rule!("rms_norm", propose);
