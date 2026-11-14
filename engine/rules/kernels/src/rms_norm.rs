//! RMS norm over rows, generated.

use crate::per_row;
use codegen::prelude::*;
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Shape, Source, Step, T, Tree};
use sim::Load;

/// One thread per row. Summing in order keeps batches alike.
#[kernel]
fn rms_norm<const WEIGHTED: bool>(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    eps: f32,
) {
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
        let gain = if WEIGHTED { w[i as u64] } else { 1.0 };
        out[at + i as u64] = x[at + i as u64] * inv * gain;
    }
}

/// One thread scales each row by its root mean square.
#[derive(Clone, Copy, Debug)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

impl RmsNorm {
    fn body(&self) -> codegen::ir::Body {
        match self.weighted {
            true => rms_norm_ir::<true>(),
            false => rms_norm_ir::<false>(),
        }
    }
}

impl Kernel for RmsNorm {
    fn name(&self) -> &'static str {
        match self.weighted {
            true => "rms_norm",
            false => "rms",
        }
    }

    /// Sums each row, then scales it.
    fn load(&self, _: usize) -> Option<Load> {
        let gain = if self.weighted {
            4.0 * self.width as f64
        } else {
            0.0
        };
        Some(per_row(self.rows, self.width, self.body().trip(), gain))
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::named(self.body(), format!("plan_{}", self.name()))
    }

    /// Without a weight, the row stands in for it, unread.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [
            Arg::count(self.rows),
            Arg::count(self.width),
            Arg::value(&self.eps),
        ];
        let w = step.read(usize::from(self.weighted));
        let buffers = [step.read(0), w, step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.rows),
        }]
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    let kernel = RmsNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
    };
    Some(vec![Draft::new(kernel, site.reads(), site.len)])
}

lower::rule!("rms_norm", propose);
