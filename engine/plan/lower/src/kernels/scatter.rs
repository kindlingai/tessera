//! Writes rows into kept rows by slot and position.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use codegen::prelude::*;

/// Row `r` goes to `slots[r] * stride + positions[r]`.
#[kernel]
fn scatter(
    x: Buf<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    stride: u32,
) {
    if gid() >= rows * width {
        return;
    }
    let r = gid() / width;
    let j = gid() % width;
    let at = (slots[r as u64] * stride + positions[r as u64]) * width + j;
    out[at as u64] = x[gid() as u64];
}

/// `rows` of `width` elements, `stride` rows a slot.
#[derive(Clone, Copy, Debug)]
pub struct Scatter {
    pub rows: usize,
    pub width: usize,
    pub stride: usize,
}

impl Kernel for Scatter {
    fn name(&self) -> &'static str {
        "scatter"
    }

    fn source(&self) -> Source {
        Source::named(scatter_ir(), "plan_scatter")
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.rows, self.width, self.stride].map(Arg::count);
        let args = [step.read(0), step.read(1), step.read(2), step.out()];
        vec![Launch {
            args: args.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.rows * self.width),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.rows * self.width) as f64
    }
}
