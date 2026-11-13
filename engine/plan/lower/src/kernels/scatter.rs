//! Writes rows into kept rows by slot and position.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use codegen::kept::{self, Layout};
use codegen::prelude::*;

/// Row `r` lands in its slot's kept row.
#[kernel]
fn scatter(
    x: Buf<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
) {
    if gid() >= rows * width {
        return;
    }
    let r = gid() / width;
    let j = gid() % width;
    let at = kept_row(slots[r as u64], positions[r as u64]) * width + j;
    out[at as u64] = x[gid() as u64];
}

/// `rows` of `width` elements into `slots` of `context` rows.
#[derive(Clone, Copy, Debug)]
pub struct Scatter {
    pub rows: usize,
    pub width: usize,
    pub slots: usize,
    pub context: usize,
    pub kept: &'static dyn Layout,
}

impl Kernel for Scatter {
    fn name(&self) -> &'static str {
        "scatter"
    }

    fn source(&self) -> Source {
        let body = kept::attach(scatter_ir(), self.kept);
        Source::named(body, format!("plan_scatter_{}", self.kept.name()))
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.rows, self.width].map(Arg::count);
        let values = self.kept.values(self.slots, self.context);
        let kept = values.into_iter().map(|v| Arg::count(v as usize));
        let args = [step.read(0), step.read(1), step.read(2), step.out()];
        vec![Launch {
            args: args.into_iter().chain(sizes).chain(kept).collect(),
            shape: Shape::Line(self.rows * self.width),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.rows * self.width) as f64
    }
}
