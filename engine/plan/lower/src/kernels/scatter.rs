//! Writes rows into kept rows by slot and position.

use crate::Kernel;
use crate::abi::{Arg, Launch, Shape, Source};
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
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "scatter",
    sizes = [],
    work = self.rows * self.width,
    source = scatter_source,
    launches = scatter_launches,
)]
pub struct Scatter {
    pub rows: usize,
    pub width: usize,
    pub slots: usize,
    pub context: usize,
    pub kept: &'static dyn Layout,
}

fn scatter_source(k: &Scatter) -> Source {
    let body = kept::attach(scatter_ir(), k.kept);
    Source::named(body, format!("plan_scatter_{}", k.kept.name()))
}

fn scatter_launches(k: &Scatter, step: &Step) -> Vec<Launch> {
    let sizes = [k.rows, k.width].map(Arg::count);
    let values = k.kept.values(k.slots, k.context);
    let kept = values.into_iter().map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.read(2), step.out()];
    vec![Launch {
        args: args.into_iter().chain(sizes).chain(kept).collect(),
        shape: Shape::Line(k.rows * k.width),
    }]
}
