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
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = rms_name(self),
    source = rms_source,
    launches = rms_launches,
    load = rms_load,
)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

fn rms_name(k: &RmsNorm) -> &'static str {
    match k.weighted {
        true => "rms_norm",
        false => "rms",
    }
}

fn rms_body(k: &RmsNorm) -> codegen::ir::Body {
    match k.weighted {
        true => rms_norm_ir::<true>(),
        false => rms_norm_ir::<false>(),
    }
}

fn rms_source(k: &RmsNorm) -> Source {
    Source::named(rms_body(k), format!("plan_{}", k.name()))
}

/// Sums each row, then scales it.
fn rms_load(k: &RmsNorm, _: usize) -> Option<Load> {
    let gain = if k.weighted {
        4.0 * k.width as f64
    } else {
        0.0
    };
    Some(per_row(k.rows, k.width, rms_body(k).trip(), gain))
}

/// Without a weight, the row stands in for it, unread.
fn rms_launches(k: &RmsNorm, step: &Step) -> Vec<Launch> {
    let sizes = [Arg::count(k.rows), Arg::count(k.width), Arg::value(&k.eps)];
    let w = step.read(usize::from(k.weighted));
    let buffers = [step.read(0), w, step.out()];
    vec![Launch {
        args: buffers.into_iter().chain(sizes).collect(),
        shape: Shape::Line(k.rows),
    }]
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
