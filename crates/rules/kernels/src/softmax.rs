//! Softmax over rows, generated.

use crate::per_row;
use codegen::prelude::*;
use graph::Stage;
use lower::{Arg, Draft, Kernel, Launch, Shape, Site, Source, Step, T, Tree};
use sim::{Load, Ops};

/// One thread per row: peak, sum, then the probabilities.
#[kernel]
fn softmax(x: Buf<F32>, mut out: BufMut<F32>, rows: u32, width: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut top = x[at];
    for i in 1..width {
        top = max(top, x[at + i as u64]);
    }
    let mut sum = 0.0f32;
    for i in 0..width {
        sum += exp(x[at + i as u64] - top);
    }
    for i in 0..width {
        out[at + i as u64] = exp(x[at + i as u64] - top) / sum;
    }
}

/// One thread turns each row into probabilities.
#[derive(Clone, Copy, Debug)]
pub struct Softmax {
    pub rows: usize,
    pub width: usize,
}

impl Kernel for Softmax {
    fn name(&self) -> &'static str {
        "softmax"
    }

    /// Finds each row's maximum, then takes two exponentials.
    fn load(&self, _: usize) -> Option<Load> {
        let (float, special, loads) = (3.0, 3.0, 3.0);
        let trip = Ops {
            float,
            special,
            loads,
            stores: 1.0,
            fresh: 1.0 / 16.0,
            int: 0.0,
            width: 0.0,
            tile_flops: 0.0,
        };
        Some(per_row(self.rows, self.width, trip, 0.0))
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        let mut text = softmax_ir().text();
        text.entry = "plan_softmax".into();
        Source::Made(text)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.rows, self.width].map(Arg::count);
        let buffers = [step.read(0), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.rows),
        }]
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Stage {
        stage: Stage::Softmax,
        x,
        len,
        width,
        ..
    } = tree.site(at)?
    else {
        return None;
    };
    let kernel = Softmax {
        rows: len / width,
        width,
    };
    Some(vec![Draft::new(kernel, [x], len)])
}

lower::rule!("softmax", propose);
