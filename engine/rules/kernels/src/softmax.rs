//! Softmax over rows, generated.

use crate::per_row;
use codegen::prelude::*;
use graph::Stage;
use lower::sites;
use lower::{Draft, Kernel, T, Tree};
use sim::Load;

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
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(softmax, threads = self.rows, load = softmax_load)]
pub struct Softmax {
    pub rows: usize,
    pub width: usize,
}

/// Finds each row's maximum, then takes two exponentials.
fn softmax_load(k: &Softmax, _: usize) -> Option<Load> {
    Some(per_row(k.rows, k.width, softmax_ir().trip(), 0.0))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Stage {
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
