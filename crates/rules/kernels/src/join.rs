//! Joins rows of two inputs along the last axis.

use codegen::prelude::*;
use lower::{Arg, Extent, Kernel, Launch, Shape, Source, Step};

/// One thread per output element, from whichever side holds it.
#[kernel]
fn join(a: Buf<F32>, b: Buf<F32>, mut out: BufMut<F32>, left: u32, right: u32, rows: u32) {
    let width = left + right;
    let row = gid() / width;
    let col = gid() % width;
    if row >= rows {
        return;
    }
    let v = if col < left {
        a[(row * left + col) as u64]
    } else {
        b[(row * right + col - left) as u64]
    };
    out[gid() as u64] = v;
}

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug)]
pub struct Join {
    pub left: usize,
    pub right: usize,
    /// The row count. Decode reads it at launch.
    pub rows: Extent,
}

impl Kernel for Join {
    fn name(&self) -> &'static str {
        "join"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.left, self.right]
    }

    fn source(&self) -> Source {
        let mut text = join_ir().text();
        text.entry = "plan_join".into();
        Source::Made(text)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.left, self.right].map(Arg::count);
        let buffers = [step.read(0), step.read(1), step.out()];
        let args = buffers
            .into_iter()
            .chain(sizes)
            .chain([Arg::size(self.rows)]);
        vec![Launch {
            args: args.collect(),
            shape: Shape::line(self.rows.times(self.left + self.right)),
        }]
    }
}
