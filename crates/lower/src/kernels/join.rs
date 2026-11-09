//! Joins rows of two inputs along the last axis.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::plan::Step;

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug)]
pub struct Join {
    pub left: usize,
    pub right: usize,
}

const BODY: &str = "    u32 width = left + right;
    u32 row = gid / width;
    u32 col = gid % width;
    if (row >= rows) {
        return;
    }
    out[gid] = col < left ? a[row * left + col] : b[row * right + col - left];
";

impl Kernel for Join {
    fn name(&self) -> &'static str {
        "join"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.left, self.right]
    }

    fn source(&self) -> Source {
        Source::Made(Text {
            entry: "plan_join".into(),
            buffers: vec![float("a", false), float("b", false), float("out", true)],
            scalars: vec![("left", "u32"), ("right", "u32"), ("rows", "u32")],
            body: BODY.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let len = step.write.len;
        let rows = len / (self.left + self.right);
        let sizes = [self.left, self.right, rows].map(Arg::count);
        let buffers = [step.read(0), step.read(1), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(len),
        }]
    }
}
