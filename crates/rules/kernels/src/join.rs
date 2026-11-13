//! Joins rows of two inputs along the last axis.

use lower::{Arg, Extent, Kernel, Launch, Shape, Source, Step, Text, float};

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug)]
pub struct Join {
    pub left: usize,
    pub right: usize,
    /// The row count. Decode reads it at launch.
    pub rows: Extent,
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
