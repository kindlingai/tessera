//! Softmax over rows, generated.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::kernels::per_row;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use graph::Stage;
use sim::{Load, Ops};

/// One thread turns each row into probabilities.
#[derive(Clone, Copy, Debug)]
pub struct Softmax {
    pub rows: usize,
    pub width: usize,
}

const BODY: &str = "    if (gid >= rows) {
        return;
    }
    u64 at = (u64)gid * width;
    float top = x[at];
    for (u32 i = 1; i < width; i++) {
        top = f_max(top, x[at + i]);
    }
    float sum = 0.0f;
    for (u32 i = 0; i < width; i++) {
        sum += f_exp(x[at + i] - top);
    }
    for (u32 i = 0; i < width; i++) {
        out[at + i] = f_exp(x[at + i] - top) / sum;
    }
";

impl Kernel for Softmax {
    fn name(&self) -> &'static str {
        "softmax"
    }

    /// Finds each row's maximum, then takes two exponentials.
    fn load(&self, _: &Step) -> Option<Load> {
        let (float, special, loads) = (3.0, 3.0, 3.0);
        let trip = Ops {
            float,
            special,
            loads,
            stores: 1.0,
            fresh: 1.0 / 16.0,
            int: 0.0,
        };
        Some(per_row(self.rows, self.width, trip, 0.0))
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.rows, self.width]
    }

    fn source(&self) -> Source {
        Source::Made(Text {
            entry: "plan_softmax".into(),
            buffers: vec![float("x", false), float("out", true)],
            scalars: vec![("rows", "u32"), ("width", "u32")],
            body: BODY.into(),
        })
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

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Stage {
        stage: Stage::Softmax,
        x,
        len,
        width,
        ..
    } = *site
    else {
        return None;
    };
    let kernel = Softmax {
        rows: len / width,
        width,
    };
    Some(vec![Draft::new(kernel, [x], len)])
}

crate::rule!("softmax", propose);
