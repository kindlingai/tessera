//! The mean over one axis, a thread per output element.

use codegen::prelude::*;
use lower::{Arg, Draft, Extent, Kernel, Launch, Shape, Source, Step, T, Term, Tree};

/// Sums `count` strided values, in order, then divides.
#[kernel]
fn mean(x: Buf<F32>, mut out: BufMut<F32>, outer: u32, count: u32, inner: u32) {
    if gid() >= outer * inner {
        return;
    }
    let o = gid() / inner;
    let i = gid() % inner;
    let mut sum = 0.0f32;
    for t in 0..count {
        sum += x[((o * count + t) * inner + i) as u64];
    }
    out[gid() as u64] = sum / count as f32;
}

/// `x` is `[outer, count, inner]`. The middle axis goes.
#[derive(Clone, Copy, Debug)]
pub struct Mean {
    pub outer: Extent,
    pub count: Extent,
    pub inner: usize,
}

impl Kernel for Mean {
    fn name(&self) -> &'static str {
        "mean"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.outer.most(), self.count.most(), self.inner]
    }

    fn source(&self) -> Source {
        Source::named(mean_ir(), "plan_mean")
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [
            Arg::size(self.outer),
            Arg::size(self.count),
            Arg::count(self.inner),
        ];
        vec![Launch {
            args: [step.read(0), step.out()]
                .into_iter()
                .chain(sizes)
                .collect(),
            shape: Shape::line(self.outer.times(self.inner)),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.outer.most() * self.count.most() * self.inner) as f64
    }
}

/// Axes before the mean fold into one, those after too.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Mean(axis) = node.term else {
        return None;
    };
    let axis = axis as usize;
    let x = tree.node(node.args[0]);
    let before = &x.extents[..axis];
    let outer = match before {
        [one] => *one,
        _ => Extent::Fixed(x.shape[..axis].iter().product()),
    };
    let inner = x.shape[axis + 1..].iter().product();
    let mean = Mean {
        outer,
        count: x.extents[axis],
        inner,
    };
    Some(vec![Draft::new(mean, [node.args[0]], outer.most() * inner)])
}

lower::rule!("mean", propose);
