//! The mean over one axis, a thread per output element.

use codegen::prelude::*;
use graph::Op;
use lower::{Draft, Extent, Kernel, T, Term, Tree};

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
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(mean, threads = self.outer.times(self.inner), work = self.outer.most() * self.count.most() * self.inner)]
pub struct Mean {
    pub outer: Extent,
    pub count: Extent,
    pub inner: usize,
}

/// Axes before the mean fold into one, those after too.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Mean(axis)) = node.term else {
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
