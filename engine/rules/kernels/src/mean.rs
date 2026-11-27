//! The mean or weighted sum over one axis.

use crate::{io_name, io_pair};
use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::form::{Load as Stored, plain};
use tessera_lower::{Draft, Extent, Kernel, Source, T, Term, Tree};

/// Sums `count` strided values, in order. `w` weights each.
#[kernel]
fn mean<const WEIGHTED: bool>(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    outer: u32,
    count: u32,
    inner: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() >= outer * inner {
        return;
    }
    let o = gid() / inner;
    let i = gid() % inner;
    let mut sum = 0.0f32;
    for t in 0..count {
        let mut v = x[((o * count + t) * inner + i) as u64];
        if WEIGHTED {
            v = v * w[(o * count + t) as u64];
        }
        sum += v;
    }
    if WEIGHTED {
        out[gid() as u64] = sum;
    } else {
        out[gid() as u64] = sum / count as f32;
    }
}

/// `x` is `[outer, count, inner]`. The middle axis goes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    mean(self.x_form, self.w_form, self.out_form),
    variant = self.weighted,
    name = match self.weighted {
        true => "mix",
        false => "mean",
    },
    threads = self.outer.times(self.inner),
    work = self.outer.most() * self.count.most() * self.inner,
    accepts = mean_accepts,
    emits = mean_emits,
    source = mean_source,
)]
pub struct Mean {
    pub outer: Extent,
    pub count: Extent,
    pub inner: usize,
    /// Weights from the second read, else the plain mean.
    pub weighted: bool,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn mean_accepts(k: &Mean, input: usize, form: &dyn Stored) -> bool {
    let want = if input == 0 { k.x_form } else { k.w_form };
    form.name() == want.name()
}

fn mean_emits(k: &Mean, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn mean_source(k: &Mean) -> Source {
    let base = if k.weighted { "mix" } else { "mean" };
    let body = if k.weighted {
        mean_ir::<true>(k.x_form, k.w_form, k.out_form)
    } else {
        mean_ir::<false>(k.x_form, k.w_form, k.out_form)
    };
    let name = if k.w_form.name() == "f32" {
        io_name(base, k.x_form, k.out_form)
    } else {
        format!(
            "plan_{base}_{}_{}_{}",
            k.x_form.name(),
            k.w_form.name(),
            k.out_form.name()
        )
    };
    Source::named(body, name)
}

/// Axes before the mean fold into one, those after too.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let (axis, weights) = match node.term {
        Term::Op(Op::Mean(axis)) => (axis, None),
        Term::Op(Op::Mix(axis)) => (axis, node.args.get(1).copied()),
        _ => return None,
    };
    let axis = axis as usize;
    let x = tree.node(node.args[0]);
    let (x_form, out_form) = io_pair(x.form, node.form);
    let w_form = weights.map_or(x_form, |w| plain(tree.node(w).form));
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
        weighted: weights.is_some(),
        x_form,
        w_form,
        out_form,
    };
    // The plain mean reads `x` again, unused.
    let reads = [node.args[0], weights.unwrap_or(node.args[0])];
    Some(vec![Draft::new(mean, reads, outer.most() * inner)])
}

tessera_lower::rule!("mean", propose);
