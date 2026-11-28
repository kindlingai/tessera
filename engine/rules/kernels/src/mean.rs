//! The mean or weighted sum over one axis.

use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Draft, Extent, Kernel, Source, T, Term, Tree};

/// Elements handled per thread on the vectorized path.
const WIDE: usize = 4;

/// Sums `count` strided values, in order. `w` weights each.
#[kernel]
#[allow(clippy::too_many_arguments)]
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
    #[build] wide: bool,
) {
    if wide {
        let inner4 = inner / 4;
        if gid() >= outer * inner4 {
            return;
        }
        let o = gid() / inner4;
        let i = (gid() % inner4) * 4;
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        for t in 0..count {
            let v = x.load4(((o * count + t) * inner + i) as u64);
            if WEIGHTED {
                let wt = w[(o * count + t) as u64];
                s0 += v.x * wt;
                s1 += v.y * wt;
                s2 += v.z * wt;
                s3 += v.w * wt;
            } else {
                s0 += v.x;
                s1 += v.y;
                s2 += v.z;
                s3 += v.w;
            }
        }
        let mut out4 = F32x4::zero();
        if WEIGHTED {
            out4.x = s0;
            out4.y = s1;
            out4.z = s2;
            out4.w = s3;
        } else {
            let denom = count as f32;
            out4.x = s0 / denom;
            out4.y = s1 / denom;
            out4.z = s2 / denom;
            out4.w = s3 / denom;
        }
        out.store4((gid() * 4) as u64, out4);
    } else {
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
}

/// `x` is `[outer, count, inner]`. The middle axis goes.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    mean(self.x_form, self.w_form, self.out_form, self.inner.is_multiple_of(WIDE)),
    variant = self.weighted,
    name = match self.weighted {
        true => "mix",
        false => "mean",
    },
    threads = if self.inner.is_multiple_of(WIDE) {
        self.outer.times(self.inner / WIDE)
    } else {
        self.outer.times(self.inner)
    },
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
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == k.w_form.name(),
        _ => false,
    }
}

fn mean_emits(k: &Mean, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn mean_source(k: &Mean) -> Source {
    let wide = k.inner.is_multiple_of(WIDE);
    let body = match k.weighted {
        true => mean_ir::<true>(k.x_form, k.w_form, k.out_form, wide),
        false => mean_ir::<false>(k.x_form, k.w_form, k.out_form, wide),
    };
    let base = if k.weighted { "mix" } else { "mean" };
    let tag = if wide { "_x4" } else { "" };
    let entry = match (k.x_form.name(), k.w_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_{base}{tag}"),
        (xf, wf, of) => format!("plan_{base}{tag}_{xf}_{wf}_{of}"),
    };
    Source::named(body, entry)
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
    let before = &x.extents[..axis];
    let outer = match before {
        [one] => *one,
        _ => Extent::Fixed(x.shape[..axis].iter().product()),
    };
    let inner = x.shape[axis + 1..].iter().product();
    let x_form = plain(x.form);
    let w_form = weights.map_or(x_form, |wt| plain(tree.node(wt).form));
    let out_form = writable(node.form);
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
