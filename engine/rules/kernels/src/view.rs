//! Slices, repeats, swaps and joins, copied into a new buffer.

use sensai_codegen::prelude::*;
use sensai_graph::{Op, View};
use sensai_lower::form::F32_FORM;
use sensai_lower::kernels::Join;
use sensai_lower::{Draft, Extent, Home, Kernel, T, Term, Tree};

/// One thread per output `[outer, upper, mid, inner]`, strided reads.
#[kernel]
fn strided(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    upper: u32,
    mid: u32,
    inner: u32,
    outer_step: u32,
    upper_step: u32,
    mid_step: u32,
    inner_step: u32,
    start: u32,
) {
    if gid() >= len {
        return;
    }
    let i = gid() % inner;
    let m = (gid() / inner) % mid;
    let u = (gid() / (inner * mid)) % upper;
    let o = gid() / (inner * mid * upper);
    let at = start + o * outer_step + u * upper_step + m * mid_step + i * inner_step;
    out[gid() as u64] = x[at as u64];
}

/// A view as four nested axes over its input.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(strided, threads = self.len)]
pub(crate) struct Strided {
    pub len: usize,
    pub upper: usize,
    pub mid: usize,
    pub inner: usize,
    pub outer_step: usize,
    pub upper_step: usize,
    pub mid_step: usize,
    pub inner_step: usize,
    pub start: usize,
}

/// Output axes as `(size, input step)`, neighbours merged.
fn runs(order: &[u32], shape: &[usize]) -> Vec<(usize, usize)> {
    let step = |a: usize| shape[a + 1..].iter().product::<usize>();
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut last = usize::MAX;
    for &a in order {
        let a = a as usize;
        if shape[a] == 1 {
            continue;
        }
        match out.last_mut() {
            Some(run) if last.wrapping_add(1) == a => *run = (run.0 * shape[a], step(a)),
            _ => out.push((shape[a], step(a))),
        }
        last = a;
    }
    out
}

/// Up to four runs as a kernel, outer runs first.
fn nested(runs: &[(usize, usize)], len: usize, start: usize) -> Option<Strided> {
    if runs.len() > 4 {
        return None;
    }
    let mut all = vec![(1, 0); 4 - runs.len()];
    all.extend_from_slice(runs);
    let [
        (_, outer_step),
        (upper, upper_step),
        (mid, mid_step),
        (inner, inner_step),
    ] = all[..]
    else {
        return None;
    };
    Some(Strided {
        len,
        upper,
        mid,
        inner,
        outer_step,
        upper_step,
        mid_step,
        inner_step,
        start,
    })
}

/// The view as strides over `shape`, its input's shape.
fn strides(tree: &Tree, view: &View, shape: &[usize], len: usize) -> Option<Strided> {
    let product = |s: &[usize]| s.iter().product::<usize>();
    let at = |axis: u32| axis as usize;
    match view {
        View::Slice { axis, start, .. } => {
            let (a, inner) = (at(*axis), product(&shape[at(*axis) + 1..]));
            // Starts are fixed sizes.
            let start = start.value(&tree.g.dims, &|_| None)? as usize;
            let mid = len / product(&shape[..a]) / inner;
            let runs = [
                (product(&shape[..a]), shape[a] * inner),
                (mid, inner),
                (inner, 1),
            ];
            nested(&runs, len, start * inner)
        }
        View::Expand { axis, .. } => {
            let inner = product(&shape[at(*axis) + 1..]);
            let mid = len / product(&shape[..at(*axis)]) / inner;
            let runs = [(product(&shape[..at(*axis)]), inner), (mid, 0), (inner, 1)];
            nested(&runs, len, 0)
        }
        View::Permute(order) => nested(&runs(order, shape), len, 0),
        View::Reshape(_) | View::Concat => None,
    }
}

/// Views of computed or fed f32 values.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::View(view)) = &node.term else {
        return None;
    };
    // The rule reading a weight owns its views.
    let ours = |t: &T| {
        let n = tree.node(*t);
        n.form.name() == F32_FORM.name()
            && !matches!(n.term, Term::Leaf(Home::Weight(_) | Home::Part(..)))
    };
    if !node.args.iter().all(ours) {
        return None;
    }
    let x = node.args[0];
    let len = tree.elements(at);
    if *view == View::Concat {
        let y = node.args[1];
        let side = |t: T| *tree.node(t).shape.last().unwrap_or(&1);
        let join = Join {
            left: side(x),
            right: side(y),
            rows: Extent::Fixed(tree.elements(x) / side(x).max(1)),
        };
        return Some(vec![Draft::new(join, [x, y], len)]);
    }
    let kernel = strides(tree, view, &tree.node(x).shape, len)?;
    Some(vec![Draft::new(kernel, [x], len)])
}

sensai_lower::rule!("view", propose);
