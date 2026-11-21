//! Slices, repeats, swaps and joins, copied into a new buffer.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::kernels::Join;
use tessera_lower::{Draft, Extent, Home, Kernel, Source, T, Term, Tree};

/// One thread per output `[outer, upper, mid, inner]`, strided reads.
#[kernel]
fn strided(
    #[form(in_form)] x: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    upper: u32,
    mid: u32,
    inner: u32,
    outer_step: u32,
    upper_step: u32,
    mid_step: u32,
    inner_step: u32,
    start: u32,
    #[build] in_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
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
#[launch(
    strided(self.in_form, self.out_form),
    threads = self.len,
    accepts = strided_accepts,
    emits = strided_emits,
    source = strided_source,
)]
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
    pub in_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn strided_accepts(k: &Strided, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.in_form.name()
}

fn strided_emits(k: &Strided, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn strided_source(k: &Strided) -> Source {
    let entry = match (k.in_form.name(), k.out_form.name()) {
        ("f32", "f32") => "plan_strided".into(),
        (a, b) => format!("plan_strided_{a}_{b}"),
    };
    Source::named(strided_ir(k.in_form, k.out_form), entry)
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
fn nested(
    runs: &[(usize, usize)],
    len: usize,
    start: usize,
    in_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
) -> Option<Strided> {
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
        in_form,
        out_form,
    })
}

/// The view as strides over `shape`, its input's shape.
fn strides(
    tree: &Tree,
    view: &View,
    shape: &[usize],
    len: usize,
    forms: (&'static dyn Stored, &'static dyn Stored),
) -> Option<Strided> {
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
            nested(&runs, len, start * inner, forms.0, forms.1)
        }
        View::Expand { axis, .. } => {
            let inner = product(&shape[at(*axis) + 1..]);
            let mid = len / product(&shape[..at(*axis)]) / inner;
            let runs = [(product(&shape[..at(*axis)]), inner), (mid, 0), (inner, 1)];
            nested(&runs, len, 0, forms.0, forms.1)
        }
        View::Permute(order) => nested(&runs(order, shape), len, 0, forms.0, forms.1),
        View::Reshape(_) | View::Concat => None,
    }
}

/// Views of computed or fed float values.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::View(view)) = &node.term else {
        return None;
    };
    // The rule reading a weight owns its views.
    let ours = |t: &T| {
        let n = tree.node(*t);
        n.form.dtype().is_float() && !matches!(n.term, Term::Leaf(Home::Weight(_) | Home::Part(..)))
    };
    if !node.args.iter().all(ours) {
        return None;
    }
    let x = node.args[0];
    let len = tree.elements(at);
    let out_form = writable(node.form);
    if *view == View::Concat {
        let y = node.args[1];
        let side = |t: T| *tree.node(t).shape.last().unwrap_or(&1);
        let join = Join {
            left: side(x),
            right: side(y),
            rows: Extent::Fixed(tree.elements(x) / side(x).max(1)),
            a_form: plain(tree.node(x).form),
            b_form: plain(tree.node(y).form),
            out_form,
        };
        return Some(vec![Draft::new(join, [x, y], len)]);
    }
    let in_form = plain(tree.node(x).form);
    let kernel = strides(tree, view, &tree.node(x).shape, len, (in_form, out_form))?;
    Some(vec![Draft::new(kernel, [x], len)])
}

tessera_lower::rule!("view", propose);
