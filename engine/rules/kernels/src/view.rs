//! Slices, repeats, swaps and joins, copied into a new buffer.

use tessera_codegen::kept::{self, Layout};
use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::kernels::Join;
use tessera_lower::{Arg, Draft, Extent, Home, Kernel, Launch, Shape, Source, Step, T, Term, Tree};

/// One thread per output or quad, strided reads.
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
    #[build] wide: bool,
) {
    if wide {
        if gid() >= len / 4 {
            return;
        }
        let inner4 = inner / 4;
        let i = (gid() % inner4) * 4;
        let m = (gid() / inner4) % mid;
        let u = (gid() / (inner4 * mid)) % upper;
        let o = gid() / (inner4 * mid * upper);
        let at = start + o * outer_step + u * upper_step + m * mid_step + i;
        let v = x.load4(at as u64);
        out.store4((gid() * 4) as u64, F32x4::new(v.x, v.y, v.z, v.w));
    } else {
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
}

/// A view as four nested axes over its input.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    strided(self.in_form, self.out_form, self.wide),
    threads = if self.wide { self.len / 4 } else { self.len },
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
    pub wide: bool,
}

fn strided_accepts(k: &Strided, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.in_form.name()
}

fn strided_emits(k: &Strided, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn strided_source(k: &Strided) -> Source {
    let suffix = if k.wide { "_x4" } else { "" };
    let entry = match (k.in_form.name(), k.out_form.name()) {
        ("f32", "f32") => format!("plan_strided{suffix}"),
        (a, b) => format!("plan_strided{suffix}_{a}_{b}"),
    };
    Source::named(strided_ir(k.in_form, k.out_form, k.wide), entry)
}

/// Output axes as `(size, input step)`, neighbours merged.
fn runs(order: &[u32], shape: &[usize], steps: &[usize]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut last = usize::MAX;
    for &a in order {
        let a = a as usize;
        if shape[a] == 1 {
            continue;
        }
        let step = steps[a];
        match out.last_mut() {
            Some(run) if last.wrapping_add(1) == a && run.1 == shape[a] * step => {
                *run = (run.0 * shape[a], step);
            }
            _ => out.push((shape[a], step)),
        }
        last = a;
    }
    out
}

/// Dense input strides for `shape`.
fn dense_steps(shape: &[usize]) -> Vec<usize> {
    (0..shape.len())
        .map(|a| shape[a + 1..].iter().product::<usize>())
        .collect()
}

/// Whether `f` supports vectorized float loads and stores.
fn float_form(f: &dyn Stored) -> bool {
    matches!(
        f.dtype(),
        DType::F32 | DType::Bf16 | DType::F16 | DType::Fp8
    )
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
    let wide = float_form(in_form)
        && float_form(out_form)
        && inner_step == 1
        && inner.is_multiple_of(4)
        && start.is_multiple_of(4)
        && mid_step.is_multiple_of(4)
        && upper_step.is_multiple_of(4)
        && outer_step.is_multiple_of(4);
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
        wide,
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
            let start = tree.eval(start)? as usize;
            let mid = len / product(&shape[..a]) / inner;
            let runs = [(product(&shape[..a]), shape[a] * inner), (mid * inner, 1)];
            nested(&runs, len, start * inner, forms.0, forms.1)
        }
        View::Expand { axis, .. } => {
            let inner = product(&shape[at(*axis) + 1..]);
            let mid = len / product(&shape[..at(*axis)]) / inner;
            let runs = [(product(&shape[..at(*axis)]), inner), (mid, 0), (inner, 1)];
            nested(&runs, len, 0, forms.0, forms.1)
        }
        View::Permute(order) => {
            let steps = dense_steps(shape);
            nested(&runs(order, shape, &steps), len, 0, forms.0, forms.1)
        }
        View::Reshape(_) | View::Concat => None,
    }
}

/// Joins two tensors where either may be kept.
#[kernel]
fn kept_join(
    #[form(a_form)] a: Buf<Dyn>,
    #[form(b_form)] b: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    keys: u32,
    left: u32,
    right: u32,
    #[build] a_kept: bool,
    #[build] b_kept: bool,
    #[build] a_form: &'static dyn Stored,
    #[build] b_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] wide: bool,
) {
    if wide {
        let left4 = left / 4;
        let total4 = left4 + right / 4;
        if total4 == 0 || gid() >= rows * total4 {
            return;
        }
        let r = gid() / total4;
        let j4 = gid() - r * total4;
        let kr = if a_kept && b_kept {
            kept_row(r / keys, r % keys)
        } else {
            r
        };
        let mut vx = 0.0f32;
        let mut vy = 0.0f32;
        let mut vz = 0.0f32;
        let mut vw = 0.0f32;
        if j4 < left4 {
            let ar = if a_kept && b_kept {
                kr
            } else if a_kept {
                kept_row(r / keys, r % keys)
            } else {
                r
            };
            let v = a.load4((ar * left + j4 * 4) as u64);
            vx = v.x;
            vy = v.y;
            vz = v.z;
            vw = v.w;
        } else {
            let br = if a_kept && b_kept {
                kr
            } else if b_kept {
                kept_row(r / keys, r % keys)
            } else {
                r
            };
            let v = b.load4((br * right + (j4 - left4) * 4) as u64);
            vx = v.x;
            vy = v.y;
            vz = v.z;
            vw = v.w;
        }
        out.store4((gid() * 4) as u64, F32x4::new(vx, vy, vz, vw));
    } else {
        let total = left + right;
        if total == 0 || gid() >= rows * total {
            return;
        }
        let r = gid() / total;
        let j = gid() - r * total;
        let kr = if a_kept && b_kept {
            kept_row(r / keys, r % keys)
        } else {
            r
        };
        let mut v = 0.0f32;
        if j < left {
            let ar = if a_kept && b_kept {
                kr
            } else if a_kept {
                kept_row(r / keys, r % keys)
            } else {
                r
            };
            v = a[(ar * left + j) as u64];
        } else {
            let br = if a_kept && b_kept {
                kr
            } else if b_kept {
                kept_row(r / keys, r % keys)
            } else {
                r
            };
            v = b[(br * right + (j - left)) as u64];
        }
        out[gid() as u64] = v;
    }
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "kept_join",
    sizes = [],
    work = self.rows * (self.left + self.right),
    accepts = kept_join_accepts,
    emits = kept_join_emits,
    source = kept_join_source,
    launches = kept_join_launches,
)]
pub(crate) struct KeptJoin {
    pub rows: usize,
    pub keys: usize,
    pub left: usize,
    pub right: usize,
    pub slots: usize,
    pub context: usize,
    pub a_kept: bool,
    pub b_kept: bool,
    pub kept: &'static dyn Layout,
    pub a_form: &'static dyn Stored,
    pub b_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub wide: bool,
}

fn kept_join_accepts(k: &KeptJoin, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.a_form.name(),
        1 => form.name() == k.b_form.name(),
        _ => !form.narrow(),
    }
}

fn kept_join_emits(k: &KeptJoin, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn kept_join_source(k: &KeptJoin) -> Source {
    let body = kept::attach(
        kept_join_ir(k.a_kept, k.b_kept, k.a_form, k.b_form, k.out_form, k.wide),
        k.kept,
    );
    let flags = usize::from(k.a_kept) | (usize::from(k.b_kept) << 1);
    let suffix = if k.wide { "_x4" } else { "" };
    let entry = format!(
        "plan_kept_join{suffix}_{}_{flags}_{}_{}_{}",
        k.kept.name(),
        k.a_form.name(),
        k.b_form.name(),
        k.out_form.name(),
    );
    Source::named(body, entry)
}

fn kept_join_launches(k: &KeptJoin, step: &Step) -> Vec<Launch> {
    let total = k.rows * (k.left + k.right);
    let threads = if k.wide { total / 4 } else { total };
    let sizes = [k.rows, k.keys, k.left, k.right].map(Arg::count);
    let kept = k
        .kept
        .values(k.slots, k.context)
        .into_iter()
        .map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.out()]
        .into_iter()
        .chain(step.kept_for(k.kept, true))
        .chain(sizes)
        .chain(kept)
        .collect();
    vec![Launch {
        args,
        shape: Shape::Line(threads),
    }]
}

/// Copies a contiguous `u32` slice along axis zero.
#[kernel]
fn slice_u32(x: Buf<U32>, mut out: BufMut<U32>, len: u32, start: u32) {
    if gid() >= len {
        return;
    }
    out[gid() as u64] = x[(start + gid()) as u64];
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    slice_u32(),
    threads = self.len,
    accepts = slice_u32_dtype,
    emits = slice_u32_emit,
    source = slice_u32_source,
)]
pub(crate) struct SliceU32 {
    pub len: usize,
    pub start: usize,
}

fn slice_u32_dtype(_: &SliceU32, _: usize, form: &dyn Stored) -> bool {
    form.dtype() == DType::U32
}

fn slice_u32_emit(_: &SliceU32, form: &dyn Stored) -> bool {
    form.dtype() == DType::U32
}

fn slice_u32_source(_: &SliceU32) -> Source {
    Source::named(slice_u32_ir(), "plan_slice_u32")
}

/// Views of computed or fed values.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::View(view)) = &node.term else {
        return None;
    };
    if let View::Slice { axis: 0, start, .. } = view
        && node.form.dtype() == DType::U32
    {
        let x = node.args[0];
        let x_node = tree.node(x);
        if x_node.form.dtype() != DType::U32 {
            return None;
        }
        let inner: usize = x_node.shape[1..].iter().product();
        let off = tree.eval(start)? as usize * inner;
        let len = tree.elements(at);
        return Some(vec![Draft::new(SliceU32 { len, start: off }, [x], len)]);
    }
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
        let kept_of = |t: T| match tree.node(t).term {
            Term::Op(Op::Append(i)) => Some(i),
            _ => None,
        };
        let (xk, yk) = (kept_of(x), kept_of(y));
        if xk.is_some() || yk.is_some() {
            let (xs, ys) = (&tree.node(x).shape, &tree.node(y).shape);
            if xs.len() < 2 || ys.len() < 2 || xs[0] != ys[0] || xs[1] != ys[1] {
                return None;
            }
            let ctx_x = xk.and_then(|i| tree.kept_rows(i).map(|(_, c)| c));
            let ctx_y = yk.and_then(|i| tree.kept_rows(i).map(|(_, c)| c));
            if let (Some(cx), Some(cy)) = (ctx_x, ctx_y)
                && cx != cy
            {
                return None;
            }
            let context = ctx_x.or(ctx_y)?;
            let left = side(x);
            let right = side(y);
            let rows = tree.elements(x) / left.max(1);
            let keys = xs[1] * (rows / (xs[0] * xs[1]).max(1));
            let a_form = plain(tree.node(x).form);
            let b_form = plain(tree.node(y).form);
            let wide = float_form(a_form)
                && float_form(b_form)
                && float_form(out_form)
                && left.is_multiple_of(4)
                && right.is_multiple_of(4);
            let kernel = KeptJoin {
                rows,
                keys,
                left,
                right,
                slots: xs[0],
                context,
                a_kept: xk.is_some(),
                b_kept: yk.is_some(),
                kept: tree.kept,
                a_form,
                b_form,
                out_form,
                wide,
            };
            return Some(vec![Draft::new(kernel, [x, y], len)]);
        }
        let join = Join {
            left: side(x),
            right: side(y),
            rows: Extent::Fixed(tree.elements(x) / side(x).max(1)),
            a_form: plain(tree.node(x).form),
            b_form: plain(tree.node(y).form),
            out_form,
        };
        if !join.fits() {
            return None;
        }
        return Some(vec![Draft::new(join, [x, y], len)]);
    }
    // Fuse a single-reader slice followed by a permute.
    if let View::Permute(order) = view
        && !tree.root(x)
        && let Term::Op(Op::View(View::Slice { axis, start, .. })) = &tree.node(x).term
    {
        let parent = tree.node(x).args[0];
        if ours(&parent)
            && let Some(off) = tree.eval(start)
        {
            let p_shape = &tree.node(parent).shape;
            let steps = dense_steps(p_shape);
            let start = off as usize * steps[*axis as usize];
            let p_form = plain(tree.node(parent).form);
            if let Some(kernel) = nested(
                &runs(order, &tree.node(x).shape, &steps),
                len,
                start,
                p_form,
                out_form,
            ) {
                return Some(vec![Draft::new(kernel, [parent], len)]);
            }
        }
    }
    let in_form = plain(tree.node(x).form);
    let kernel = strides(tree, view, &tree.node(x).shape, len, (in_form, out_form))?;
    Some(vec![Draft::new(kernel, [x], len)])
}

tessera_lower::rule!("view", propose);
