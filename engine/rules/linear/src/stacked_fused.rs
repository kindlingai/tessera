//! CUDA BF16 stacked matmul with fused input/output view strides.

use tessera_graph::{Op, Order, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Backend, ByBackend, Draft, Home, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T,
    Term, Tree,
};

const KERNEL: &str = include_str!("stacked_fused.cu");
const LACKS: &[Backend] = &[Backend::Metal, Backend::Rocm];
const BLOCK: usize = 128;
const THREADS: usize = 256;
const MMA16_THREADS: usize = 128;
const WARPS: usize = 8;
const MAX_GRID_Y: usize = 65_535;
const P64_SMEM: usize = 2 * 64 * 36 * 4;
const P16_SMEM: usize = (16 + 32) * 36 * 4;
const P16T_SMEM: usize = (16 + 64) * 36 * 4;
const WIDE_SMEM: usize = 4 * WARPS * 128 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum SfKind {
    Scalar = 0,
    Mma16 = 1,
    Mma64 = 2,
}

/// CUDA BF16 stacked matmul reading/writing permuted or sliced views directly.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = sf_name(self),
    sizes = [self.batches, self.rows, self.n, self.k],
    work = self.len * self.k,
    accepts = sf_accepts,
    emits = sf_emits,
    needs = sf_needs(self),
    order = Order::PerConfig,
    source = sf_source(self),
    launches = sf_launches,
    load = sf_load,
    load_key = tessera_lower::load_key((
        self.batches,
        self.rows,
        self.n,
        self.k,
        self.wide,
        self.kind as u8,
        self.x_form.name(),
        self.out_form.name(),
    )),
    unproven,
)]
struct StackedFused {
    len: usize,
    batches: usize,
    rows: usize,
    n: usize,
    k: usize,
    block: usize,
    first: usize,
    step: usize,
    x_bstep: usize,
    x_rstep: usize,
    x_off: usize,
    out_bstep: usize,
    out_rstep: usize,
    wide: bool,
    kind: SfKind,
    form: &'static dyn Stored,
    x_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
}

fn sf_name(k: &StackedFused) -> &'static str {
    match (k.kind, k.wide) {
        (SfKind::Mma64, true) => "stacked_fused_mma_t",
        (SfKind::Mma64, false) => "stacked_fused_mma",
        (SfKind::Mma16, true) => "stacked_fused_mma16_t",
        (SfKind::Mma16, false) => "stacked_fused_mma16",
        (SfKind::Scalar, true) => "stacked_fused_wide",
        (SfKind::Scalar, false) => "stacked_fused_lanes",
    }
}

fn sf_accepts(k: &StackedFused, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        _ => form.name() == k.x_form.name(),
    }
}

fn sf_emits(k: &StackedFused, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn sf_needs(k: &StackedFused) -> Needs {
    let (group_bytes, group_threads) = match (k.kind, k.wide) {
        (SfKind::Mma64, _) => (P64_SMEM, THREADS),
        (SfKind::Mma16, true) => (P16T_SMEM, MMA16_THREADS),
        (SfKind::Mma16, false) => (P16_SMEM, MMA16_THREADS),
        (SfKind::Scalar, true) => (WIDE_SMEM, THREADS),
        (SfKind::Scalar, false) => (0, THREADS),
    };
    Needs {
        lanes: 32,
        group_bytes,
        group_threads,
        compiler: true,
        bf16_tile: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn sf_source(k: &StackedFused) -> Source {
    let entry = match (k.kind, k.wide) {
        (SfKind::Mma64, true) => "tessera_sf_mma_t",
        (SfKind::Mma64, false) => "tessera_sf_mma",
        (SfKind::Mma16, true) => "tessera_sf_mma16_t",
        (SfKind::Mma16, false) => "tessera_sf_mma16",
        (SfKind::Scalar, true) => "tessera_sf_wide",
        (SfKind::Scalar, false) => "tessera_sf_lanes",
    };
    Source::Fixed {
        text: ByBackend(["", KERNEL, ""]),
        entry,
    }
}

fn sf_load(k: &StackedFused, _: usize) -> Option<Load> {
    let x_bytes = k.x_form.bytes(k.batches * k.rows * k.k) as f64;
    let written = k.out_form.bytes(k.len) as f64;
    let (groups, threads, trips, w_mult, group_bytes, flight, trip) = match k.kind {
        SfKind::Mma64 => {
            let g = k.n.div_ceil(64) * k.batches * k.rows.div_ceil(64);
            let op = Ops {
                fresh: 1.0,
                bf16_flops: 2048.0,
                ..Ops::default()
            };
            (
                g,
                THREADS,
                (k.k / 64) as f64,
                1usize,
                P64_SMEM as f64,
                1024.0,
                op,
            )
        }
        SfKind::Mma16 => {
            let bn = if k.wide { 64 } else { 32 };
            let rt = k.rows.div_ceil(16);
            let g = k.n.div_ceil(bn) * k.batches * rt;
            let smem = if k.wide { P16T_SMEM } else { P16_SMEM };
            let op = Ops {
                fresh: 1.0,
                bf16_flops: if k.wide { 1024.0 } else { 512.0 },
                ..Ops::default()
            };
            (
                g,
                MMA16_THREADS,
                (k.k / 64) as f64,
                rt,
                smem as f64,
                1024.0,
                op,
            )
        }
        SfKind::Scalar if k.wide => {
            let rt = k.rows.div_ceil(4);
            let r = k.rows.min(4);
            let op = Ops {
                float: (4 * r) as f64,
                int: 4.0,
                loads: (1 + r) as f64,
                width: (8.0 + r as f64 / 8.0) / (1 + r) as f64,
                fresh: 1.0,
                ..Ops::default()
            };
            (
                (k.n / BLOCK) * k.batches * rt,
                THREADS,
                k.k as f64 / WARPS as f64,
                rt,
                WIDE_SMEM as f64,
                512.0,
                op,
            )
        }
        SfKind::Scalar => {
            let rt = k.rows.div_ceil(8);
            let r = k.rows.min(8);
            let loads = (1 + 2 * r) as f64;
            let op = Ops {
                float: (8 * r) as f64,
                int: 8.0,
                loads,
                width: (16 + 32 * r) as f64 / loads,
                fresh: 1.0,
                ..Ops::default()
            };
            (
                (k.batches * k.n).div_ceil(WARPS) * rt,
                THREADS,
                k.k as f64 / 256.0,
                rt,
                0.0,
                512.0,
                op,
            )
        }
    };
    Some(Load {
        threads: groups * threads,
        group: threads,
        group_bytes,
        trips,
        trip,
        reads: vec![(2 * k.batches * w_mult * k.n * k.k) as f64, x_bytes],
        written,
        flight,
    })
}

fn sf_launches(k: &StackedFused, step: &Step) -> Vec<Launch> {
    let (across, down, threads, prefix) = match (k.kind, k.wide) {
        (SfKind::Mma64, _) => (
            k.n.div_ceil(64),
            k.batches * k.rows.div_ceil(64),
            THREADS,
            None,
        ),
        (SfKind::Mma16, true) => (
            k.n.div_ceil(64),
            k.batches * k.rows.div_ceil(16),
            MMA16_THREADS,
            None,
        ),
        (SfKind::Mma16, false) => (
            k.n.div_ceil(32),
            k.batches * k.rows.div_ceil(16),
            MMA16_THREADS,
            None,
        ),
        (SfKind::Scalar, true) => (k.n / BLOCK, k.batches * k.rows.div_ceil(4), THREADS, None),
        (SfKind::Scalar, false) => {
            let bn = k.batches * k.n;
            (bn.div_ceil(WARPS), k.rows.div_ceil(8), THREADS, Some(bn))
        }
    };
    let mut args = vec![step.read(0), step.read(1), step.out()];
    if let Some(bn) = prefix {
        args.push(Arg::count(bn));
    }
    let tail = [
        k.rows,
        k.n,
        k.k,
        k.block,
        k.first,
        k.step,
        k.x_bstep,
        k.x_rstep,
        k.x_off,
        k.out_bstep,
        k.out_rstep,
        usize::from(k.x_form.name() == "bf16"),
        usize::from(k.out_form.name() == "bf16"),
    ];
    args.extend(tail.map(Arg::count));
    vec![Launch {
        args,
        shape: Shape::Grid {
            across,
            down,
            threads,
        },
    }]
}

/// Checks `form` is `f32` or `bf16`.
fn float_form(form: &dyn Stored) -> bool {
    matches!(form.name(), "f32" | "bf16")
}

/// Checks `t` is a plain float intermediate.
fn plain_float(tree: &Tree, t: T) -> bool {
    let n = tree.node(t);
    float_form(n.form)
        && !matches!(
            n.term,
            Term::Leaf(Home::Weight(_) | Home::Part(..)) | Term::Op(Op::Append(_))
        )
}

/// Peels optional `Permute([1, 0, 2])` and trailing `Slice(axis=2)` from `x`.
fn peel_x(
    tree: &Tree,
    x: T,
    batches: usize,
    rows: usize,
    k: usize,
) -> (T, usize, usize, usize, bool) {
    let mut cur = x;
    let mut swapped = false;
    if !tree.root(cur)
        && let Term::Op(Op::View(View::Permute(order))) = &tree.node(cur).term
        && order[..] == [1, 0, 2]
        && tree.node(cur).shape[..] == [batches, rows, k]
        && plain_float(tree, tree.node(cur).args[0])
    {
        cur = tree.node(cur).args[0];
        swapped = true;
    }
    let mut stride = k;
    let mut off = 0usize;
    let mut sliced = false;
    if !tree.root(cur)
        && let Term::Op(Op::View(View::Slice { axis: 2, start, .. })) = &tree.node(cur).term
        && let Some(val) = start.value(&tree.g.dims, &|d| tree.size(d))
    {
        let parent = tree.node(cur).args[0];
        if plain_float(tree, parent)
            && let &[d0, d1, w] = tree.node(parent).shape.as_slice()
            && !(w == 576
                && k == 512
                && val == 0
                && matches!(tree.node(parent).term, Term::Op(Op::Attend(_)))
                && tree.node(tree.node(parent).args[1]).form.name() == "fp8")
            && ((swapped && d0 == rows && d1 == batches)
                || (!swapped && d0 == batches && d1 == rows))
        {
            cur = parent;
            stride = w;
            off = val as usize;
            sliced = true;
        }
    }
    let (bstep, rstep) = if swapped {
        (stride, batches * stride)
    } else {
        (rows * stride, stride)
    };
    (cur, bstep, rstep, off, swapped || sliced)
}

/// Unwraps reshaped BF16 weight blocks, optional row slice, and transpose.
fn peel_w(tree: &Tree, mut w: T) -> Option<(T, usize, usize, usize, bool, &'static dyn Stored)> {
    let swapped = matches!(
        &tree.node(w).term,
        Term::Op(Op::View(View::Permute(p))) if p[..] == [0, 2, 1]
    );
    if swapped {
        w = tree.node(w).args[0];
    }
    let mut first_row = 0usize;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = &tree.node(w).term {
        first_row = start.value(&tree.g.dims, &|d| tree.size(d))? as usize;
        w = tree.node(w).args[0];
    }
    if !matches!(&tree.node(w).term, Term::Op(Op::View(View::Reshape(_)))) {
        return None;
    }
    let leaf = tree.node(w).args[0];
    let &[_, block_rows, width] = tree.node(w).shape.as_slice() else {
        return None;
    };
    let &[_, stored_w] = tree.node(leaf).shape.as_slice() else {
        return None;
    };
    let form = tree.node(leaf).form;
    if !tree.leaf(leaf) || stored_w != width || form.name() != "bf16" {
        return None;
    }
    Some((
        leaf,
        block_rows * width,
        first_row * width,
        width,
        swapped,
        form,
    ))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let root_node = tree.node(at);
    let out_form = root_node.form;
    if !float_form(out_form) {
        return None;
    }
    let (mat, out_swap) = match &root_node.term {
        Term::Op(Op::View(View::Permute(order)))
            if order[..] == [1, 0, 2]
                && !tree.root(root_node.args[0])
                && tree.node(root_node.args[0]).term == Term::Op(Op::Matmul) =>
        {
            (root_node.args[0], true)
        }
        Term::Op(Op::Matmul) => (at, false),
        _ => return None,
    };
    let mnode = tree.node(mat);
    if !float_form(mnode.form) {
        return None;
    }
    let (x, w) = (mnode.args[0], mnode.args[1]);
    let &[batches, n, k] = tree.node(w).shape.as_slice() else {
        return None;
    };
    let &[xb, rows, xk] = tree.node(x).shape.as_slice() else {
        return None;
    };
    let x_form = tree.node(x).form;
    if xb != batches
        || xk != k
        || batches == 0
        || rows == 0
        || n == 0
        || k == 0
        || !float_form(x_form)
    {
        return None;
    }
    let (leaf, block, first, step, swapped, form) = peel_w(tree, w)?;
    let align = if swapped { 4 } else { 8 };
    if !block.is_multiple_of(align) || !first.is_multiple_of(align) || !step.is_multiple_of(align) {
        return None;
    }
    let (x_src, x_bstep, x_rstep, x_off, peeled_in) = peel_x(tree, x, batches, rows, k);
    let x_form = tree.node(x_src).form;
    if !peeled_in && !out_swap {
        return None;
    }
    if !x_bstep.is_multiple_of(4) || !x_rstep.is_multiple_of(4) || !x_off.is_multiple_of(4) {
        return None;
    }
    let (out_bstep, out_rstep) = if out_swap {
        (n, batches * n)
    } else {
        (rows * n, n)
    };
    let align_mma = k.is_multiple_of(64)
        && if swapped {
            n.is_multiple_of(64)
        } else {
            n.is_multiple_of(16)
        };
    let can_mma64 =
        rows >= 32 && align_mma && batches.saturating_mul(rows.div_ceil(64)) <= MAX_GRID_Y;
    let can_mma16 = (2..32).contains(&rows)
        && align_mma
        && batches.saturating_mul(rows.div_ceil(16)) <= MAX_GRID_Y;
    let scalar_ok = if swapped {
        n.is_multiple_of(BLOCK)
            && k <= 512
            && batches.saturating_mul(rows.div_ceil(4)) <= MAX_GRID_Y
    } else {
        k.is_multiple_of(8) && rows.div_ceil(8) <= MAX_GRID_Y
    };
    if !scalar_ok && !can_mma64 && !can_mma16 {
        return None;
    }
    let len = batches * rows * n;
    let base = StackedFused {
        len,
        batches,
        rows,
        n,
        k,
        block,
        first,
        step,
        x_bstep,
        x_rstep,
        x_off,
        out_bstep,
        out_rstep,
        wide: swapped,
        kind: SfKind::Scalar,
        form,
        x_form,
        out_form,
    };
    let mut drafts = Vec::with_capacity(3);
    if scalar_ok {
        drafts.push(Draft::new(base, [x_src, leaf], len).choice(0));
    }
    if can_mma64 {
        let m64 = StackedFused {
            kind: SfKind::Mma64,
            ..base
        };
        drafts.push(Draft::new(m64, [x_src, leaf], len).choice(1));
    }
    if can_mma16 {
        let m16 = StackedFused {
            kind: SfKind::Mma16,
            ..base
        };
        drafts.push(Draft::new(m16, [x_src, leaf], len).choice(2));
    }
    Some(drafts)
}

tessera_lower::rule!("stacked_fused", propose);
