//! Shared epilogue matching, split-K divisor, matmul site, and GEMV helpers.

use super::{CUDA_LACKS, THREADS};
use tessera_graph::{Act, Op, Stage};
use tessera_linear::Dense;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Launch, Load, Needs, Ops, Shape, Step, T, Term, Tree};

/// One thread's trip. Bf16 widens with one integer op.
pub(crate) fn trip(float: usize, int: usize, loads: usize, bytes: usize) -> Ops {
    Ops {
        float: float as f64,
        int: int as f64,
        loads: loads as f64,
        width: bytes as f64 / loads as f64,
        fresh: 1.0,
        ..Ops::default()
    }
}

pub(crate) fn cuda_needs(group_bytes: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads: THREADS,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

pub(crate) fn io_defs<'a>(
    base: impl IntoIterator<Item = (&'a str, usize)>,
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
) -> Vec<(&'a str, usize)> {
    let mut defs: Vec<_> = base.into_iter().collect();
    if x_bf16 {
        defs.push(("X_BF16", 1));
    }
    if y_bf16 {
        defs.push(("Y_BF16", 1));
    }
    if o_bf16 {
        defs.push(("O_BF16", 1));
    }
    defs
}

pub(crate) fn x_accepts(x_bf16: bool, form: &dyn Stored) -> bool {
    form.name() == if x_bf16 { "bf16" } else { "f32" }
}

pub(crate) fn weight_io_accepts(
    input: usize,
    x_bf16: bool,
    o_bf16: bool,
    expected: &dyn Stored,
    form: &dyn Stored,
) -> bool {
    match input {
        0 => x_accepts(x_bf16, form),
        1 => form.name() == expected.name(),
        2 => x_accepts(o_bf16, form),
        _ => form.name() == "f32",
    }
}

pub(crate) fn weight_x_accepts(
    input: usize,
    x_bf16: bool,
    expected: &dyn Stored,
    form: &dyn Stored,
) -> bool {
    weight_io_accepts(input, x_bf16, false, expected, form)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Epi {
    pub(crate) mode: u32,
    pub(crate) other: Option<T>,
    pub(crate) bias2: Option<T>,
    pub(crate) y_bf16: bool,
    pub(crate) o_bf16: bool,
}

pub(crate) trait MmDims {
    fn dims(&self) -> (usize, usize);
}

impl MmDims for Dense {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

fn is_io(form: &dyn Stored, allow_bf16_io: bool) -> bool {
    form.name() == "f32" || (allow_bf16_io && form.name() == "bf16")
}

pub(crate) fn pick_divisor(chunks: usize, want: usize, cap: usize) -> usize {
    for d in want..=cap {
        if chunks.is_multiple_of(d) {
            return d;
        }
    }
    for d in (1..want).rev() {
        if chunks.is_multiple_of(d) {
            return d;
        }
    }
    1
}

pub(crate) fn r64_ks(m: usize, n: usize, chunks: usize) -> usize {
    let base = n.div_ceil(64) * m.div_ceil(64);
    if chunks < 4 || base >= 96 {
        return 1;
    }
    let cap = if base >= 32 { 4 } else { 8 }.min(chunks / 2);
    let want = 96usize.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

pub(crate) fn has_other(mode: u32) -> bool {
    matches!(mode & 15, 1..=4 | 6..=8)
}

pub(crate) fn has_bias2(mode: u32) -> bool {
    (mode & 15) == 8
}

pub(crate) fn is_bias_bf16(mode: u32) -> bool {
    (mode & 16) != 0
}

pub(crate) fn rocm_o_bf16(mode: u32) -> bool {
    matches!(mode & 15, 2 | 6 | 7) && is_bias_bf16(mode)
}

pub(crate) fn epi_args(step: &Step, other_at: usize, mode: u32, active: bool) -> (Arg, Arg) {
    let has_o = active && has_other(mode);
    let other = if has_o {
        step.read(other_at)
    } else {
        step.read(0)
    };
    let bias = if active && has_bias2(mode) {
        step.read(other_at + usize::from(has_o))
    } else {
        step.read(0)
    };
    (other, bias)
}

pub(crate) fn epi_accepts(
    mode: u32,
    input: usize,
    other_at: usize,
    o_bf16: bool,
    form: &dyn Stored,
) -> bool {
    if has_other(mode) && input == other_at {
        return x_accepts(o_bf16, form);
    }
    if has_bias2(mode) && input == other_at + 1 {
        return x_accepts(is_bias_bf16(mode), form);
    }
    form.name() == "f32"
}

pub(crate) fn push_epi_reads(reads: &mut Vec<f64>, mode: u32, o_bf16: bool, m: usize, n: usize) {
    let base = mode & 15;
    let o_elem = if o_bf16 { 2 } else { 4 };
    if matches!(base, 1 | 3 | 4 | 8) {
        reads.push((o_elem * m * n) as f64);
    }
    if matches!(base, 2 | 6 | 7) {
        reads.push((o_elem * n) as f64);
    }
    if base == 8 {
        let b_elem = if is_bias_bf16(mode) { 2 } else { 4 };
        reads.push((b_elem * n) as f64);
    }
}

pub(crate) struct MatmulBase {
    pub(crate) x: T,
    pub(crate) w: T,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) x_bf16: bool,
    pub(crate) form: &'static dyn Stored,
}

/// Checks common 2D multi-row matmul preconditions.
pub(crate) fn check_mm_base(
    tree: &Tree,
    mat: T,
    allow_bf16_io: bool,
) -> Option<Result<MatmulBase, String>> {
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    if n == 0 || k == 0 {
        return None;
    }
    let elems = tree.elements(x);
    if !elems.is_multiple_of(k) {
        return None;
    }
    let m = elems / k;
    if m < 2 {
        return None;
    }
    let y_name = node.form.name();
    if !is_io(node.form, allow_bf16_io) {
        let req = if allow_bf16_io { "f32 or bf16" } else { "f32" };
        return Some(Err(format!("output form {y_name} (requires {req})")));
    }
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    let x_name = tree.node(x).form.name();
    let x_bf16 = match x_name {
        "f32" => false,
        "bf16" if allow_bf16_io => true,
        _ => {
            let req = if allow_bf16_io { "f32 or bf16" } else { "f32" };
            return Some(Err(format!("input form {x_name} (requires {req})")));
        }
    };
    Some(Ok(MatmulBase {
        x,
        w,
        m,
        n,
        k,
        x_bf16,
        form: tree.node(w).form,
    }))
}

fn match_mm_add_with<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
    bf16_bias_bit: bool,
) -> Option<(S, Epi)> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Add) || !is_io(node.form, allow_bf16_io) {
        return None;
    }
    let y_bf16 = node.form.name() == "bf16";
    let (a, b) = (node.args[0], *node.args.get(1)?);
    for (mat, add) in [(a, b), (b, a)] {
        if mat == add || tree.root(mat) {
            continue;
        }
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        let add_node = tree.node(add);
        let add_elems = tree.elements(add);
        let add_bf16 = add_node.form.name() == "bf16";
        let mode = if add_elems == m * n
            && add_node.shape == node.shape
            && is_io(add_node.form, allow_bf16_io)
        {
            1
        } else if add_elems == n
            && add_node.shape.last() == Some(&n)
            && is_io(add_node.form, allow_bf16_io)
        {
            2
        } else if bf16_bias_bit && add_elems == n && add_node.shape.last() == Some(&n) && add_bf16 {
            2 | 16
        } else {
            continue;
        };
        let epi = Epi {
            mode,
            other: Some(add),
            bias2: None,
            y_bf16,
            o_bf16: add_bf16,
        };
        return Some((site, epi));
    }
    None
}

fn match_mm_mul_with<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
) -> Option<(S, Epi)> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Mul) || !is_io(node.form, allow_bf16_io) {
        return None;
    }
    let y_bf16 = node.form.name() == "bf16";
    let (a, b) = (node.args[0], *node.args.get(1)?);
    let silu = Term::Stage(Stage::Act(Act::Silu));
    for (cand, other) in [(a, b), (b, a)] {
        if cand == other || tree.root(cand) || !is_io(tree.node(other).form, allow_bf16_io) {
            continue;
        }
        let cnode = tree.node(cand);
        if cnode.term != silu || !is_io(cnode.form, allow_bf16_io) {
            continue;
        }
        let mat = cnode.args[0];
        if mat == other || tree.root(mat) {
            continue;
        }
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            let epi = Epi {
                mode: 3,
                other: Some(other),
                bias2: None,
                y_bf16,
                o_bf16: tree.node(other).form.name() == "bf16",
            };
            return Some((site, epi));
        }
    }
    for (mat, other) in [(a, b), (b, a)] {
        if mat == other || tree.root(mat) || !is_io(tree.node(other).form, allow_bf16_io) {
            continue;
        }
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            let epi = Epi {
                mode: 4,
                other: Some(other),
                bias2: None,
                y_bf16,
                o_bf16: tree.node(other).form.name() == "bf16",
            };
            return Some((site, epi));
        }
    }
    None
}

fn match_mm_site_with<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
    bf16_bias_bit: bool,
) -> Option<(S, Epi)> {
    if let Some(hit) = match_mm_add_with(tree, at, pick, allow_bf16_io, bf16_bias_bit) {
        return Some(hit);
    }
    if let Some(hit) = match_mm_mul_with(tree, at, pick, allow_bf16_io) {
        return Some(hit);
    }
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Act(Act::Silu)) && is_io(node.form, allow_bf16_io) {
        let mat = node.args[0];
        if !tree.root(mat)
            && let Some(site) = pick(tree, mat)
        {
            let epi = Epi {
                mode: 5,
                y_bf16: node.form.name() == "bf16",
                ..Epi::default()
            };
            return Some((site, epi));
        }
    }
    let site = pick(tree, at)?;
    let epi = Epi {
        y_bf16: allow_bf16_io && tree.node(at).form.name() == "bf16",
        ..Epi::default()
    };
    Some((site, epi))
}

fn match_epi_with<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
) -> Option<(S, Epi)> {
    let bf16_bias_bit = !allow_bf16_io;
    let node = tree.node(at);
    if is_io(node.form, allow_bf16_io) {
        let y_bf16 = node.form.name() == "bf16";
        if let Term::Stage(Stage::Act(act @ (Act::GeluTanh | Act::Gelu))) = node.term {
            let add = node.args[0];
            if !tree.root(add)
                && let Some((site, inner)) =
                    match_mm_add_with(tree, add, pick, allow_bf16_io, bf16_bias_bit)
                && (inner.mode & 15) == 2
            {
                let base = if act == Act::GeluTanh { 6 } else { 7 };
                let epi = Epi {
                    mode: base | (inner.mode & 16),
                    other: inner.other,
                    bias2: None,
                    y_bf16,
                    o_bf16: inner.o_bf16,
                };
                return Some((site, epi));
            }
        }
        if node.term == Term::Stage(Stage::Add)
            && let Some(&b) = node.args.get(1)
        {
            let a = node.args[0];
            for (inner_t, resid) in [(a, b), (b, a)] {
                if inner_t == resid
                    || tree.root(inner_t)
                    || !is_io(tree.node(resid).form, allow_bf16_io)
                {
                    continue;
                }
                if let Some((site, inner)) =
                    match_mm_add_with(tree, inner_t, pick, allow_bf16_io, bf16_bias_bit)
                    && (inner.mode & 15) == 2
                {
                    let (m, n) = site.dims();
                    if tree.elements(resid) == m * n && tree.node(resid).shape == node.shape {
                        let epi = Epi {
                            mode: 8 | if inner.o_bf16 { 16 } else { 0 },
                            other: Some(resid),
                            bias2: inner.other,
                            y_bf16,
                            o_bf16: tree.node(resid).form.name() == "bf16",
                        };
                        return Some((site, epi));
                    }
                }
            }
        }
    }
    match_mm_site_with(tree, at, pick, allow_bf16_io, bf16_bias_bit)
}

pub(crate) fn match_mm_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, Epi)> {
    match_mm_site_with(tree, at, pick, true, false)
}

pub(crate) fn match_cuda_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, Epi)> {
    match_epi_with(tree, at, pick, true)
}

pub(crate) fn match_epi_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, Epi)> {
    match_epi_with(tree, at, pick, false)
}

pub(crate) fn gemv_load_for(
    form: &dyn Stored,
    [n, k]: [usize; 2],
    [groups, group]: [usize; 2],
    trips: f64,
    trip: Ops,
    [x_elem, y_elem]: [usize; 2],
) -> Option<Load> {
    Some(Load {
        threads: groups * group,
        group,
        trips,
        trip,
        reads: vec![form.bytes(n * k) as f64, (x_elem * k) as f64],
        written: (y_elem * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

pub(crate) fn gemv_launch(
    shape: Shape,
    n: usize,
    k: usize,
    act: Option<(u32, f32, f32)>,
    step: &Step,
) -> Vec<Launch> {
    let mut args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(n),
        Arg::count(k),
    ];
    if let Some((mode, a, b)) = act {
        args.extend([Arg::value(&mode), Arg::value(&a), Arg::value(&b)]);
    }
    vec![Launch { args, shape }]
}

pub(crate) fn gemv_epilogue(tree: &Tree, mut at: T) -> Option<(T, u32, f32, f32)> {
    let (mut mode, mut a, mut b) = (0u32, 1.0f32, 1.0f32);
    if let Term::Stage(Stage::Scale(s)) = tree.node(at).term {
        let prev = tree.node(at).args[0];
        if !tree.root(prev) && tree.node(prev).term == Term::Stage(Stage::Act(Act::Sigmoid)) {
            b = s;
            at = prev;
        }
    }
    match tree.node(at).term {
        Term::Stage(Stage::Act(Act::Silu)) if b == 1.0 => (mode, at) = (1, tree.node(at).args[0]),
        Term::Stage(Stage::Act(Act::Sigmoid)) => (mode, at) = (2, tree.node(at).args[0]),
        _ if b != 1.0 => return None,
        _ => {}
    }
    if mode != 0
        && !tree.root(at)
        && let Term::Stage(Stage::Scale(s)) = tree.node(at).term
    {
        (a, at) = (s, tree.node(at).args[0]);
    }
    (mode == 0 || !tree.root(at)).then_some((at, mode, a, b))
}

pub(crate) type GemvCheck = fn(&Tree, T, u32, f32, f32) -> Option<Result<Vec<Draft>, String>>;

pub(crate) fn gemv_propose(tree: &Tree, at: T, check: GemvCheck) -> Option<Vec<Draft>> {
    if tree.node(at).form.name() != "f32" {
        return None;
    }
    let (mat, mode, a, b) = gemv_epilogue(tree, at)?;
    check(tree, mat, mode, a, b)?.ok()
}
