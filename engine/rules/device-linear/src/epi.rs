//! Shared matmul epilogue matching, divisor selection, and GEMV helpers.

use tessera_graph::{Act, Op, Stage};
use tessera_linear::Dense;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Launch, Load, Ops, Shape, Step, T, Term, Tree};

pub(super) fn pick_divisor(chunks: usize, want: usize, cap: usize) -> usize {
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

pub(super) fn has_other(mode: u32) -> bool {
    matches!(mode & 15, 1..=4 | 6..=8)
}

pub(super) fn has_bias2(mode: u32) -> bool {
    (mode & 15) == 8
}

pub(super) fn is_bias(mode: u32) -> bool {
    matches!(mode & 15, 2 | 6 | 7)
}

pub(super) fn is_bias_bf16(mode: u32) -> bool {
    (mode & 16) != 0
}

pub(super) fn epi_accepts(mode: u32, input: usize, other_at: usize, form: &dyn Stored) -> bool {
    let bias_slot = if is_bias(mode) {
        Some(other_at)
    } else if has_bias2(mode) {
        Some(other_at + 1)
    } else {
        None
    };
    if bias_slot == Some(input) && is_bias_bf16(mode) {
        return form.name() == "bf16";
    }
    form.name() == "f32"
}

pub(super) fn push_epi_reads(reads: &mut Vec<f64>, mode: u32, m: usize, n: usize) {
    let base = mode & 15;
    if matches!(base, 1 | 3 | 4 | 8) {
        reads.push((4 * m * n) as f64);
    }
    if matches!(base, 2 | 6 | 7 | 8) {
        let elem = if is_bias_bf16(mode) { 2 } else { 4 };
        reads.push((elem * n) as f64);
    }
}

pub(super) struct MatmulBase {
    pub(super) x: T,
    pub(super) w: T,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) x_bf16: bool,
    pub(super) y_bf16: bool,
    pub(super) form: &'static dyn Stored,
}

/// Checks common 2D multi-row matmul preconditions.
pub(super) fn check_mm_base(
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
    let y_bf16 = match y_name {
        "f32" => false,
        "bf16" if allow_bf16_io => true,
        _ if allow_bf16_io => {
            return Some(Err(format!("output form {y_name} (requires f32 or bf16)")));
        }
        _ => return Some(Err(format!("output form {y_name} (requires f32)"))),
    };
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    let x_name = tree.node(x).form.name();
    let x_bf16 = match x_name {
        "f32" => false,
        "bf16" if allow_bf16_io => true,
        _ if allow_bf16_io => {
            return Some(Err(format!("input form {x_name} (requires f32 or bf16)")));
        }
        _ => return Some(Err(format!("input form {x_name} (requires f32)"))),
    };
    Some(Ok(MatmulBase {
        x,
        w,
        m,
        n,
        k,
        x_bf16,
        y_bf16,
        form: tree.node(w).form,
    }))
}

pub(super) trait MmDims {
    fn dims(&self) -> (usize, usize);
    fn set_io(&mut self, _y_bf16: bool, _o_bf16: bool) {}
}

impl MmDims for Dense {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

fn is_io(form: &dyn Stored, allow_bf16_io: bool) -> bool {
    form.name() == "f32" || (allow_bf16_io && form.name() == "bf16")
}

pub(super) fn match_mm_add<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
    bf16_bias_bit: bool,
) -> Option<(S, u32, Option<T>)> {
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
        let Some(mut site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        let add_node = tree.node(add);
        let add_elems = tree.elements(add);
        let form = add_node.form.name();
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
        } else if bf16_bias_bit
            && add_elems == n
            && add_node.shape.last() == Some(&n)
            && form == "bf16"
        {
            2 | 16
        } else {
            continue;
        };
        site.set_io(y_bf16, !bf16_bias_bit && form == "bf16");
        return Some((site, mode, Some(add)));
    }
    None
}

fn match_mm_mul<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
) -> Option<(S, u32, Option<T>)> {
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
        let Some(mut site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            site.set_io(y_bf16, tree.node(other).form.name() == "bf16");
            return Some((site, 3, Some(other)));
        }
    }
    for (mat, other) in [(a, b), (b, a)] {
        if mat == other || tree.root(mat) || !is_io(tree.node(other).form, allow_bf16_io) {
            continue;
        }
        let Some(mut site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            site.set_io(y_bf16, tree.node(other).form.name() == "bf16");
            return Some((site, 4, Some(other)));
        }
    }
    None
}

pub(super) fn match_mm_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16_io: bool,
) -> Option<(S, u32, Option<T>)> {
    if let Some(hit) = match_mm_add(tree, at, pick, allow_bf16_io, false) {
        return Some(hit);
    }
    if let Some(hit) = match_mm_mul(tree, at, pick, allow_bf16_io) {
        return Some(hit);
    }
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Act(Act::Silu)) && is_io(node.form, allow_bf16_io) {
        let mat = node.args[0];
        if !tree.root(mat)
            && let Some(mut site) = pick(tree, mat)
        {
            site.set_io(node.form.name() == "bf16", false);
            return Some((site, 5, None));
        }
    }
    Some((pick(tree, at)?, 0, None))
}

pub(super) fn match_epi_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>, Option<T>)> {
    let node = tree.node(at);
    if node.form.name() == "f32" {
        if let Term::Stage(Stage::Act(act @ (Act::GeluTanh | Act::Gelu))) = node.term {
            let add = node.args[0];
            if !tree.root(add)
                && let Some((site, m_add, bias)) = match_mm_add(tree, add, pick, false, true)
                && (m_add & 15) == 2
            {
                let base = if act == Act::GeluTanh { 6 } else { 7 };
                return Some((site, base | (m_add & 16), bias, None));
            }
        }
        if node.term == Term::Stage(Stage::Add)
            && let Some(&b) = node.args.get(1)
        {
            let a = node.args[0];
            for (inner, resid) in [(a, b), (b, a)] {
                if inner == resid || tree.root(inner) || tree.node(resid).form.name() != "f32" {
                    continue;
                }
                if let Some((site, m_add, bias)) = match_mm_add(tree, inner, pick, false, true)
                    && (m_add & 15) == 2
                {
                    let (m, n) = site.dims();
                    if tree.elements(resid) == m * n && tree.node(resid).shape == node.shape {
                        return Some((site, 8 | (m_add & 16), Some(resid), bias));
                    }
                }
            }
            if let Some((site, m_add, other)) = match_mm_add(tree, at, pick, false, true) {
                return Some((site, m_add, other, None));
            }
        }
    }
    let (site, mode, other) = match_mm_site(tree, at, pick, false)?;
    Some((site, mode, other, None))
}

pub(super) fn gemv_load_for(
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

pub(super) fn gemv_launch(
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

pub(super) fn gemv_epilogue(tree: &Tree, mut at: T) -> Option<(T, u32, f32, f32)> {
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

pub(super) type GemvCheck = fn(&Tree, T, u32, f32, f32) -> Option<Result<Vec<Draft>, String>>;

pub(super) fn gemv_propose(tree: &Tree, at: T, check: GemvCheck) -> Option<Vec<Draft>> {
    if tree.node(at).form.name() != "f32" {
        return None;
    }
    let (mat, mode, a, b) = gemv_epilogue(tree, at)?;
    check(tree, mat, mode, a, b)?.ok()
}
