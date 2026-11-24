//! Streaming BF16 and MXFP8 Tensor-Core small-batch matmul rule.

use super::{CUDA_LACKS, cuda_source_entry, f32_only, weight_accepts};
use tessera_graph::{Act, Op, Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Step, T, Term, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmKind {
    Bf16,
    Bf16Split,
    Mxfp8,
    Mxfp8Wide,
    Mxfp8Tall,
}

impl MmKind {
    fn name(self) -> &'static str {
        match self {
            MmKind::Bf16 | MmKind::Bf16Split => "fast_mm_bf16",
            MmKind::Mxfp8 | MmKind::Mxfp8Wide | MmKind::Mxfp8Tall => "fast_mm_mxfp8",
        }
    }

    fn entry(self) -> &'static str {
        match self {
            MmKind::Bf16 => "tessera_mm_bf16",
            MmKind::Bf16Split => "tessera_mm_bf16_ks2",
            MmKind::Mxfp8 => "tessera_mm_mxfp8_r8",
            MmKind::Mxfp8Wide => "tessera_mm_mxfp8_r16",
            MmKind::Mxfp8Tall => "tessera_mm_mxfp8_r32",
        }
    }

    fn smem(self) -> usize {
        match self {
            MmKind::Bf16 => (2 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Bf16Split => (4 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Mxfp8 | MmKind::Mxfp8Wide | MmKind::Mxfp8Tall => 4 * 16 * 36 * 4,
        }
    }

    fn grid(self, m: usize, n: usize) -> (usize, usize) {
        match self {
            MmKind::Bf16 => (n / 32, m.div_ceil(16)),
            MmKind::Bf16Split | MmKind::Mxfp8Wide => (n / 16, m.div_ceil(16)),
            MmKind::Mxfp8 => (n / 16, 1),
            MmKind::Mxfp8Tall => (n / 16, m.div_ceil(32)),
        }
    }

    fn step(self) -> f64 {
        match self {
            MmKind::Bf16 => 64.0,
            MmKind::Bf16Split => 128.0,
            MmKind::Mxfp8 | MmKind::Mxfp8Wide | MmKind::Mxfp8Tall => 512.0,
        }
    }
}

/// Streaming BF16 and MXFP8 Tensor-Core small-batch matmul.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.name(),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = mm_accepts,
    emits = f32_only,
    needs = mm_needs(self),
    order = Order::PerConfig,
    source = cuda_source_entry(self.kind.entry()),
    launches = mm_launches,
    load = mm_load,
    unproven,
)]
struct FastMm {
    m: usize,
    n: usize,
    k: usize,
    mode: u32,
    kind: MmKind,
    form: &'static dyn Stored,
}

fn mm_accepts(k: &FastMm, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn mm_needs(k: &FastMm) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: k.kind.smem(),
        group_threads: 128,
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn mm_load(k: &FastMm, _: usize) -> Option<Load> {
    let (across, down) = k.kind.grid(k.m, k.n);
    let groups = across * down;
    let mut reads = vec![k.form.bytes(k.n * k.k) as f64, (4 * k.m * k.k) as f64];
    if matches!(k.mode, 1 | 3 | 4) {
        reads.push((4 * k.m * k.n) as f64);
    } else if k.mode == 2 {
        reads.push((4 * k.n) as f64);
    }
    let mut load = Load {
        threads: groups * 128,
        group: 128,
        trips: k.k as f64 / k.kind.step(),
        reads,
        written: (4 * k.m * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn mm_launches(k: &FastMm, step: &Step) -> Vec<Launch> {
    let (across, down) = k.kind.grid(k.m, k.n);
    let shape = Shape::Grid {
        across,
        down,
        threads: 128,
    };
    let other = if matches!(k.mode, 1..=4) {
        step.read(2)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        other,
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
    ];
    vec![Launch { args, shape }]
}

struct MmSite {
    x: T,
    w: T,
    m: usize,
    n: usize,
    k: usize,
    kind: MmKind,
    form: &'static dyn Stored,
}

impl MmSite {
    fn draft(self, mode: u32, other: Option<T>) -> Vec<Draft> {
        let kernel = FastMm {
            m: self.m,
            n: self.n,
            k: self.k,
            mode,
            kind: self.kind,
            form: self.form,
        };
        let mut reads = vec![self.x, self.w];
        if let Some(extra) = other {
            reads.push(extra);
        }
        vec![Draft::new(kernel, reads, self.m * self.n)]
    }
}

/// Validates a multi-row 2D matmul site or explains its rejection.
fn check_mm(tree: &Tree, mat: T) -> Option<Result<MmSite, String>> {
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
    if node.form.name() != "f32" {
        return Some(Err(format!(
            "output form {} (requires f32)",
            node.form.name()
        )));
    }
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    if tree.node(x).form.name() != "f32" {
        return Some(Err(format!(
            "input form {} (requires f32)",
            tree.node(x).form.name()
        )));
    }
    if m > 64 {
        return Some(Err(format!("m={m} exceeds max 64")));
    }
    let form = tree.node(w).form;
    let kind = match form.name() {
        "bf16"
            if n.is_multiple_of(16)
                && k.is_multiple_of(128)
                && (!n.is_multiple_of(32) || (n / 32) * m.div_ceil(16) < 144) =>
        {
            MmKind::Bf16Split
        }
        "bf16" if n.is_multiple_of(32) && k.is_multiple_of(64) => MmKind::Bf16,
        "bf16" => {
            return Some(Err(format!(
                "n={n}, k={k} not aligned for bf16 (needs n%32 & k%64, or n%16 & k%128)"
            )));
        }
        "mxfp8" if n.is_multiple_of(16) && k.is_multiple_of(512) => {
            if m <= 8 {
                MmKind::Mxfp8
            } else if m <= 16 {
                MmKind::Mxfp8Wide
            } else {
                MmKind::Mxfp8Tall
            }
        }
        "mxfp8" if !n.is_multiple_of(16) => {
            return Some(Err(format!("n={n} is not a multiple of 16 for mxfp8")));
        }
        "mxfp8" => {
            return Some(Err(format!("k={k} is not a multiple of 512 for mxfp8")));
        }
        other => {
            return Some(Err(format!("weight form {other} (supports bf16, mxfp8)")));
        }
    };
    Some(Ok(MmSite {
        x,
        w,
        m,
        n,
        k,
        kind,
        form,
    }))
}

fn mm_site(tree: &Tree, mat: T) -> Option<MmSite> {
    check_mm(tree, mat)?.ok()
}

fn propose_mm_add(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Add) || node.form.name() != "f32" {
        return None;
    }
    let (a, b) = (node.args[0], *node.args.get(1)?);
    for (mat, add) in [(a, b), (b, a)] {
        if mat == add || tree.root(mat) || tree.node(add).form.name() != "f32" {
            continue;
        }
        let Some(site) = mm_site(tree, mat) else {
            continue;
        };
        let add_node = tree.node(add);
        let add_elems = tree.elements(add);
        let mode = if add_elems == site.m * site.n && add_node.shape == node.shape {
            1
        } else if add_elems == site.n && add_node.shape.last() == Some(&site.n) {
            2
        } else {
            continue;
        };
        return Some(site.draft(mode, Some(add)));
    }
    None
}

fn propose_mm_mul(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Mul) || node.form.name() != "f32" {
        return None;
    }
    let (a, b) = (node.args[0], *node.args.get(1)?);
    let silu = Term::Stage(Stage::Act(Act::Silu));
    for (cand, other) in [(a, b), (b, a)] {
        if cand == other || tree.root(cand) || tree.node(other).form.name() != "f32" {
            continue;
        }
        let cnode = tree.node(cand);
        if cnode.term != silu || cnode.form.name() != "f32" {
            continue;
        }
        let mat = cnode.args[0];
        if mat == other || tree.root(mat) {
            continue;
        }
        let Some(site) = mm_site(tree, mat) else {
            continue;
        };
        if tree.elements(other) == site.m * site.n && tree.node(other).shape == node.shape {
            return Some(site.draft(3, Some(other)));
        }
    }
    for (mat, other) in [(a, b), (b, a)] {
        if mat == other || tree.root(mat) || tree.node(other).form.name() != "f32" {
            continue;
        }
        let Some(site) = mm_site(tree, mat) else {
            continue;
        };
        if tree.elements(other) == site.m * site.n && tree.node(other).shape == node.shape {
            return Some(site.draft(4, Some(other)));
        }
    }
    None
}

pub(crate) fn propose_mm(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    if let Some(drafts) = propose_mm_add(tree, at) {
        return Some(drafts);
    }
    if let Some(drafts) = propose_mm_mul(tree, at) {
        return Some(drafts);
    }
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Act(Act::Silu)) && node.form.name() == "f32" {
        let mat = node.args[0];
        if !tree.root(mat)
            && let Some(site) = mm_site(tree, mat)
        {
            return Some(site.draft(5, None));
        }
    }
    Some(mm_site(tree, at)?.draft(0, None))
}

fn why_mm(tree: &Tree, at: T) -> Option<String> {
    check_mm(tree, at)?.err()
}

tessera_lower::rule!("fast_mm", propose_mm, why = why_mm, lacks = CUDA_LACKS);
