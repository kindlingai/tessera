//! Streaming BF16 and MXFP8 Tensor-Core small-batch matmul rule.

use super::{CUDA_LACKS, cuda_source_entry, f32_only, weight_accepts};
use tessera_graph::{Act, Op, Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Term, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmKind {
    Bf16,
    Bf16Split,
    Bf16SplitK,
    Mxfp8,
    Mxfp8Split,
    Mxfp8Wide,
    Mxfp8WideSplit,
    Mxfp8Tall,
    Mxfp8TallSplit,
}

impl MmKind {
    fn name(self) -> &'static str {
        match self {
            MmKind::Bf16 | MmKind::Bf16Split => "fast_mm_bf16",
            MmKind::Bf16SplitK => "fast_mm_bf16_sk",
            MmKind::Mxfp8 | MmKind::Mxfp8Wide | MmKind::Mxfp8Tall => "fast_mm_mxfp8",
            MmKind::Mxfp8Split | MmKind::Mxfp8WideSplit | MmKind::Mxfp8TallSplit => {
                "fast_mm_mxfp8_sk"
            }
        }
    }

    fn entry(self) -> &'static str {
        match self {
            MmKind::Bf16 => "tessera_mm_bf16",
            MmKind::Bf16Split => "tessera_mm_bf16_ks2",
            MmKind::Bf16SplitK => "tessera_mm_bf16_sk",
            MmKind::Mxfp8 => "tessera_mm_mxfp8_r8",
            MmKind::Mxfp8Split => "tessera_mm_mxfp8_r8_sk",
            MmKind::Mxfp8Wide => "tessera_mm_mxfp8_r16",
            MmKind::Mxfp8WideSplit => "tessera_mm_mxfp8_r16_sk",
            MmKind::Mxfp8Tall => "tessera_mm_mxfp8_r64",
            MmKind::Mxfp8TallSplit => "tessera_mm_mxfp8_r64_sk",
        }
    }

    fn threads(self) -> usize {
        match self {
            MmKind::Mxfp8Tall | MmKind::Mxfp8TallSplit => 256,
            _ => 128,
        }
    }

    fn smem(self) -> usize {
        match self {
            MmKind::Bf16 => (2 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Bf16Split | MmKind::Bf16SplitK => (4 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Mxfp8 | MmKind::Mxfp8Split | MmKind::Mxfp8Wide | MmKind::Mxfp8WideSplit => {
                4 * 16 * 36 * 4
            }
            MmKind::Mxfp8Tall | MmKind::Mxfp8TallSplit => (64 * 136 + 64 * 36) * 4,
        }
    }

    fn grid(self, m: usize, n: usize, ks: usize) -> (usize, usize) {
        match self {
            MmKind::Bf16 => (n / 32, m.div_ceil(16)),
            MmKind::Bf16Split | MmKind::Mxfp8Wide => (n / 16, m.div_ceil(16)),
            MmKind::Bf16SplitK => ((n / 16) * ks, m.div_ceil(16)),
            MmKind::Mxfp8 => (n / 16, 1),
            MmKind::Mxfp8Split | MmKind::Mxfp8WideSplit => ((n / 16) * ks, 1),
            MmKind::Mxfp8Tall => (n.div_ceil(64), m.div_ceil(64)),
            MmKind::Mxfp8TallSplit => (n.div_ceil(64) * ks, m.div_ceil(64)),
        }
    }

    fn step(self) -> f64 {
        match self {
            MmKind::Bf16 => 64.0,
            MmKind::Bf16Split | MmKind::Bf16SplitK => 128.0,
            MmKind::Mxfp8
            | MmKind::Mxfp8Split
            | MmKind::Mxfp8Wide
            | MmKind::Mxfp8WideSplit
            | MmKind::Mxfp8Tall
            | MmKind::Mxfp8TallSplit => 512.0,
        }
    }

    fn is_split(self) -> bool {
        matches!(
            self,
            MmKind::Bf16SplitK
                | MmKind::Mxfp8Split
                | MmKind::Mxfp8WideSplit
                | MmKind::Mxfp8TallSplit
        )
    }
}

fn pick_divisor(chunks: usize, want: usize, cap: usize) -> usize {
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

tessera_lower::knob!("fast_mm_bf16_sk", "splits", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "splits", 1, 16);

/// The split count a rule set, fitted to `chunks`.
fn set_ks(tree: &Tree, kernel: &str, chunks: usize) -> Option<usize> {
    let want = tree.knobs.get(kernel, "splits")?;
    Some(pick_divisor(chunks, want, want))
}

fn bf16_ks(m: usize, n: usize, k: usize) -> usize {
    let chunks = k / 128;
    let base = (n / 16) * m.div_ceil(16);
    if chunks < 4 || base >= 192 {
        return 1;
    }
    let cap = if base >= 48 { 6 } else { 16 }.min(chunks / 2);
    let want = 320usize.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

fn mxfp8_ks(n: usize, k: usize, wide: bool) -> usize {
    let chunks = k / 512;
    let nb = n / 16;
    let max_nb = if wide { 112 } else { 256 };
    if chunks <= 1 || nb >= max_nb {
        return 1;
    }
    let cap = if nb >= 32 { 6 } else { 12 }.min(chunks);
    let want = 320usize.div_ceil(nb).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

fn mxfp8_tall_ks(m: usize, n: usize, k: usize) -> usize {
    let chunks = k / 512;
    let base = n.div_ceil(64) * m.div_ceil(64);
    if chunks <= 1 || base >= 160 {
        return 1;
    }
    let cap = if base >= 64 { 4 } else { 12 }.min(chunks);
    let want = 192usize.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
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
    ks: usize,
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
        group_threads: k.kind.threads(),
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn mm_load(k: &FastMm, _: usize) -> Option<Load> {
    let (across, down) = k.kind.grid(k.m, k.n, k.ks);
    let groups = across * down;
    let threads = k.kind.threads();
    let mut reads = vec![k.form.bytes(k.n * k.k) as f64, (4 * k.m * k.k) as f64];
    if matches!(k.mode, 1 | 3 | 4) {
        reads.push((4 * k.m * k.n) as f64);
    } else if k.mode == 2 {
        reads.push((4 * k.n) as f64);
    }
    let mut load = Load {
        threads: groups * threads,
        group: threads,
        trips: k.k as f64 / (k.kind.step() * k.ks as f64),
        reads,
        written: (4 * k.ks * k.m * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn mm_launches(k: &FastMm, step: &Step) -> Vec<Launch> {
    let (across, down) = k.kind.grid(k.m, k.n, k.ks);
    let shape = Shape::Grid {
        across,
        down,
        threads: k.kind.threads(),
    };
    let other = if matches!(k.mode, 1..=4) {
        step.read(2)
    } else {
        step.read(0)
    };
    let mut args = vec![
        step.read(0),
        step.read(1),
        other,
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
    ];
    if k.kind.is_split() {
        args.push(Arg::count(k.ks));
    }
    vec![Launch { args, shape }]
}

/// Deterministic split-K reduction and matmul epilogue.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_mm_reduce",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = mm_red_accepts,
    emits = f32_only,
    needs = mm_red_needs(),
    order = Order::PerConfig,
    source = cuda_source_entry("tessera_mm_reduce"),
    launches = mm_red_launches,
    load = mm_red_load,
    unproven,
)]
struct FastMmReduce {
    m: usize,
    n: usize,
    ks: usize,
    mode: u32,
}

fn mm_red_accepts(_: &FastMmReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn mm_red_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 0,
        group_threads: 256,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn mm_red_load(r: &FastMmReduce, _: usize) -> Option<Load> {
    let mut reads = vec![(4 * r.ks * r.m * r.n) as f64];
    if matches!(r.mode, 1 | 3 | 4) {
        reads.push((4 * r.m * r.n) as f64);
    } else if r.mode == 2 {
        reads.push((4 * r.n) as f64);
    }
    Some(Load {
        threads: (r.m * r.n).div_ceil(256) * 256,
        group: 256,
        trips: r.ks as f64,
        reads,
        written: (4 * r.m * r.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn mm_red_launches(r: &FastMmReduce, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (r.m * r.n).div_ceil(256),
        down: 1,
        threads: 256,
    };
    let other = if matches!(r.mode, 1..=4) {
        step.read(1)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        other,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::value(&r.mode),
    ];
    vec![Launch { args, shape }]
}

struct MmSite {
    x: T,
    w: T,
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    kind: MmKind,
    form: &'static dyn Stored,
}

impl MmSite {
    fn draft(self, mode: u32, other: Option<T>) -> Vec<Draft> {
        if self.kind.is_split() {
            let mm = FastMm {
                m: self.m,
                n: self.n,
                k: self.k,
                ks: self.ks,
                mode: 0,
                kind: self.kind,
                form: self.form,
            };
            let red = FastMmReduce {
                m: self.m,
                n: self.n,
                ks: self.ks,
                mode,
            };
            let mut red_reads = vec![Read::Draft(0)];
            if let Some(extra) = other {
                red_reads.push(Read::Node(extra));
            }
            return vec![
                Draft::new(mm, [self.x, self.w], self.ks * self.m * self.n),
                Draft::new(red, red_reads, self.m * self.n),
            ];
        }
        let kernel = FastMm {
            m: self.m,
            n: self.n,
            k: self.k,
            ks: 1,
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
    if m > crate::FAST_MAX_ROWS {
        return Some(Err(format!("m={m} exceeds max {}", crate::FAST_MAX_ROWS)));
    }
    let form = tree.node(w).form;
    let (kind, ks) = match form.name() {
        "bf16"
            if n.is_multiple_of(16)
                && k.is_multiple_of(128)
                && (!n.is_multiple_of(32) || (n / 32) * m.div_ceil(16) < 144) =>
        {
            let set = set_ks(tree, "fast_mm_bf16_sk", k / 128);
            let ks = set.unwrap_or_else(|| bf16_ks(m, n, k));
            if ks > 1 {
                (MmKind::Bf16SplitK, ks)
            } else {
                (MmKind::Bf16Split, 1)
            }
        }
        "bf16" if n.is_multiple_of(32) && k.is_multiple_of(64) => (MmKind::Bf16, 1),
        "bf16" => {
            return Some(Err(format!(
                "n={n}, k={k} not aligned for bf16 (needs n%32 & k%64, or n%16 & k%128)"
            )));
        }
        "mxfp8" if n.is_multiple_of(16) && k.is_multiple_of(512) => {
            let set = set_ks(tree, "fast_mm_mxfp8_sk", k / 512);
            if m <= 8 {
                let ks = set.unwrap_or_else(|| mxfp8_ks(n, k, false));
                if ks > 1 {
                    (MmKind::Mxfp8Split, ks)
                } else {
                    (MmKind::Mxfp8, 1)
                }
            } else if m <= 16 {
                let ks = set.unwrap_or_else(|| mxfp8_ks(n, k, true));
                if ks > 1 {
                    (MmKind::Mxfp8WideSplit, ks)
                } else {
                    (MmKind::Mxfp8Wide, 1)
                }
            } else {
                let ks = set.unwrap_or_else(|| mxfp8_tall_ks(m, n, k));
                if ks > 1 {
                    (MmKind::Mxfp8TallSplit, ks)
                } else {
                    (MmKind::Mxfp8Tall, 1)
                }
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
        ks,
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

super::cuda_rule!("fast_mm", propose_mm, why = why_mm);
