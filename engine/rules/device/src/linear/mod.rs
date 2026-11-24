//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

mod gemv;

use crate::{Backend, Options, flatten};
use gemv::propose_gemv;
use std::sync::OnceLock;
use tessera_graph::{Act, Op, Order, Stage, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

const KERNEL: &str = include_str!("linear.cu");
const BLOCK: usize = 128;
const THREADS: usize = 256;
const WARPS: usize = 8;
const DTR_SMEM: usize = 4096 * 4;

fn flat(backend: Backend) -> &'static str {
    let options = Options {
        backend,
        remote_kv: false,
        marks: false,
    };
    let text = flatten("linear.kernel", KERNEL, &|_| None, &options)
        .expect("linear kernel flattens")
        .text;
    Box::leak(text.into_boxed_str())
}

pub(crate) fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flat(Backend::Cuda))
}

#[cfg(test)]
pub(crate) fn hip_source() -> &'static str {
    static HIP: OnceLock<&'static str> = OnceLock::new();
    HIP.get_or_init(|| flat(Backend::Rocm))
}

pub(crate) fn parse_trellis(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("trellis_k")?;
    let (k, book) = if let Some(k) = rest.strip_suffix("_pmcgx") {
        (k, 2u32)
    } else if let Some(k) = rest.strip_suffix("_mul1") {
        (k, 1u32)
    } else {
        (rest, 0u32)
    };
    let bits: u32 = k.parse().ok()?;
    (2..=4).contains(&bits).then_some((bits, book))
}

fn dtr_ks(m: usize, n: usize, k: usize) -> usize {
    let ibs = k / BLOCK;
    if m > 16 || ibs <= 1 {
        return 1;
    }
    let target: usize = if m <= 2 { 288 } else { 128 };
    let want = target.div_ceil(n / BLOCK).clamp(1, 16).min(ibs);
    for d in want..=ibs.min(16) {
        if ibs.is_multiple_of(d) {
            return d;
        }
    }
    for d in (1..want).rev() {
        if ibs.is_multiple_of(d) {
            return d;
        }
    }
    1
}

fn dtr_tile(m: usize) -> usize {
    if m <= 8 {
        8
    } else if m <= 16 {
        16
    } else {
        32
    }
}

fn dtr_entry(t: &TrellisMma) -> &'static str {
    match (dtr_tile(t.m), t.bits, t.book, t.turned, t.ks > 1) {
        (8, 4, 0, true, true) => "dtr_k4_t1s",
        (8, 4, 0, true, false) => "dtr_k4_t1",
        (8, 4, 0, false, true) => "dtr_k4_p1s",
        (8, 4, 0, false, false) => "dtr_k4_p1",
        (16, 4, 0, true, true) => "dtr_k4_t2s",
        (16, 4, 0, true, false) => "dtr_k4_t2",
        (32, 4, 0, true, false) => "dtr_k4_t4",
        (8, _, _, _, _) => "dtr_gen1",
        (16, _, _, _, _) => "dtr_gen2",
        (_, _, _, _, _) => "dtr_gen4",
    }
}

fn dtr_needs(bytes: usize) -> Needs {
    Needs {
        bf16_tile: true,
        ..cuda_needs(bytes)
    }
}

/// Dense trellis matmul with warp-per-tile MMA and split-K.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "trellis_turned" } else { "trellis_mma" }),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = mma_accepts,
    emits = f32_only,
    needs = dtr_needs(DTR_SMEM),
    order = Order::PerConfig,
    source = cuda_source_entry(dtr_entry(self)),
    launches = mma_launches,
    load = mma_load,
    unproven,
)]
struct TrellisMma {
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    bits: u32,
    book: u32,
    turned: bool,
    form: &'static dyn Stored,
}

fn mma_accepts(k: &TrellisMma, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn mma_load(t: &TrellisMma, _: usize) -> Option<Load> {
    let groups = (t.n / BLOCK) * t.ks * t.m.div_ceil(dtr_tile(t.m));
    let mut reads = vec![t.form.bytes(t.n * t.k) as f64, (4 * t.m * t.k) as f64];
    if t.turned {
        reads.push((4 * t.k) as f64);
        if t.ks == 1 {
            reads.push((4 * t.n) as f64);
        }
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (t.k / BLOCK / t.ks) as f64,
        reads,
        written: (4 * t.ks * t.m * t.n) as f64,
        flight: 1024.0,
        ..Load::default()
    })
}

fn mma_launches(t: &TrellisMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(dtr_tile(t.m)),
        threads: THREADS,
    };
    let suh = if t.turned { step.read(2) } else { step.read(0) };
    let svh = if t.turned && t.ks == 1 {
        step.read(3)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        suh,
        svh,
        step.out(),
        Arg::count(t.m),
        Arg::count(t.n),
        Arg::count(t.k),
        Arg::count(t.ks),
        Arg::value(&t.bits),
        Arg::value(&t.book),
        Arg::count(usize::from(t.turned)),
    ];
    vec![Launch { args, shape }]
}

/// Deterministic split-K reduction and optional output rotation.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "trellis_reduce",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = reduce_accepts,
    emits = f32_only,
    needs = dtr_needs(0),
    order = Order::PerConfig,
    source = cuda_source_entry("dtr_reduce"),
    launches = reduce_launches,
    load = reduce_load,
    unproven,
)]
struct TrellisReduce {
    m: usize,
    n: usize,
    ks: usize,
    turned: bool,
}

fn reduce_accepts(_: &TrellisReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn reduce_load(r: &TrellisReduce, _: usize) -> Option<Load> {
    let mut reads = vec![(4 * r.ks * r.m * r.n) as f64];
    if r.turned {
        reads.push((4 * r.n) as f64);
    }
    Some(Load {
        threads: (r.n / BLOCK) * THREADS,
        group: THREADS,
        trips: r.ks as f64,
        reads,
        written: (4 * r.m * r.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_launches(r: &TrellisReduce, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: r.n / BLOCK,
        down: 1,
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let args = vec![
        step.read(0),
        svh,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::count(usize::from(r.turned)),
    ];
    vec![Launch { args, shape }]
}

fn step_args<'a>(tree: &'a Tree, t: T, term: Term) -> Option<&'a [T]> {
    let node = tree.node(t);
    (node.term == term && !tree.root(t)).then_some(&node.args[..])
}

fn trellis_drafts(t: TrellisMma, x: T, w: T, suh: Option<T>, svh: Option<T>) -> Vec<Draft> {
    let (m, n, ks, turned) = (t.m, t.n, t.ks, t.turned);
    if ks == 1 {
        let mut reads = vec![x, w];
        if let (Some(u), Some(v)) = (suh, svh) {
            reads.extend([u, v]);
        }
        return vec![Draft::new(t, reads, m * n)];
    }
    let mut s_reads = vec![Read::Node(x), Read::Node(w)];
    if let Some(u) = suh {
        s_reads.push(Read::Node(u));
    }
    let mut r_reads = vec![Read::Draft(0)];
    if let Some(v) = svh {
        r_reads.push(Read::Node(v));
    }
    let red = TrellisReduce { m, n, ks, turned };
    vec![
        Draft::new(t, s_reads, ks * m * n),
        Draft::new(red, r_reads, m * n),
    ]
}

fn trellis_kernel(tree: &Tree, x: T, w: T, turned: bool) -> Option<TrellisMma> {
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    let form = tree.node(w).form;
    let (bits, book) = parse_trellis(form.name())?;
    let m = tree.elements(x) / k;
    let ok = tree.leaf(w)
        && n.is_multiple_of(BLOCK)
        && k.is_multiple_of(BLOCK)
        && m > 0
        && tree.node(x).form.name() == "f32";
    ok.then(|| TrellisMma {
        m,
        n,
        k,
        ks: dtr_ks(m, n, k),
        bits,
        book,
        turned,
        form,
    })
}

/// Rotated or plain dense trellis matmul.
fn propose_trellis(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let turn = Term::Stage(Stage::Hadamard {
        block: BLOCK as u32,
    });
    let mul = Term::Stage(Stage::Mul);
    let node = tree.node(at);
    if node.form.name() != "f32" {
        return None;
    }
    if node.term == mul {
        let (back, svh) = (node.args[0], *node.args.get(1)?);
        let &[product] = step_args(tree, back, turn.clone())? else {
            return None;
        };
        let &[turned, w] = step_args(tree, product, Term::Op(Op::Matmul))? else {
            return None;
        };
        let &[scaled] = step_args(tree, turned, turn)? else {
            return None;
        };
        let &[x, suh] = step_args(tree, scaled, mul)? else {
            return None;
        };
        let kernel = trellis_kernel(tree, x, w, true)?;
        let ok = tree.elements(suh) == kernel.k
            && tree.elements(svh) == kernel.n
            && tree.node(suh).form.name() == "f32"
            && tree.node(svh).form.name() == "f32";
        return ok.then(|| trellis_drafts(kernel, x, w, Some(suh), Some(svh)));
    }
    if node.term == Term::Op(Op::Matmul) {
        let (x, w) = (node.args[0], node.args[1]);
        let kernel = trellis_kernel(tree, x, w, false)?;
        return Some(trellis_drafts(kernel, x, w, None, None));
    }
    None
}

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn cuda_needs(group_bytes: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads: THREADS,
        compiler: true,
        lacks: &[Backend::Metal, Backend::Rocm],
        ..Needs::default()
    }
}

fn cuda_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", cuda_source(), ""]),
        entry,
    }
}

fn weight_accepts(input: usize, expected: &dyn Stored, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == expected.name(),
        _ => form.name() == "f32",
    }
}

/// Fast CUDA stacked BF16 matmul, transposed or row-major.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.wide { "fast_stacked_wide" } else { "fast_stacked_lanes" },
    sizes = [self.batches, self.rows, self.n, self.k],
    work = self.len * self.k,
    accepts = stacked_accepts,
    emits = f32_only,
    needs = stacked_needs(self),
    order = Order::PerConfig,
    source = stacked_source(self),
    launches = stacked_launches,
    load = stacked_load,
    unproven,
)]
struct FastStacked {
    len: usize,
    batches: usize,
    rows: usize,
    n: usize,
    k: usize,
    block: usize,
    first: usize,
    step: usize,
    wide: bool,
    form: &'static dyn Stored,
}

fn stacked_accepts(k: &FastStacked, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn stacked_needs(k: &FastStacked) -> Needs {
    cuda_needs(if k.wide { WARPS * 128 * 4 } else { 0 })
}

fn stacked_source(k: &FastStacked) -> Source {
    cuda_source_entry(if k.wide {
        "tessera_stacked_wide"
    } else {
        "tessera_stacked_lanes"
    })
}

fn stacked_load(k: &FastStacked, _: usize) -> Option<Load> {
    let rtiles = if k.wide {
        k.rows.div_ceil(4)
    } else {
        k.rows.div_ceil(8)
    };
    let groups = if k.wide {
        (k.n / BLOCK) * k.batches * rtiles
    } else {
        (k.batches * k.n).div_ceil(WARPS) * rtiles
    };
    let trips = if k.wide {
        k.k as f64 / WARPS as f64
    } else {
        k.k as f64 / 256.0
    };
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips,
        reads: vec![
            (2 * k.batches * rtiles * k.n * k.k) as f64,
            (4 * k.batches * k.rows * k.k) as f64,
        ],
        written: (4 * k.len) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn stacked_launches(k: &FastStacked, step: &Step) -> Vec<Launch> {
    if k.wide {
        let shape = Shape::Grid {
            across: k.n / BLOCK,
            down: k.batches * k.rows.div_ceil(4),
            threads: THREADS,
        };
        let args = vec![
            step.read(0),
            step.read(1),
            step.out(),
            Arg::count(k.rows),
            Arg::count(k.n),
            Arg::count(k.k),
            Arg::count(k.block),
            Arg::count(k.first),
            Arg::count(k.step),
        ];
        return vec![Launch { args, shape }];
    }
    let bn = k.batches * k.n;
    let shape = Shape::Grid {
        across: bn.div_ceil(WARPS),
        down: k.rows.div_ceil(8),
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(bn),
        Arg::count(k.rows),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::count(k.block),
        Arg::count(k.first),
        Arg::count(k.step),
    ];
    vec![Launch { args, shape }]
}

fn propose_stacked(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) || node.form.name() != "f32" {
        return None;
    }
    let (x, mut w) = (node.args[0], node.args[1]);
    if tree.node(x).form.name() != "f32" {
        return None;
    }
    let &[batches, n, k] = tree.node(w).shape.as_slice() else {
        return None;
    };
    let &[xb, rows, xk] = tree.node(x).shape.as_slice() else {
        return None;
    };
    if xb != batches || xk != k || rows == 0 {
        return None;
    }
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
    let (block, first) = (block_rows * width, first_row * width);
    let align = if swapped { 4 } else { 8 };
    let ok = block.is_multiple_of(align)
        && first.is_multiple_of(align)
        && width.is_multiple_of(align)
        && if swapped {
            n.is_multiple_of(BLOCK) && k <= 512
        } else {
            k.is_multiple_of(8)
        };
    if !ok {
        return None;
    }
    let len = batches * rows * n;
    let kernel = FastStacked {
        len,
        batches,
        rows,
        n,
        k,
        block,
        first,
        step: width,
        wide: swapped,
        form,
    };
    Some(vec![Draft::new(kernel, [x, leaf], len)])
}

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
        lacks: &[Backend::Metal, Backend::Rocm],
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

fn mm_site(tree: &Tree, mat: T) -> Option<MmSite> {
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) || node.form.name() != "f32" {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    if !tree.leaf(w) || tree.node(x).form.name() != "f32" {
        return None;
    }
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
    if !(2..=64).contains(&m) {
        return None;
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
        "mxfp8" if n.is_multiple_of(16) && k.is_multiple_of(512) => {
            if m <= 8 {
                MmKind::Mxfp8
            } else if m <= 16 {
                MmKind::Mxfp8Wide
            } else {
                MmKind::Mxfp8Tall
            }
        }
        _ => return None,
    };
    Some(MmSite {
        x,
        w,
        m,
        n,
        k,
        kind,
        form,
    })
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

tessera_lower::rule!("trellis_mma", propose_trellis);
tessera_lower::rule!("fast_gemv", propose_gemv);
tessera_lower::rule!("fast_mm", propose_mm);
tessera_lower::rule!("fast_stacked", propose_stacked);
