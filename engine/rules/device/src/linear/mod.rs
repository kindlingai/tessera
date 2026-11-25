//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

mod gemv;
pub(crate) mod mm;

use crate::{Backend, Options, flatten};
use gemv::{propose_gemv, why_gemv};
use std::sync::OnceLock;
use tessera_graph::{Op, Order, Stage, View};
use tessera_lower::Backend as LowerBackend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Ops, Read, Shape, Source, Step, T, Term,
    Tree,
};

pub(crate) const CUDA_LACKS: &[LowerBackend] = &[LowerBackend::Metal, LowerBackend::Rocm];

const KERNEL: &str = include_str!("linear.cu");
const BLOCK: usize = 128;
const WIDE_COLS: usize = 32;
const WIDE_THREADS: usize = 128;
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

fn check_trellis(tree: &Tree, x: T, w: T, turned: bool) -> Option<Result<TrellisMma, String>> {
    let form = tree.node(w).form;
    if !form.name().starts_with("trellis") {
        return None;
    }
    let Some((bits, book)) = parse_trellis(form.name()) else {
        return Some(Err(format!("unsupported trellis form {}", form.name())));
    };
    let [n, k] = tree.node(w).shape[..] else {
        return Some(Err("trellis weight is not 2D".into()));
    };
    if !tree.leaf(w) {
        return Some(Err("trellis weight is not a direct leaf".into()));
    }
    if !n.is_multiple_of(BLOCK) {
        return Some(Err(format!(
            "n={n} is not a multiple of {BLOCK} for trellis"
        )));
    }
    if !k.is_multiple_of(BLOCK) {
        return Some(Err(format!(
            "k={k} is not a multiple of {BLOCK} for trellis"
        )));
    }
    if tree.node(x).form.name() != "f32" {
        return Some(Err(format!(
            "input form {} (requires f32)",
            tree.node(x).form.name()
        )));
    }
    let m = tree.elements(x) / k;
    if m == 0 {
        return Some(Err("m=0".into()));
    }
    Some(Ok(TrellisMma {
        m,
        n,
        k,
        ks: dtr_ks(m, n, k),
        bits,
        book,
        turned,
        form,
    }))
}

/// Rotated or plain dense trellis matmul.
fn eval_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let turn = Term::Stage(Stage::Hadamard {
        block: BLOCK as u32,
    });
    let mul = Term::Stage(Stage::Mul);
    let node = tree.node(at);
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
        if node.form.name() != "f32" {
            return Some(Err(format!(
                "output form {} (requires f32)",
                node.form.name()
            )));
        }
        let kernel = match check_trellis(tree, x, w, true)? {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        if tree.elements(suh) != kernel.k || tree.elements(svh) != kernel.n {
            return Some(Err("rotation scale length mismatch".into()));
        }
        if tree.node(suh).form.name() != "f32" || tree.node(svh).form.name() != "f32" {
            return Some(Err("rotation scales must be f32".into()));
        }
        return Some(Ok(trellis_drafts(kernel, x, w, Some(suh), Some(svh))));
    }
    if node.term == Term::Op(Op::Matmul) {
        let (x, w) = (node.args[0], node.args[1]);
        if !tree.node(w).form.name().starts_with("trellis") {
            return None;
        }
        if node.form.name() != "f32" {
            return Some(Err(format!(
                "output form {} (requires f32)",
                node.form.name()
            )));
        }
        let kernel = match check_trellis(tree, x, w, false)? {
            Ok(k) => k,
            Err(e) => return Some(Err(e)),
        };
        return Some(Ok(trellis_drafts(kernel, x, w, None, None)));
    }
    None
}

fn propose_trellis(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    eval_trellis(tree, at)?.ok()
}

fn why_trellis(tree: &Tree, at: T) -> Option<String> {
    eval_trellis(tree, at)?.err()
}

/// One thread's trip. Bf16 widens with one integer op.
fn trip(float: usize, int: usize, loads: usize, bytes: usize) -> Ops {
    Ops {
        float: float as f64,
        int: int as f64,
        loads: loads as f64,
        width: bytes as f64 / loads as f64,
        fresh: 1.0,
        ..Ops::default()
    }
}

pub(crate) fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn cuda_needs(group_bytes: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads: THREADS,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

pub(super) fn cuda_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", cuda_source(), ""]),
        entry,
    }
}

pub(super) fn weight_accepts(input: usize, expected: &dyn Stored, form: &dyn Stored) -> bool {
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

impl FastStacked {
    fn lanes_tc(&self) -> bool {
        !self.wide && self.n.is_multiple_of(16) && self.k.is_multiple_of(128)
    }
}

fn stacked_accepts(k: &FastStacked, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn stacked_needs(k: &FastStacked) -> Needs {
    if k.wide || k.lanes_tc() {
        Needs {
            lanes: 32,
            group_bytes: if k.wide {
                16 * 18 * 4 * 2 + 32 * 34 * 2
            } else {
                (4 * 16 * 36 + 4 * 8 * 36) * 4
            },
            group_threads: WIDE_THREADS,
            compiler: true,
            lacks: CUDA_LACKS,
            bf16_tile: true,
            ..Needs::default()
        }
    } else {
        cuda_needs(0)
    }
}

fn stacked_source(k: &FastStacked) -> Source {
    cuda_source_entry(if k.wide {
        "tessera_stacked_wide"
    } else if k.lanes_tc() {
        "tessera_stacked_lanes_tc"
    } else {
        "tessera_stacked_lanes"
    })
}

fn stacked_load(k: &FastStacked, _: usize) -> Option<Load> {
    if k.lanes_tc() {
        let rtiles = k.rows.div_ceil(16);
        let groups = (k.n / 16) * k.batches * rtiles;
        let mut load = Load {
            threads: groups * 128,
            group: 128,
            trips: k.k as f64 / 128.0,
            reads: vec![
                (2 * k.batches * rtiles * k.n * k.k) as f64,
                (4 * k.batches * k.rows * k.k) as f64,
            ],
            written: (4 * k.len) as f64,
            flight: 1024.0,
            ..Load::default()
        };
        load.trip.fresh = 1.0;
        return Some(load);
    }
    let rtiles = if k.wide {
        k.rows.div_ceil(16)
    } else {
        k.rows.div_ceil(8)
    };
    let (groups, group) = if k.wide {
        ((k.n / WIDE_COLS) * k.batches * rtiles, WIDE_THREADS)
    } else {
        ((k.batches * k.n).div_ceil(WARPS) * rtiles, THREADS)
    };
    // A trip reads weights and each row's x.
    let (trips, trip) = if k.wide {
        (k.k as f64 / 32.0, Ops::default())
    } else {
        let r = k.rows.min(8);
        (k.k as f64 / 256.0, trip(8 * r, 8, 1 + 2 * r, 16 + 32 * r))
    };
    Some(Load {
        threads: groups * group,
        group,
        trips,
        trip,
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
    if k.wide || k.lanes_tc() {
        let shape = Shape::Grid {
            across: k.n / if k.wide { WIDE_COLS } else { 16 },
            down: k.batches * k.rows.div_ceil(16),
            threads: WIDE_THREADS,
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

fn check_stacked(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, mut w) = (node.args[0], node.args[1]);
    let &[batches, n, k] = tree.node(w).shape.as_slice() else {
        return None;
    };
    let &[xb, rows, xk] = tree.node(x).shape.as_slice() else {
        return None;
    };
    if xb != batches || xk != k || rows == 0 {
        return None;
    }
    if node.form.name() != "f32" {
        return Some(Err(format!(
            "output form {} (requires f32)",
            node.form.name()
        )));
    }
    if tree.node(x).form.name() != "f32" {
        return Some(Err(format!(
            "input form {} (requires f32)",
            tree.node(x).form.name()
        )));
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
        let Some(offset) = start.value(&tree.g.dims, &|d| tree.size(d)) else {
            return Some(Err("slice start is dynamic".into()));
        };
        first_row = offset as usize;
        w = tree.node(w).args[0];
    }
    if !matches!(&tree.node(w).term, Term::Op(Op::View(View::Reshape(_)))) {
        return Some(Err("weight is not a reshaped 2D leaf".into()));
    }
    let leaf = tree.node(w).args[0];
    let &[_, block_rows, width] = tree.node(w).shape.as_slice() else {
        return Some(Err("reshaped weight is not 3D".into()));
    };
    let &[_, stored_w] = tree.node(leaf).shape.as_slice() else {
        return Some(Err("weight leaf is not 2D".into()));
    };
    let form = tree.node(leaf).form;
    if !tree.leaf(leaf) {
        return Some(Err("weight source is not a direct leaf".into()));
    }
    if stored_w != width {
        return Some(Err(format!(
            "weight width {stored_w} differs from view width {width}"
        )));
    }
    if form.name() != "bf16" {
        return Some(Err(format!("weight form {} (supports bf16)", form.name())));
    }
    let (block, first) = (block_rows * width, first_row * width);
    let align = 8;
    if !block.is_multiple_of(align) || !first.is_multiple_of(align) || !width.is_multiple_of(align)
    {
        return Some(Err(format!(
            "strides (width={width}, block={block}, first={first}) not multiple of {align}"
        )));
    }
    if swapped {
        if !n.is_multiple_of(WIDE_COLS) {
            return Some(Err(format!(
                "n={n} is not a multiple of {WIDE_COLS} for transposed stacked"
            )));
        }
        if !k.is_multiple_of(32) || k > 512 {
            return Some(Err(format!("k={k} must be a multiple of 32 up to 512")));
        }
    } else if !k.is_multiple_of(8) {
        return Some(Err(format!("k={k} is not a multiple of 8 for stacked")));
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
    Some(Ok(vec![Draft::new(kernel, [x, leaf], len)]))
}

fn propose_stacked(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    check_stacked(tree, at)?.ok()
}

fn why_stacked(tree: &Tree, at: T) -> Option<String> {
    check_stacked(tree, at)?.err()
}

tessera_lower::rule!(
    "trellis_mma",
    propose_trellis,
    why = why_trellis,
    lacks = CUDA_LACKS
);
tessera_lower::rule!(
    "fast_gemv",
    propose_gemv,
    why = why_gemv,
    lacks = CUDA_LACKS
);
tessera_lower::rule!(
    "fast_stacked",
    propose_stacked,
    why = why_stacked,
    lacks = CUDA_LACKS
);
