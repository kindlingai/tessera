//! Device-accelerated trellis MMA and streaming decode GEMV rules.

use crate::{Backend, Options, flatten};
use std::sync::OnceLock;
use tessera_graph::{Op, Order, Stage, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree};

const KERNEL: &str = include_str!("linear.cu");
const BLOCK: usize = 128;
const ROWS: usize = 4;
const THREADS: usize = 256;
const WARPS: usize = 8;

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

fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flat(Backend::Cuda))
}

fn hip_source() -> &'static str {
    static HIP: OnceLock<&'static str> = OnceLock::new();
    HIP.get_or_init(|| flat(Backend::Hip))
}

fn trellis_choice(t: &TrellisMma) -> (&'static str, usize) {
    let entry = match (t.m, t.bits, t.book, t.turned) {
        (1, 4, 0, true) => "tessera_trellis_k4_mcg_t1",
        (1, 4, 0, false) => "tessera_trellis_k4_mcg_p1",
        (1, 3, 1, true) => "tessera_trellis_k3_mul1_t1",
        (1, 4, 1, true) => "tessera_trellis_k4_mul1_t1",
        (1, _, _, _) => "tessera_trellis_mma1",
        _ => "tessera_trellis_mma",
    };
    (entry, WARPS)
}

fn trellis_source(t: &TrellisMma) -> Source {
    Source::Fixed {
        metal: "",
        cuda: cuda_source(),
        hip: hip_source(),
        entry: trellis_choice(t).0,
    }
}

/// Bit width and codebook index of a trellis form name.
fn parse_trellis(name: &str) -> Option<(u32, u32)> {
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

/// Plain or rotated trellis matmul using tensor-core warp MMA.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "trellis_turned" } else { "trellis_mma" }),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = mma_accepts,
    emits = f32_only,
    needs = mma_needs(self),
    order = Order::PerConfig,
    source = trellis_source(self),
    launches = mma_launches,
    load = mma_load,
    unproven,
)]
struct TrellisMma {
    m: usize,
    n: usize,
    k: usize,
    bits: u32,
    book: u32,
    turned: bool,
    form: &'static dyn Stored,
}

fn mma_accepts(k: &TrellisMma, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        _ => form.name() == "f32",
    }
}

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn mma_needs(t: &TrellisMma) -> Needs {
    let warps = trellis_choice(t).1;
    let rows = if t.m == 1 { 1 } else { ROWS };
    Needs {
        lanes: 32,
        group_bytes: warps * 256 * 4 + warps * rows * 128 * 4,
        group_threads: warps * 32,
        compiler: true,
        lacks: &["metal"],
        bf16_tile: true,
        ..Needs::default()
    }
}

fn mma_load(t: &TrellisMma, _: usize) -> Option<Load> {
    let threads = trellis_choice(t).1 * 32;
    let groups = (t.n / BLOCK) * t.m.div_ceil(ROWS);
    let mut reads = vec![t.form.bytes(t.n * t.k) as f64, (4 * t.m * t.k) as f64];
    if t.turned {
        reads.push((4 * t.k) as f64);
        reads.push((4 * t.n) as f64);
    }
    Some(Load {
        threads: groups * threads,
        group: threads,
        trips: (t.k / 16) as f64 / 8.0,
        reads,
        written: (4 * t.m * t.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn mma_launches(t: &TrellisMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: t.n / BLOCK,
        down: t.m.div_ceil(ROWS),
        threads: trellis_choice(t).1 * 32,
    };
    let mut args = if t.turned {
        vec![step.read(0), step.read(1), step.read(2), step.read(3)]
    } else {
        vec![step.read(0), step.read(1), step.read(0), step.read(0)]
    };
    args.push(step.out());
    args.extend([
        Arg::count(t.m),
        Arg::count(t.n),
        Arg::count(t.k),
        Arg::value(&t.bits),
        Arg::value(&t.book),
        Arg::count(usize::from(t.turned)),
    ]);
    vec![Launch { args, shape }]
}

fn step_args<'a>(tree: &'a Tree, t: T, term: Term) -> Option<&'a [T]> {
    let node = tree.node(t);
    (node.term == term && !tree.root(t)).then_some(&node.args[..])
}

/// Rotated `Mul(Hadamard(Matmul(Hadamard(Mul(x, suh)), w)), svh)`.
fn propose_turned(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let turn = || {
        Term::Stage(Stage::Hadamard {
            block: BLOCK as u32,
        })
    };
    let mul = || Term::Stage(Stage::Mul);
    let node = tree.node(at);
    if node.term != mul() || node.form.name() != "f32" {
        return None;
    }
    let (back, svh) = (node.args[0], *node.args.get(1)?);
    let &[product] = step_args(tree, back, turn())? else {
        return None;
    };
    let &[turned, w] = step_args(tree, product, Term::Op(Op::Matmul))? else {
        return None;
    };
    let &[scaled] = step_args(tree, turned, turn())? else {
        return None;
    };
    let &[x, suh] = step_args(tree, scaled, mul())? else {
        return None;
    };
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
        && tree.elements(suh) == k
        && tree.elements(svh) == n
        && tree.node(x).form.name() == "f32"
        && tree.node(suh).form.name() == "f32"
        && tree.node(svh).form.name() == "f32";
    if !ok {
        return None;
    }
    let kernel = TrellisMma {
        m,
        n,
        k,
        bits,
        book,
        turned: true,
        form,
    };
    Some(vec![Draft::new(kernel, [x, w, suh, svh], m * n)])
}

/// Plain trellis `Matmul(x, w)`.
fn propose_plain(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) || node.form.name() != "f32" {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
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
    if !ok {
        return None;
    }
    let kernel = TrellisMma {
        m,
        n,
        k,
        bits,
        book,
        turned: false,
        form,
    };
    Some(vec![Draft::new(kernel, [x, w], m * n)])
}

fn propose_trellis(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose_turned(tree, at).or_else(|| propose_plain(tree, at))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemvKind {
    Bf16,
    Mxfp8,
    Int4,
}

impl GemvKind {
    fn entry(self) -> &'static str {
        match self {
            GemvKind::Bf16 => "tessera_gemv_bf16",
            GemvKind::Mxfp8 => "tessera_gemv_mxfp8",
            GemvKind::Int4 => "tessera_gemv_int4",
        }
    }

    fn name(self) -> &'static str {
        match self {
            GemvKind::Bf16 => "fast_gemv_bf16",
            GemvKind::Mxfp8 => "fast_gemv_mxfp8",
            GemvKind::Int4 => "fast_gemv_int4_g32",
        }
    }

    fn smem(self) -> usize {
        match self {
            GemvKind::Bf16 => 0,
            GemvKind::Mxfp8 => 512 * 4 + WARPS * 268 * 4,
            GemvKind::Int4 => 512 * 4 + WARPS * 296 * 4,
        }
    }

    fn groups(self, n: usize) -> usize {
        match self {
            GemvKind::Int4 => (n / 2).div_ceil(WARPS),
            GemvKind::Bf16 | GemvKind::Mxfp8 => n.div_ceil(WARPS),
        }
    }
}

/// Streaming decode GEMV for BF16, MXFP8, and INT4_G32 weights.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.name(),
    sizes = [1, self.n, self.k],
    work = self.n * self.k,
    accepts = gemv_accepts,
    emits = f32_only,
    needs = gemv_needs(self),
    order = Order::PerConfig,
    source = gemv_source(self),
    launches = gemv_launches,
    load = gemv_load,
    unproven,
)]
struct FastGemv {
    n: usize,
    k: usize,
    kind: GemvKind,
    form: &'static dyn Stored,
}

fn cuda_needs(group_bytes: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads: THREADS,
        compiler: true,
        lacks: &["metal", "rocm"],
        ..Needs::default()
    }
}

fn cuda_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        metal: "",
        cuda: cuda_source(),
        hip: "",
        entry,
    }
}

fn weight_accepts(input: usize, expected: &dyn Stored, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == expected.name(),
        _ => form.name() == "f32",
    }
}

fn gemv_accepts(k: &FastGemv, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn gemv_needs(k: &FastGemv) -> Needs {
    cuda_needs(k.kind.smem())
}

fn gemv_source(k: &FastGemv) -> Source {
    cuda_source_entry(k.kind.entry())
}

fn gemv_load(k: &FastGemv, _: usize) -> Option<Load> {
    let groups = k.kind.groups(k.n);
    let step = match k.kind {
        GemvKind::Bf16 => 256.0,
        GemvKind::Mxfp8 | GemvKind::Int4 => 1024.0,
    };
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: k.k as f64 / step,
        reads: vec![k.form.bytes(k.n * k.k) as f64, (4 * k.k) as f64],
        written: (4 * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn gemv_launches(k: &FastGemv, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.kind.groups(k.n),
        down: 1,
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(k.n),
        Arg::count(k.k),
    ];
    vec![Launch { args, shape }]
}

fn propose_gemv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
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
    if tree.elements(x) != k || n == 0 {
        return None;
    }
    let form = tree.node(w).form;
    let kind = match form.name() {
        "bf16" if k.is_multiple_of(256) => GemvKind::Bf16,
        "mxfp8" if k.is_multiple_of(1024) => GemvKind::Mxfp8,
        "int4_g32" if k.is_multiple_of(1024) && n.is_multiple_of(2) => GemvKind::Int4,
        _ => return None,
    };
    let kernel = FastGemv { n, k, kind, form };
    Some(vec![Draft::new(kernel, [x, w], n)])
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
    cuda_needs(if k.wide { (512 + WARPS * 128) * 4 } else { 0 })
}

fn stacked_source(k: &FastStacked) -> Source {
    cuda_source_entry(if k.wide {
        "tessera_stacked_wide"
    } else {
        "tessera_stacked_lanes"
    })
}

fn stacked_load(k: &FastStacked, _: usize) -> Option<Load> {
    let groups = if k.wide {
        (k.n / BLOCK) * k.batches * k.rows
    } else {
        k.len.div_ceil(WARPS)
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
            (2 * k.len * k.k) as f64,
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
            down: k.batches * k.rows,
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
    let shape = Shape::Grid {
        across: k.len.div_ceil(WARPS),
        down: 1,
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(k.len),
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

tessera_lower::rule!("trellis_mma", propose_trellis);
tessera_lower::rule!("fast_gemv", propose_gemv);
tessera_lower::rule!("fast_stacked", propose_stacked);
