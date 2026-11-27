//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

#[cfg(test)]
mod device_test;
mod gemv;
pub(crate) mod mm;
#[cfg(test)]
mod stacked_test;

use gemv::{propose_gemv, why_gemv};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
pub(crate) use tessera_device::{CUDA_LACKS, cuda_rule, f32_only};
use tessera_graph::{DType, DimExpr, Op, Order, Pairs, Stage, View};
use tessera_linear::{Dense, launch_args};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Ops, Read, Shape, Source, Step, T, Term,
    Tree,
};

const KERNEL: &str = include_str!("linear.cu");
const PREFILL: &str = include_str!("prefill.cu");
const STACKED: &str = include_str!("stacked.cu");
const BLOCK: usize = 128;
const WIDE_COLS: usize = 32;
const WIDE_THREADS: usize = 128;
const THREADS: usize = 256;
const WARPS: usize = 8;
const DTR_SMEM: usize = 4096 * 4;
const DTR_PRE_SMEM: usize = 40_960;

pub(crate) fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "linear.kernel", KERNEL))
}

pub(crate) fn prefill_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "prefill.kernel", PREFILL))
}

fn prefill_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", prefill_source(), ""]),
        entry,
    }
}

pub(crate) fn stacked_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "stacked.kernel", STACKED))
}

fn stacked_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", stacked_source(), ""]),
        entry,
    }
}

#[cfg(test)]
pub(crate) fn hip_source() -> &'static str {
    static HIP: OnceLock<&'static str> = OnceLock::new();
    HIP.get_or_init(|| flatten_for(Backend::Rocm, "linear.kernel", KERNEL))
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

fn trellis_mma_load(
    form: &dyn Stored,
    dims: [usize; 4],
    tile: usize,
    x_elem: usize,
    scales: &[usize],
) -> Option<Load> {
    let [m, n, k, ks] = dims;
    let groups = (n / BLOCK) * ks * m.div_ceil(tile);
    let mut reads = vec![form.bytes(n * k) as f64, (x_elem * m * k) as f64];
    reads.extend(scales.iter().map(|&len| (4 * len) as f64));
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (k / BLOCK / ks) as f64,
        reads,
        written: (4 * ks * m * n) as f64,
        flight: 1024.0,
        ..Load::default()
    })
}

fn mma_load(t: &TrellisMma, _: usize) -> Option<Load> {
    let scales: &[usize] = match (t.turned, t.ks) {
        (true, 1) => &[t.k, t.n],
        (true, _) => &[t.k],
        (false, _) => &[],
    };
    trellis_mma_load(t.form, [t.m, t.n, t.k, t.ks], dtr_tile(t.m), 4, scales)
}

fn mma_launches(t: &TrellisMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(dtr_tile(t.m)),
        threads: THREADS,
    };
    let sizes = [t.m, t.n, t.k, t.ks];
    let args = launch_args(step, sizes, t.bits, t.book, t.turned);
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

fn dtr_pre_tile(m: usize, n: usize) -> usize {
    if m >= 128 && m.div_ceil(128) * (n / BLOCK) >= 96 {
        128
    } else {
        64
    }
}

fn dtr_pre_entry(t: &TrellisPreMma) -> &'static str {
    match (dtr_pre_tile(t.m, t.n), t.bits, t.book, t.turned) {
        (128, 4, 0, true) => "dtr_pre_k4_t16",
        (64, 4, 0, true) => "dtr_pre_k4_t8",
        (128, _, _, _) => "dtr_pre_gen16",
        (_, _, _, _) => "dtr_pre_gen8",
    }
}

/// Rotates and quantizes dense inputs once per 64x128 block.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "trellis_prep",
    sizes = [self.m, self.k],
    work = self.m * self.k,
    accepts = prep_accepts,
    emits = f32_only,
    needs = dtr_needs(DTR_SMEM),
    order = Order::PerConfig,
    source = prefill_source_entry("dtr_prep"),
    launches = prep_launches,
    load = prep_load,
    shared,
    unproven,
)]
struct TrellisPrep {
    m: usize,
    k: usize,
    turned: bool,
}

fn prep_accepts(_: &TrellisPrep, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn prep_load(p: &TrellisPrep, _: usize) -> Option<Load> {
    let groups = (p.k / BLOCK) * p.m.div_ceil(64);
    let mut reads = vec![(4 * p.m * p.k) as f64];
    if p.turned {
        reads.push((4 * p.k) as f64);
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: 1.0,
        reads,
        written: (p.m * p.k) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn prep_launches(p: &TrellisPrep, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: p.k / BLOCK,
        down: p.m.div_ceil(64),
        threads: THREADS,
    };
    let suh = if p.turned { step.read(1) } else { step.read(0) };
    let args = vec![
        step.read(0),
        suh,
        step.out(),
        Arg::count(p.m),
        Arg::count(p.k),
        Arg::count(usize::from(p.turned)),
    ];
    vec![Launch { args, shape }]
}

/// Prefill dense trellis matmul from pre-rotated input blocks.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "trellis_turned_pf" } else { "trellis_mma_pf" }),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = pre_mma_accepts,
    emits = f32_only,
    needs = dtr_needs(DTR_PRE_SMEM),
    order = Order::PerConfig,
    source = prefill_source_entry(dtr_pre_entry(self)),
    launches = pre_mma_launches,
    load = pre_mma_load,
    unproven,
)]
struct TrellisPreMma {
    m: usize,
    n: usize,
    k: usize,
    bits: u32,
    book: u32,
    turned: bool,
    form: &'static dyn Stored,
}

fn pre_mma_accepts(k: &TrellisPreMma, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn pre_mma_load(t: &TrellisPreMma, _: usize) -> Option<Load> {
    let scales: &[usize] = if t.turned { &[t.n] } else { &[] };
    trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, 1],
        dtr_pre_tile(t.m, t.n),
        1,
        scales,
    )
}

fn pre_mma_launches(t: &TrellisPreMma, step: &Step) -> Vec<Launch> {
    let tile = dtr_pre_tile(t.m, t.n);
    let shape = Shape::Grid {
        across: t.n / BLOCK,
        down: t.m.div_ceil(tile),
        threads: THREADS,
    };
    let svh = if t.turned { step.read(2) } else { step.read(0) };
    let args = vec![
        step.read(0),
        step.read(1),
        svh,
        step.out(),
        Arg::count(t.m),
        Arg::count(t.n),
        Arg::count(t.k),
        Arg::value(&t.bits),
        Arg::value(&t.book),
        Arg::count(usize::from(t.turned)),
    ];
    vec![Launch { args, shape }]
}

fn trellis_drafts(d: &Dense) -> Vec<Draft> {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let ks = dtr_ks(m, n, k);
    if ks == 1 && m >= 64 {
        let [suh, svh] = d.turn.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
        let p_reads = [d.x].into_iter().chain(suh).map(Read::Node);
        let m_reads = [Read::Draft(0), Read::Node(d.w)]
            .into_iter()
            .chain(svh.map(Read::Node));
        let prep = TrellisPrep { m, k, turned };
        let pre_len = m.div_ceil(64) * (k / BLOCK) * 8 * 512;
        let mma = TrellisPreMma {
            m,
            n,
            k,
            bits: d.bits,
            book: d.book,
            turned,
            form: d.form,
        };
        return vec![
            Draft::new(prep, p_reads, pre_len),
            Draft::new(mma, m_reads, m * n),
        ];
    }
    let mma = TrellisMma {
        m,
        n,
        k,
        ks,
        bits: d.bits,
        book: d.book,
        turned,
        form: d.form,
    };
    d.drafts(mma, ks, TrellisReduce { m, n, ks, turned })
}

/// Rotated or plain dense trellis matmul.
fn eval_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    Some(Dense::find(tree, at, BLOCK)?.map(|d| trellis_drafts(&d)))
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

pub(crate) fn cuda_source_entry(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", cuda_source(), ""]),
        entry,
    }
}

pub(crate) fn weight_accepts(input: usize, expected: &dyn Stored, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == expected.name(),
        _ => form.name() == "f32",
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeTail {
    width: usize,
    rot_stride: usize,
    rot_off: usize,
    turned: usize,
    adjacent: bool,
    rope_first: bool,
    theta: f32,
}

/// Fast CUDA stacked BF16 matmul, transposed or row-major.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.wide { "fast_stacked_wide" } else { "fast_stacked_lanes" },
    sizes = [
        self.batches, self.rows, self.n, self.k,
        self.x_stride[0], self.x_stride[1], self.o_stride[0], self.o_stride[1],
    ],
    work = self.len * self.k,
    accepts = stacked_accepts,
    emits = f32_only,
    needs = stacked_needs(self),
    order = Order::PerConfig,
    source = stacked_source_for(self),
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
    x_stride: [usize; 2],
    o_stride: [usize; 2],
    wide: bool,
    rope: Option<RopeTail>,
    form: &'static dyn Stored,
}

impl FastStacked {
    fn wide_64(&self) -> bool {
        self.wide && self.rows >= 32 && self.n.is_multiple_of(64) && self.k.is_multiple_of(32)
    }

    fn lanes_64(&self) -> bool {
        !self.wide && self.rows >= 32 && self.n.is_multiple_of(64) && self.k.is_multiple_of(32)
    }

    fn tiled_64(&self) -> bool {
        self.wide_64() || self.lanes_64()
    }

    fn lanes_tc(&self) -> bool {
        !self.wide && self.n.is_multiple_of(16) && self.k.is_multiple_of(128)
    }
}

fn stacked_accepts(k: &FastStacked, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        3 => form.name() == "u32",
        _ => form.name() == "f32",
    }
}

fn stacked_needs(k: &FastStacked) -> Needs {
    if k.tiled_64() {
        return Needs {
            bf16_tile: true,
            ..cuda_needs(64 * 18 * 4 * 2 + 64 * 40 * 2)
        };
    }
    if !(k.wide || k.lanes_tc()) {
        return cuda_needs(0);
    }
    let group_bytes = if k.wide {
        16 * 18 * 8 + 32 * 34 * 2
    } else {
        4 * 24 * 36 * 4
    };
    Needs {
        group_threads: WIDE_THREADS,
        bf16_tile: true,
        ..cuda_needs(group_bytes)
    }
}

fn stacked_source_for(k: &FastStacked) -> Source {
    match (k.wide, k.tiled_64(), k.lanes_tc()) {
        (true, true, _) => stacked_source_entry("tessera_stacked_wide_64"),
        (false, true, _) => stacked_source_entry("tessera_stacked_lanes_64"),
        (true, false, _) => stacked_source_entry("tessera_stacked_wide"),
        (false, false, true) => cuda_source_entry("tessera_stacked_lanes_tc"),
        (false, false, false) => cuda_source_entry("tessera_stacked_lanes"),
    }
}

fn stacked_load(k: &FastStacked, _: usize) -> Option<Load> {
    let (tiled, tc) = (k.tiled_64(), k.lanes_tc());
    let rtiles = k.rows.div_ceil(match (tiled, k.wide || tc) {
        (true, _) => 64,
        (false, true) => 16,
        (false, false) => 8,
    });
    let r = k.rows.min(8);
    let (groups, group, trips, mut trip_ops, flight) = if tiled {
        (
            (k.n / 64) * k.batches * rtiles,
            THREADS,
            k.k as f64 / 32.0,
            Ops::default(),
            1024.0,
        )
    } else if tc {
        (
            (k.n / 16) * k.batches * rtiles,
            128,
            k.k as f64 / 128.0,
            Ops::default(),
            1024.0,
        )
    } else if k.wide {
        (
            (k.n / WIDE_COLS) * k.batches * rtiles,
            WIDE_THREADS,
            k.k as f64 / 32.0,
            Ops::default(),
            512.0,
        )
    } else {
        (
            (k.batches * k.n).div_ceil(WARPS) * rtiles,
            THREADS,
            k.k as f64 / 256.0,
            trip(8 * r, 8, 1 + 2 * r, 16 + 32 * r),
            512.0,
        )
    };
    trip_ops.fresh = f64::from(u8::from(tiled || tc));
    let mut reads = vec![
        (2 * k.batches * rtiles * k.n * k.k) as f64,
        (4 * k.batches * k.rows * k.k) as f64,
    ];
    if let Some(r) = k.rope {
        reads.push((4 * k.batches * k.rows * r.width) as f64);
        reads.push((4 * k.rows) as f64);
    }
    Some(Load {
        threads: groups * group,
        group,
        trips,
        trip: trip_ops,
        reads,
        written: (4 * k.len) as f64,
        flight,
        ..Load::default()
    })
}

fn stacked_launches(k: &FastStacked, step: &Step) -> Vec<Launch> {
    let ([xb, xr], [ob, or]) = (k.x_stride, k.o_stride);
    let dims = [k.rows, k.n, k.k, k.block, k.first, k.step, xb, xr, ob, or];
    if k.wide {
        let (across, down, threads) = if k.wide_64() {
            (k.n / 64, k.batches * k.rows.div_ceil(64), THREADS)
        } else {
            (
                k.n / WIDE_COLS,
                k.batches * k.rows.div_ceil(16),
                WIDE_THREADS,
            )
        };
        let (rot, pos, mat_off, rope_out_off, r) = match k.rope {
            Some(r) => {
                let (m_off, r_off) = if r.rope_first { (r.width, 0) } else { (0, k.n) };
                (step.read(2), step.read(3), m_off, r_off, r)
            }
            None => (
                step.read(0),
                step.read(0),
                0,
                0,
                RopeTail {
                    width: 0,
                    rot_stride: 0,
                    rot_off: 0,
                    turned: 0,
                    adjacent: false,
                    rope_first: false,
                    theta: 1.0,
                },
            ),
        };
        let mut args = vec![
            step.read(0),
            step.read(1),
            rot,
            pos,
            step.out(),
            Arg::count(k.batches),
        ];
        args.extend(dims.map(Arg::count));
        args.extend([
            Arg::count(mat_off),
            Arg::count(r.width),
            Arg::count(r.rot_stride),
            Arg::count(r.rot_off),
            Arg::count(rope_out_off),
            Arg::count(r.turned),
            Arg::count(usize::from(r.adjacent)),
            Arg::value(&r.theta),
        ]);
        let shape = Shape::Grid {
            across,
            down,
            threads,
        };
        return vec![Launch { args, shape }];
    }
    let tc = k.lanes_tc() || k.lanes_64();
    let bn = k.batches * k.n;
    let (across, down, threads) = if k.lanes_64() {
        (k.n / 64, k.batches * k.rows.div_ceil(64), THREADS)
    } else if k.lanes_tc() {
        (k.n / 16, k.batches * k.rows.div_ceil(16), WIDE_THREADS)
    } else {
        (bn.div_ceil(WARPS), k.rows.div_ceil(8), THREADS)
    };
    let shape = Shape::Grid {
        across,
        down,
        threads,
    };
    let mut args = vec![step.read(0), step.read(1), step.out()];
    args.extend((!tc).then(|| Arg::count(bn)));
    args.extend(dims.map(Arg::count));
    vec![Launch { args, shape }]
}

fn peel_input(tree: &Tree, x: T, batches: usize, rows: usize, k: usize) -> (T, [usize; 2]) {
    let is_swap = |t: T| matches!(&tree.node(t).term, Term::Op(Op::View(View::Permute(p))) if p[..] == [1, 0, 2]);
    if !is_swap(x) || tree.root(x) || tree.node(tree.node(x).args[0]).form.name() != "f32" {
        return (x, [rows * k, k]);
    }
    let inner = tree.node(x).args[0];
    let slice = matches!(
        &tree.node(inner).term,
        Term::Op(Op::View(View::Slice {
            axis: 2,
            start: DimExpr::Const(0),
            ..
        }))
    );
    if slice && !tree.root(inner) {
        let base = tree.node(inner).args[0];
        if let &[r, b, fk] = tree.node(base).shape.as_slice()
            && (r, b) == (rows, batches)
            && fk.is_multiple_of(4)
            && tree.node(base).form.name() == "f32"
        {
            return (base, [fk, batches * fk]);
        }
    }
    (inner, [k, batches * k])
}

/// Peels an optional single-reader last-axis slice from `t`.
fn peel_rot_slice(tree: &Tree, t: T) -> Option<(T, usize, usize)> {
    if tree.root(t) {
        return None;
    }
    let n = tree.node(t);
    let Term::Op(Op::View(View::Slice {
        axis: 2,
        start: off,
        ..
    })) = &n.term
    else {
        return None;
    };
    let parent = n.args[0];
    if tree.node(parent).form.name() != "f32" {
        return None;
    }
    let off = off.value(&tree.g.dims, &|d| tree.size(d))? as usize;
    let stride = *tree.node(parent).shape.last()?;
    Some((parent, off, stride))
}

/// Matches `concat(permute(matmul), rope)` or `concat(rope, permute(matmul))`.
fn match_rope_concat(tree: &Tree, at: T) -> Option<(T, RopeTail, T, T)> {
    let outer = tree.node(at);
    if outer.term != Term::Op(Op::View(View::Concat)) || outer.form.name() != "f32" {
        return None;
    }
    let [a, b] = outer.args[..] else {
        return None;
    };
    let is_rope = |t: T| matches!(tree.node(t).term, Term::Stage(Stage::Rope { .. }));
    let (plain_t, rope_t, rope_first) = match (is_rope(a), is_rope(b)) {
        (false, true) => (a, b, false),
        (true, false) => (b, a, true),
        _ => return None,
    };
    if tree.root(plain_t) || tree.root(rope_t) {
        return None;
    }
    let perm = tree.node(plain_t);
    let Term::Op(Op::View(View::Permute(order))) = &perm.term else {
        return None;
    };
    if order[..] != [1, 0, 2] || perm.form.name() != "f32" {
        return None;
    }
    let mat = perm.args[0];
    if tree.root(mat) || tree.node(mat).term != Term::Op(Op::Matmul) {
        return None;
    }
    let rope_node = tree.node(rope_t);
    let Term::Stage(Stage::Rope {
        theta,
        pairs,
        turned,
    }) = rope_node.term
    else {
        return None;
    };
    let [rot_in, pos_t] = rope_node.args[..] else {
        return None;
    };
    if rope_node.form.name() != "f32"
        || tree.node(rot_in).form.name() != "f32"
        || tree.node(pos_t).form.dtype() != DType::U32
    {
        return None;
    }
    let &[r_rows, r_heads, rope_w] = rope_node.shape.as_slice() else {
        return None;
    };
    if rope_w == 0
        || !rope_w.is_multiple_of(2)
        || tree.elements(pos_t) != r_rows
        || perm.shape[..2] != [r_rows, r_heads]
    {
        return None;
    }
    let (rot_src, rot_off, rot_stride) =
        peel_rot_slice(tree, rot_in).unwrap_or((rot_in, 0, rope_w));
    let tail = RopeTail {
        width: rope_w,
        rot_stride,
        rot_off,
        turned: turned.map_or(rope_w / 2, |t| t as usize),
        adjacent: pairs == Pairs::Adjacent,
        rope_first,
        theta,
    };
    Some((mat, tail, rot_src, pos_t))
}

fn check_stacked(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let outer = tree.node(at);
    let (node, out_swap, rope_match) =
        if let Some((mat, tail, rot, pos)) = match_rope_concat(tree, at) {
            (tree.node(mat), true, Some((tail, rot, pos)))
        } else {
            match &outer.term {
                Term::Op(Op::View(View::Permute(p))) if p[..] == [1, 0, 2] => {
                    let inner = outer.args[0];
                    if tree.root(inner) || tree.node(inner).term != Term::Op(Op::Matmul) {
                        return None;
                    }
                    (tree.node(inner), true, None)
                }
                Term::Op(Op::Matmul) => (outer, false, None),
                _ => return None,
            }
        };
    let (x, mut w) = (node.args[0], node.args[1]);
    let (&[batches, n, k], &[xb, rows, xk]) =
        (tree.node(w).shape.as_slice(), tree.node(x).shape.as_slice())
    else {
        return None;
    };
    if xb != batches || xk != k || rows == 0 {
        return None;
    }
    if [outer.form, node.form, tree.node(x).form]
        .iter()
        .any(|f| f.name() != "f32")
    {
        return Some(Err("stacked requires f32 input and output".into()));
    }
    let wide =
        matches!(&tree.node(w).term, Term::Op(Op::View(View::Permute(p))) if p[..] == [0, 2, 1]);
    if rope_match.is_some() && !wide {
        return None;
    }
    let (x, x_stride) = peel_input(tree, x, batches, rows, k);
    let out_w = n + rope_match.map_or(0, |(r, _, _)| r.width);
    let o_stride = if out_swap {
        [out_w, batches * out_w]
    } else {
        [rows * out_w, out_w]
    };
    w = if wide { tree.node(w).args[0] } else { w };
    let mut first_row = 0usize;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = &tree.node(w).term {
        let Some(offset) = start.value(&tree.g.dims, &|d| tree.size(d)) else {
            return Some(Err("slice start is dynamic".into()));
        };
        (first_row, w) = (offset as usize, tree.node(w).args[0]);
    }
    if !matches!(&tree.node(w).term, Term::Op(Op::View(View::Reshape(_)))) {
        return Some(Err("weight is not a reshaped 2D leaf".into()));
    }
    let leaf = tree.node(w).args[0];
    let (&[_, block_rows, width], &[_, stored_w]) = (
        tree.node(w).shape.as_slice(),
        tree.node(leaf).shape.as_slice(),
    ) else {
        return Some(Err("reshaped weight is not 3D over 2D leaf".into()));
    };
    let form = tree.node(leaf).form;
    if !tree.leaf(leaf) || stored_w != width || form.name() != "bf16" {
        return Some(Err(
            "stacked requires direct bf16 leaf with matching width".into()
        ));
    }
    let (block, first) = (block_rows * width, first_row * width);
    if ![block, first, width].iter().all(|v| v.is_multiple_of(8)) {
        return Some(Err(format!(
            "strides ({width}, {block}, {first}) not multiple of 8"
        )));
    }
    let ok = if wide {
        n > 0 && n.is_multiple_of(WIDE_COLS) && k > 0 && k.is_multiple_of(32)
    } else {
        n > 0 && k > 0 && k.is_multiple_of(8)
    };
    if !ok {
        return Some(Err(format!(
            "stacked shape n={n} k={k} wide={wide} unsupported"
        )));
    }
    let (rope, reads) = match rope_match {
        Some((tail, rot, pos)) => (Some(tail), vec![x, leaf, rot, pos]),
        None => (None, vec![x, leaf]),
    };
    let len = batches * rows * out_w;
    let k = FastStacked {
        len,
        batches,
        rows,
        n,
        k,
        block,
        first,
        step: width,
        x_stride,
        o_stride,
        wide,
        rope,
        form,
    };
    Some(Ok(vec![Draft::new(k, reads, len)]))
}

cuda_rule!("trellis_mma", check = eval_trellis);
cuda_rule!("fast_stacked", check = check_stacked);
cuda_rule!("fast_gemv", propose_gemv, why = why_gemv);
