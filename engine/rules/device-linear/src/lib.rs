//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

#[cfg(test)]
mod device_test;
mod gemv;
pub(crate) mod mm;
mod stacked;
#[cfg(test)]
mod stacked_test;

use gemv::{propose_gemv, why_gemv};
use stacked::check_stacked;
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
pub(crate) use tessera_device::{CUDA_LACKS, cuda_rule, f32_only};
use tessera_graph::Order;
use tessera_linear::{Dense, launch_args};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Draft, Kernel, Launch, Load, Needs, Ops, Read, Shape, Source, Step, T, Tree,
};

const KERNEL: &str = include_str!("linear.cu");
const PREFILL: &str = include_str!("prefill.cu");
const INT4: &str = include_str!("int4.cu");
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

fn prefill_source_with<'a>(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'a str, usize)>,
) -> Source {
    tessera_device::cuda_source_with(prefill_source(), entry, defs)
}

pub(crate) fn int4_source() -> &'static str {
    static INT4_SRC: OnceLock<&'static str> = OnceLock::new();
    INT4_SRC.get_or_init(|| flatten_for(Backend::Cuda, "int4.kernel", INT4))
}

fn int4_source_with<'a>(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'a str, usize)>,
) -> Source {
    tessera_device::cuda_source_with(int4_source(), entry, defs)
}

#[cfg(test)]
pub(crate) fn hip_source() -> &'static str {
    static HIP: OnceLock<&'static str> = OnceLock::new();
    HIP.get_or_init(|| flatten_for(Backend::Rocm, "linear.kernel", KERNEL))
}

macro_rules! dtr_knobs {
    ($k:literal) => {
        tessera_lower::knob!($k, "splits", 1, 16);
        tessera_lower::knob!($k, "split_max_rows", 1, 64);
        tessera_lower::knob!($k, "target_small", 1, 1024);
        tessera_lower::knob!($k, "target_large", 1, 1024);
        tessera_lower::knob!($k, "tile_rows", 8, 32);
        tessera_lower::knob!($k, "prefill_min_rows", 1, 1024);
        tessera_lower::knob!($k, "prefill_tile", 64, 128);
        tessera_lower::knob!($k, "prefill_large_rows", 64, 2048);
        tessera_lower::knob!($k, "prefill_min_groups", 1, 1024);
        tessera_lower::knob!($k, "min_blocks", 1, 16);
        tessera_lower::knob!($k, "t8_min_blocks", 1, 16);
        tessera_lower::knob!($k, "t16_min_blocks", 1, 16);
        tessera_lower::knob!($k, "prep_min_blocks", 1, 16);
        tessera_lower::knob!($k, "reduce_min_blocks", 1, 16);
    };
}

dtr_knobs!("trellis_mma");
dtr_knobs!("trellis_turned");

fn dtr_ks(m: usize, n: usize, k: usize, tree: &Tree, rule: &str) -> usize {
    let ibs = k / BLOCK;
    if let Some(want) = tree.knobs.get(rule, "splits") {
        let cap = ibs.clamp(1, 16);
        let w = want.clamp(1, cap);
        for d in w..=cap {
            if ibs.is_multiple_of(d) {
                return d;
            }
        }
        for d in (1..w).rev() {
            if ibs.is_multiple_of(d) {
                return d;
            }
        }
        return 1;
    }
    let split_max_rows = tree.knobs.get(rule, "split_max_rows").unwrap_or(16);
    if m > split_max_rows || ibs <= 1 {
        return 1;
    }
    let small = tree.knobs.get(rule, "target_small").unwrap_or(288);
    let large = tree.knobs.get(rule, "target_large").unwrap_or(128);
    let target = if m <= 2 { small } else { large };
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
    match (t.tile, t.bits, t.book, t.turned, t.ks > 1) {
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

fn dtr_source(t: &TrellisMma) -> Source {
    let entry = dtr_entry(t);
    let def = match entry {
        "dtr_k4_t1s" | "dtr_k4_t1" | "dtr_k4_p1s" | "dtr_k4_p1" => "DTR_T1_BLK",
        "dtr_k4_t2s" | "dtr_k4_t2" => "DTR_T2_BLK",
        "dtr_k4_t4" => "DTR_T4_BLK",
        _ => "DTR_GEN_BLK",
    };
    cuda_source_with(entry, t.min_blocks.map(|v| (def, v as usize)))
}

fn dtr_needs(bytes: usize) -> Needs {
    Needs {
        bf16_tile: true,
        ..cuda_needs(bytes)
    }
}

macro_rules! dtr_mma_kernel {
    (
        $(#[$meta:meta])*
        struct $name:ident,
        accepts = $accepts:ident,
        smem = $smem:expr,
        source = $source:ident,
        launches = $launches:ident,
        load = $load:ident,
        suffix = $suffix:literal
        $(, extra: $extra:ident : $extra_ty:ty)?
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Kernel)]
        #[launch(
            name = self.form.suffixed(if self.turned {
                concat!("trellis_turned", $suffix)
            } else {
                concat!("trellis_mma", $suffix)
            }),
            sizes = [self.m, self.n, self.k],
            work = self.m * self.n * self.k,
            accepts = $accepts, emits = f32_only,
            needs = dtr_needs($smem), order = Order::PerConfig,
            source = $source(self), launches = $launches, load = $load,
            load_key = tessera_lower::load_key((
                self.m, self.n, self.k, $(self.$extra,)? self.tile, self.min_blocks,
                self.bits, self.book, self.turned, self.form.name(),
            )),
            unproven,
        )]
        struct $name {
            m: usize,
            n: usize,
            k: usize,
            $($extra: $extra_ty,)?
            tile: usize,
            min_blocks: Option<u32>,
            bits: u32,
            book: u32,
            turned: bool,
            form: &'static dyn Stored,
        }

        fn $accepts(k: &$name, input: usize, form: &dyn Stored) -> bool {
            weight_accepts(input, k.form, form)
        }
    };
}

dtr_mma_kernel! {
    /// Dense trellis matmul with warp-per-tile MMA and split-K.
    struct TrellisMma,
    accepts = mma_accepts,
    smem = DTR_SMEM,
    source = dtr_source,
    launches = mma_launches,
    load = mma_load,
    suffix = "",
    extra: ks: usize
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
    trellis_mma_load(t.form, [t.m, t.n, t.k, t.ks], t.tile, 4, scales)
}

fn mma_launches(t: &TrellisMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(t.tile),
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
    source = cuda_source_with(
        "dtr_reduce",
        self.min_blocks.map(|v| ("DTR_RED_BLK", v as usize)),
    ),
    launches = reduce_launches,
    load = reduce_load,
    load_key = tessera_lower::load_key((self.m, self.n, self.ks, self.min_blocks, self.turned)),
    unproven,
)]
struct TrellisReduce {
    m: usize,
    n: usize,
    ks: usize,
    min_blocks: Option<u32>,
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

fn dtr_pre_entry(t: &TrellisPreMma) -> &'static str {
    match (t.tile, t.bits, t.book, t.turned) {
        (128, 4, 0, true) => "dtr_pre_k4_t16",
        (64, 4, 0, true) => "dtr_pre_k4_t8",
        (128, _, _, _) => "dtr_pre_gen16",
        (_, _, _, _) => "dtr_pre_gen8",
    }
}

fn dtr_pre_source(t: &TrellisPreMma) -> Source {
    let entry = dtr_pre_entry(t);
    let def = match entry {
        "dtr_pre_k4_t16" => "DTR_T16_BLK",
        "dtr_pre_k4_t8" => "DTR_T8_BLK",
        "dtr_pre_gen16" => "DTR_GEN16_BLK",
        _ => "DTR_GEN8_BLK",
    };
    prefill_source_with(entry, t.min_blocks.map(|v| (def, v as usize)))
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
    source = prefill_source_with(
        "dtr_prep",
        self.min_blocks.map(|v| ("DTR_PREP_BLK", v as usize)),
    ),
    launches = prep_launches,
    load = prep_load,
    load_key = tessera_lower::load_key((self.m, self.k, self.min_blocks, self.turned)),
    shared,
    unproven,
)]
struct TrellisPrep {
    m: usize,
    k: usize,
    min_blocks: Option<u32>,
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

dtr_mma_kernel! {
    /// Prefill dense trellis matmul from pre-rotated input blocks.
    struct TrellisPreMma,
    accepts = pre_mma_accepts,
    smem = DTR_PRE_SMEM,
    source = dtr_pre_source,
    launches = pre_mma_launches,
    load = pre_mma_load,
    suffix = "_pf"
}

fn pre_mma_load(t: &TrellisPreMma, _: usize) -> Option<Load> {
    let scales: &[usize] = if t.turned { &[t.n] } else { &[] };
    trellis_mma_load(t.form, [t.m, t.n, t.k, 1], t.tile, 1, scales)
}

fn pre_mma_launches(t: &TrellisPreMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: t.n / BLOCK,
        down: t.m.div_ceil(t.tile),
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

fn trellis_drafts(tree: &Tree, d: &Dense) -> Option<Result<Vec<Draft>, String>> {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let rule = if turned {
        "trellis_turned"
    } else {
        "trellis_mma"
    };
    let ks = dtr_ks(m, n, k, tree, rule);
    let pre_min_rows = tree.knobs.get(rule, "prefill_min_rows").unwrap_or(64);
    if ks == 1 && m >= pre_min_rows {
        let large_rows = tree.knobs.get(rule, "prefill_large_rows").unwrap_or(128);
        let min_groups = tree.knobs.get(rule, "prefill_min_groups").unwrap_or(96);
        let pre_tile = match tree.knobs.get(rule, "prefill_tile") {
            Some(64) => 64,
            Some(128) => 128,
            Some(v) => return Some(Err(format!("prefill_tile={v} (requires 64 or 128)"))),
            None => {
                if m >= large_rows && m.div_ceil(128) * (n / BLOCK) >= min_groups {
                    128
                } else {
                    64
                }
            }
        };
        let t8_blk = tree.knobs.get(rule, "t8_min_blocks");
        let t16_blk = tree.knobs.get(rule, "t16_min_blocks");
        let pre_blk = match (pre_tile, d.bits, d.book, turned) {
            (128, 4, 0, true) => t16_blk.or_else(|| tree.knobs.get(rule, "min_blocks")),
            (64, 4, 0, true) => t8_blk.or_else(|| tree.knobs.get(rule, "min_blocks")),
            _ => tree.knobs.get(rule, "min_blocks"),
        }
        .map(|v| v as u32);
        let [suh, svh] = d.turn.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
        let p_reads = [d.x].into_iter().chain(suh).map(Read::Node);
        let m_reads = [Read::Draft(0), Read::Node(d.w)]
            .into_iter()
            .chain(svh.map(Read::Node));
        let prep = TrellisPrep {
            m,
            k,
            min_blocks: tree.knobs.get(rule, "prep_min_blocks").map(|v| v as u32),
            turned,
        };
        let pre_len = m.div_ceil(64) * (k / BLOCK) * 8 * 512;
        let mma = TrellisPreMma {
            m,
            n,
            k,
            tile: pre_tile,
            min_blocks: pre_blk,
            bits: d.bits,
            book: d.book,
            turned,
            form: d.form,
        };
        return Some(Ok(vec![
            Draft::new(prep, p_reads, pre_len),
            Draft::new(mma, m_reads, m * n),
        ]));
    }
    let tile = match tree.knobs.get(rule, "tile_rows") {
        Some(8) => 8,
        Some(16) => 16,
        Some(32) => 32,
        Some(v) => return Some(Err(format!("tile_rows={v} (requires 8, 16, or 32)"))),
        None => dtr_tile(m),
    };
    let mma = TrellisMma {
        m,
        n,
        k,
        ks,
        tile,
        min_blocks: tree.knobs.get(rule, "min_blocks").map(|v| v as u32),
        bits: d.bits,
        book: d.book,
        turned,
        form: d.form,
    };
    let red = TrellisReduce {
        m,
        n,
        ks,
        min_blocks: tree.knobs.get(rule, "reduce_min_blocks").map(|v| v as u32),
        turned,
    };
    Some(Ok(d.drafts(mma, ks, red)))
}

/// Rotated or plain dense trellis matmul.
fn eval_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    match Dense::find(tree, at, BLOCK)? {
        Ok(d) => trellis_drafts(tree, &d),
        Err(e) => Some(Err(e)),
    }
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

pub(crate) fn cuda_source_with<'a>(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'a str, usize)>,
) -> Source {
    tessera_device::cuda_source_with(cuda_source(), entry, defs)
}

pub(crate) fn weight_accepts(input: usize, expected: &dyn Stored, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == expected.name(),
        _ => form.name() == "f32",
    }
}

cuda_rule!("trellis_mma", check = eval_trellis);
cuda_rule!("fast_stacked", check = check_stacked);
cuda_rule!("fast_gemv", propose_gemv, why = why_gemv);
