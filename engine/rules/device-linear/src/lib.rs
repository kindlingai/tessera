//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

#[cfg(test)]
mod device_test;
mod epi;
mod gemv;
pub(crate) mod mm;
mod moe;
mod rocm_mm;
#[cfg(test)]
mod rocm_test;
mod rocm_trellis;
mod stacked;
mod stacked_fused;
#[cfg(test)]
mod stacked_test;

use epi::{Epi, epi_accepts, epi_args, match_cuda_site, pick_divisor, push_epi_reads};
pub(crate) use epi::{cuda_needs, io_defs, trip, weight_io_accepts, weight_x_accepts, x_accepts};
use gemv::{propose_gemv, why_gemv};
pub(crate) use rocm_mm::{rocm_needs, rocm_source};
use stacked::check_stacked;
use tessera_device::{Backend, Options, flatten};
pub(crate) use tessera_device::{CUDA_LACKS, ROCM_LACKS, cuda_rule, f32_only, rocm_rule};
use tessera_graph::Order;
use tessera_linear::{Dense, launch_args};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Compute, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree,
};

const COMMON: &str = include_str!("common.cuh");
const KERNEL: &str = include_str!("linear.cu");
const BF16_MM: &str = include_str!("bf16_mm.cu");
const TRELLIS: &str = include_str!("trellis.cu");
const PREFILL: &str = include_str!("prefill.cu");
const INT4: &str = include_str!("int4.cu");
const BLOCK: usize = 128;
const WIDE_COLS: usize = 32;
const WIDE_THREADS: usize = 128;
const THREADS: usize = 256;
const WARPS: usize = 8;
const DTR_SMEM: usize = 4096 * 4;

pub(crate) fn flatten_local(backend: Backend, name: &str, text: &str) -> &'static str {
    let load = |path: &str| (path == "common.cuh").then(|| COMMON.into());
    let flat = flatten(name, text, &load, &Options::plain(backend));
    Box::leak(
        flat.expect("built-in kernel flattens")
            .text
            .into_boxed_str(),
    )
}

macro_rules! cuda_module {
    ($vis:vis $src:ident, $with_vis:vis $with:ident, $name:literal, $text:expr) => {
        $vis fn $src() -> &'static str {
            static CUDA: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
            CUDA.get_or_init(|| {
                $crate::flatten_local(tessera_device::Backend::Cuda, $name, $text)
            })
        }

        $with_vis fn $with<'a>(
            entry: &'static str,
            defs: impl IntoIterator<Item = (&'a str, usize)>,
        ) -> tessera_lower::Source {
            tessera_device::cuda_source_with($src(), entry, defs)
        }
    };
}
pub(crate) use cuda_module;

cuda_module!(pub(crate) cuda_source, pub(crate) cuda_source_with, "linear.kernel", KERNEL);
cuda_module!(pub(crate) bf16_mm_source, pub(crate) bf16_mm_source_with, "bf16_mm.kernel", BF16_MM);
cuda_module!(pub(crate) trellis_source, pub(crate) trellis_source_with, "trellis.kernel", TRELLIS);
cuda_module!(pub(crate) prefill_source, pub(crate) prefill_source_with, "prefill.kernel", PREFILL);
cuda_module!(pub(crate) int4_source, pub(crate) int4_source_with, "int4.kernel", INT4);

#[cfg(test)]
pub(crate) fn hip_source() -> &'static str {
    static HIP: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    HIP.get_or_init(|| flatten_local(Backend::Rocm, "linear.kernel", KERNEL))
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
tessera_lower::knob!("trellis_pre_sk", "splits", 1, 16);

fn dtr_ks(m: usize, n: usize, k: usize, tree: &Tree, rule: &str) -> usize {
    let ibs = k / BLOCK;
    if let Some(want) = tree.knobs.get(rule, "splits") {
        let cap = ibs.clamp(1, 16);
        return pick_divisor(ibs, want.clamp(1, cap), cap);
    }
    let split_max_rows = tree.knobs.get(rule, "split_max_rows").unwrap_or(32);
    if m > split_max_rows || ibs <= 1 {
        return 1;
    }
    let small = tree.knobs.get(rule, "target_small").unwrap_or(288);
    let target = if m <= 2 {
        small
    } else if m <= 16 {
        tree.knobs.get(rule, "target_large").unwrap_or(128)
    } else {
        tree.knobs.get(rule, "target_large").unwrap_or(96)
    };
    let cap = if m <= 16 { 16 } else { 8 }.min(ibs);
    let want = target.div_ceil(n / BLOCK).clamp(1, cap);
    pick_divisor(ibs, want, cap)
}

fn dtr_pre_tile(m: usize, n: usize) -> usize {
    if (m > 64 && n >= 2048) || (m >= 128 && m.div_ceil(128) * (n / BLOCK) >= 48) {
        128
    } else {
        64
    }
}

fn cuda_pre_ks(tree: &Tree, rule: &str, m: usize, n: usize, k: usize, tile: usize) -> usize {
    let ibs = k / BLOCK;
    if let Some(want) = tree
        .knobs
        .get("trellis_pre_sk", "splits")
        .or_else(|| tree.knobs.get(rule, "splits"))
    {
        let cap = want.min((ibs / 2).max(1));
        return pick_divisor(ibs, want.min(cap), cap);
    }
    if ibs < 8 {
        return 1;
    }
    let base = (n / BLOCK) * m.div_ceil(tile);
    if base >= 96 {
        return 1;
    }
    let max_k = if base <= 16 {
        8
    } else if tile == 128 {
        4
    } else {
        6
    };
    let cap = max_k.min(ibs / 2);
    pick_divisor(ibs, 96usize.div_ceil(base).clamp(1, cap), cap)
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
        (32, 4, 0, true, true) => "dtr_k4_t4s",
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
        "dtr_k4_t4s" | "dtr_k4_t4" => "DTR_T4_BLK",
        _ => "DTR_GEN_BLK",
    };
    trellis_source_with(
        entry,
        io_defs(
            t.min_blocks.map(|v| (def, v as usize)),
            t.x_bf16,
            t.y_bf16,
            false,
        ),
    )
}

fn dtr_needs(bytes: usize) -> Needs {
    Needs {
        bf16_tile: true,
        ..cuda_needs(bytes)
    }
}

fn prefill_dtr_defs(
    mx8: bool,
    only_k4: bool,
    blk: Option<(&'static str, u32)>,
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
) -> Vec<(&'static str, usize)> {
    let mut defs = Vec::new();
    if !mx8 {
        defs.push(("DTR_FORCE_FP16", 1));
    }
    if only_k4 {
        defs.push(("PRE_ONLY_DTR_K4", 1));
    }
    if let Some((name, val)) = blk {
        defs.push((name, val as usize));
    }
    io_defs(defs, x_bf16, y_bf16, o_bf16)
}

fn trellis_name(form: &dyn Stored, rocm: bool, turned: bool, pre: Option<bool>) -> &'static str {
    let base = match (rocm, turned, pre) {
        (true, true, None) => "rocm_trellis_turned",
        (true, false, None) => "rocm_trellis_mma",
        (false, true, None) => "trellis_turned",
        (false, false, None) => "trellis_mma",
        (true, true, Some(_)) => "rocm_trellis_pf",
        (true, false, Some(_)) => "rocm_trellis_plain_pf",
        (false, true, Some(true)) => "trellis_turned_pf",
        (false, false, Some(true)) => "trellis_mma_pf",
        (false, true, Some(false)) => "trellis_turned_f16",
        (false, false, Some(false)) => "trellis_mma_f16",
    };
    form.suffixed(base)
}

macro_rules! dtr_mma_kernel {
    (
        $(#[$meta:meta])*
        struct $name:ident {
            name: $kname:expr,
            accepts: $accepts:ident,
            emits: $emits:ident,
            rocm_needs: $rneeds:expr,
            cuda_smem: $csmem:expr,
            rocm_entry: $rentry:path,
            cuda_source: $csource:ident,
            launches: $launches:ident,
            load: $load:ident,
            extra: { $($field:ident : $fty:ty,)* }
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Kernel)]
        #[launch(
            name = ($kname)(self),
            sizes = [self.m, self.n, self.k],
            work = self.m * self.n * self.k,
            accepts = $accepts,
            emits = $emits,
            needs = if self.rocm { ($rneeds)(self) } else { dtr_needs(($csmem)(self)) },
            order = Order::PerConfig,
            source = if self.rocm {
                rocm_source(rocm_trellis::rocm_trellis_source(), $rentry(self))
            } else {
                $csource(self)
            },
            launches = $launches,
            load = $load,
            load_key = tessera_lower::load_key((
                (self.m, self.n, self.k, self.ks, self.tile, self.min_blocks),
                (self.bits, self.book, self.turned, self.y_bf16, self.rocm),
                ($(self.$field,)*),
                self.form.name(),
            )),
            unproven,
        )]
        pub(crate) struct $name {
            pub(crate) m: usize,
            pub(crate) n: usize,
            pub(crate) k: usize,
            pub(crate) ks: usize,
            pub(crate) tile: usize,
            pub(crate) min_blocks: Option<u32>,
            pub(crate) bits: u32,
            pub(crate) book: u32,
            pub(crate) turned: bool,
            pub(crate) y_bf16: bool,
            pub(crate) rocm: bool,
            $(pub(crate) $field: $fty,)*
            pub(crate) form: &'static dyn Stored,
        }

        fn $emits(k: &$name, form: &dyn Stored) -> bool {
            x_accepts(k.y_bf16, form)
        }
    };
}

dtr_mma_kernel! {
    /// Dense trellis matmul with warp-per-tile MMA and split-K.
    struct TrellisMma {
        name: |t: &TrellisMma| trellis_name(t.form, t.rocm, t.turned, None),
        accepts: mma_accepts,
        emits: mma_emits,
        rocm_needs: |t: &TrellisMma| rocm_trellis::small_needs(t.tile),
        cuda_smem: |_: &TrellisMma| DTR_SMEM,
        rocm_entry: rocm_trellis::small_entry,
        cuda_source: dtr_source,
        launches: mma_launches,
        load: mma_load,
        extra: {
            x_bf16: bool,
        }
    }
}

fn mma_accepts(k: &TrellisMma, input: usize, form: &dyn Stored) -> bool {
    weight_x_accepts(input, k.x_bf16, k.form, form)
}

fn trellis_mma_load(
    form: &dyn Stored,
    dims: [usize; 4],
    tile: usize,
    x_elem: usize,
    y_elem: usize,
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
        written: (y_elem * ks * m * n) as f64,
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
    let x_elem = if t.x_bf16 { 2 } else { 4 };
    let y_elem = if t.y_bf16 { 2 } else { 4 };
    trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        t.tile,
        x_elem,
        y_elem,
        scales,
    )
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
    name = if self.rocm { "rocm_trellis_reduce" } else { "trellis_reduce" },
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = reduce_accepts,
    emits = reduce_emits,
    needs = if self.rocm { rocm_needs(0, THREADS, false) } else { dtr_needs(0) },
    order = Order::PerConfig,
    source = if self.rocm {
        rocm_source(rocm_trellis::rocm_trellis_source(), "rocm_dtr_reduce")
    } else {
        trellis_source_with(
            "dtr_reduce",
            io_defs(
                self.min_blocks.map(|v| ("DTR_RED_BLK", v as usize)),
                false,
                self.y_bf16,
                false,
            ),
        )
    },
    launches = reduce_launches,
    load = reduce_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.ks, self.min_blocks, self.turned, self.y_bf16, self.rocm,
    )),
    unproven,
)]
pub(crate) struct TrellisReduce {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) ks: usize,
    pub(crate) min_blocks: Option<u32>,
    pub(crate) turned: bool,
    pub(crate) y_bf16: bool,
    pub(crate) rocm: bool,
}

fn reduce_accepts(_: &TrellisReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn reduce_emits(r: &TrellisReduce, form: &dyn Stored) -> bool {
    x_accepts(r.y_bf16, form)
}

pub(crate) fn reduce_load_for(
    m: usize,
    n: usize,
    ks: usize,
    turned: bool,
    y_bf16: bool,
) -> Option<Load> {
    let mut reads = vec![(4 * ks * m * n) as f64];
    if turned {
        reads.push((4 * n) as f64);
    }
    let y_elem = if y_bf16 { 2 } else { 4 };
    Some(Load {
        threads: (n / BLOCK) * THREADS,
        group: THREADS,
        trips: ks as f64,
        reads,
        written: (y_elem * m * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_load(r: &TrellisReduce, _: usize) -> Option<Load> {
    reduce_load_for(r.m, r.n, r.ks, r.turned, r.y_bf16)
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

/// Split-K trellis reduction with output rotation and epilogue.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "trellis_reduce_epi",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = cred_accepts,
    emits = cred_emits,
    needs = dtr_needs(0),
    order = Order::PerConfig,
    source = prefill_source_with(
        "dtr_reduce_epi",
        prefill_dtr_defs(
            self.mx8,
            true,
            self.min_blocks.map(|v| ("DTR_RED_BLK", v)),
            false,
            self.y_bf16,
            self.o_bf16,
        ),
    ),
    launches = cred_launches,
    load = cred_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.ks, self.turned, self.mode, self.mx8, self.min_blocks,
        self.y_bf16, self.o_bf16,
    )),
    unproven,
)]
struct CudaTrellisRed {
    m: usize,
    n: usize,
    ks: usize,
    turned: bool,
    mode: u32,
    mx8: bool,
    min_blocks: Option<u32>,
    y_bf16: bool,
    o_bf16: bool,
}

fn cred_accepts(r: &CudaTrellisRed, input: usize, form: &dyn Stored) -> bool {
    let other_at = 1 + usize::from(r.turned);
    if input >= other_at {
        return epi_accepts(r.mode, input, other_at, r.o_bf16, form);
    }
    form.name() == "f32"
}

fn cred_emits(r: &CudaTrellisRed, form: &dyn Stored) -> bool {
    x_accepts(r.y_bf16, form)
}

fn cred_load(r: &CudaTrellisRed, _: usize) -> Option<Load> {
    let mut load = reduce_load_for(r.m, r.n, r.ks, r.turned, r.y_bf16)?;
    push_epi_reads(&mut load.reads, r.mode, r.o_bf16, r.m, r.n);
    Some(load)
}

fn cred_launches(r: &CudaTrellisRed, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: r.n / BLOCK,
        down: r.m.div_ceil(8),
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let (other, bias) = epi_args(step, 1 + usize::from(r.turned), r.mode, true);
    let args = vec![
        step.read(0),
        svh,
        other,
        bias,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::count(usize::from(r.turned)),
        Arg::value(&r.mode),
    ];
    vec![Launch { args, shape }]
}

/// Rotates and quantizes dense inputs once per 64x128 block.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.rocm {
        "rocm_trellis_prep"
    } else if self.mx8 {
        "trellis_prep"
    } else {
        "trellis_prep_f16"
    },
    sizes = [self.m, self.k],
    work = self.m * self.k,
    accepts = prep_accepts,
    emits = f32_only,
    needs = if self.rocm {
        rocm_needs(16 * 1024, THREADS, true)
    } else {
        dtr_needs(DTR_SMEM)
    },
    order = Order::PerConfig,
    source = if self.rocm {
        rocm_source(rocm_trellis::rocm_trellis_source(), "rocm_dtr_prep")
    } else {
        prefill_source_with(
            "dtr_prep",
            prefill_dtr_defs(
                self.mx8,
                true,
                self.min_blocks.map(|v| ("DTR_PREP_BLK", v)),
                self.x_bf16,
                false,
                false,
            ),
        )
    },
    launches = prep_launches,
    load = prep_load,
    load_key = tessera_lower::load_key((
        self.m, self.k, self.min_blocks, self.turned, self.mx8, self.x_bf16, self.rocm,
    )),
    shared,
    unproven,
)]
pub(crate) struct TrellisPrep {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) min_blocks: Option<u32>,
    pub(crate) turned: bool,
    pub(crate) mx8: bool,
    pub(crate) x_bf16: bool,
    pub(crate) rocm: bool,
}

fn prep_accepts(p: &TrellisPrep, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => x_accepts(p.x_bf16, form),
        _ => form.name() == "f32",
    }
}

fn prep_load(p: &TrellisPrep, _: usize) -> Option<Load> {
    let groups = (p.k / BLOCK) * p.m.div_ceil(64);
    let x_elem = if p.x_bf16 { 2 } else { 4 };
    let mut reads = vec![(x_elem * p.m * p.k) as f64];
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

fn dtr_pre_entry(t: &TrellisPreMma) -> &'static str {
    match (t.tile, t.bits, t.book, t.turned, t.ks > 1) {
        (128, 4, 0, true, true) => "dtr_pre_k4_t16s",
        (128, 4, 0, true, false) => "dtr_pre_k4_t16",
        (64, 4, 0, true, true) => "dtr_pre_k4_t8s",
        (64, 4, 0, true, false) => "dtr_pre_k4_t8",
        (128, _, _, _, _) => "dtr_pre_gen16",
        (_, _, _, _, _) => "dtr_pre_gen8",
    }
}

fn dtr_pre_source(t: &TrellisPreMma) -> Source {
    let entry = dtr_pre_entry(t);
    let def = match entry {
        "dtr_pre_k4_t16" | "dtr_pre_k4_t16s" => "DTR_T16_BLK",
        "dtr_pre_k4_t8" | "dtr_pre_k4_t8s" => "DTR_T8_BLK",
        "dtr_pre_gen16" => "DTR_GEN16_BLK",
        _ => "DTR_GEN8_BLK",
    };
    let only_k4 = !entry.starts_with("dtr_pre_gen");
    prefill_source_with(
        entry,
        prefill_dtr_defs(
            t.mx8,
            only_k4,
            t.min_blocks.map(|v| (def, v)),
            false,
            t.y_bf16,
            t.o_bf16,
        ),
    )
}

dtr_mma_kernel! {
    /// Prefill dense trellis matmul from pre-rotated input blocks.
    struct TrellisPreMma {
        name: |t: &TrellisPreMma| trellis_name(t.form, t.rocm, t.turned, Some(t.mx8)),
        accepts: pre_mma_accepts,
        emits: pre_mma_emits,
        rocm_needs: |_: &TrellisPreMma| rocm_trellis::pre_needs(),
        cuda_smem: |t: &TrellisPreMma| if t.mx8 { 33_792 } else { 49_152 },
        rocm_entry: rocm_trellis::pre_entry,
        cuda_source: dtr_pre_source,
        launches: pre_mma_launches,
        load: pre_mma_load,
        extra: {
            mx8: bool,
            mode: u32,
            o_bf16: bool,
        }
    }
}

fn pre_mma_accepts(k: &TrellisPreMma, input: usize, form: &dyn Stored) -> bool {
    let other_at = 2 + usize::from(k.turned && k.ks == 1);
    if k.ks == 1 && input >= other_at {
        return epi_accepts(k.mode, input, other_at, k.o_bf16, form);
    }
    weight_x_accepts(input, false, k.form, form)
}

fn pre_mma_load(t: &TrellisPreMma, _: usize) -> Option<Load> {
    let scales: &[usize] = match (t.turned, t.ks) {
        (true, 1) => &[t.n],
        _ => &[],
    };
    let x_elem = if t.mx8 { 1 } else { 2 };
    let y_elem = if t.y_bf16 { 2 } else { 4 };
    let mut load = trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        t.tile,
        x_elem,
        y_elem,
        scales,
    )?;
    if t.ks == 1 {
        push_epi_reads(&mut load.reads, t.mode, t.o_bf16, t.m, t.n);
    }
    Some(load)
}

fn pre_mma_launches(t: &TrellisPreMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(t.tile),
        threads: THREADS,
    };
    let has_svh = t.turned && t.ks == 1;
    let svh = if has_svh { step.read(2) } else { step.read(0) };
    let (other, bias) = epi_args(step, 2 + usize::from(has_svh), t.mode, t.ks == 1);
    let args = vec![
        step.read(0),
        step.read(1),
        svh,
        other,
        bias,
        step.out(),
        Arg::count(t.m),
        Arg::count(t.n),
        Arg::count(t.k),
        Arg::value(&t.bits),
        Arg::value(&t.book),
        Arg::count(usize::from(t.turned)),
        Arg::value(&t.mode),
        Arg::count(t.ks),
    ];
    vec![Launch { args, shape }]
}

fn prefill_drafts(
    tree: &Tree,
    at: T,
    d: &Dense,
    rule: &str,
    epi: Epi,
) -> Result<Vec<Draft>, String> {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let custom_large = tree.knobs.get(rule, "prefill_large_rows");
    let custom_groups = tree.knobs.get(rule, "prefill_min_groups");
    let pre_tile = match tree.knobs.get(rule, "prefill_tile") {
        Some(64) => 64,
        Some(128) => 128,
        Some(v) => return Err(format!("prefill_tile={v} (requires 64 or 128)")),
        None if custom_large.is_none() && custom_groups.is_none() => dtr_pre_tile(m, n),
        None => {
            let large = custom_large.unwrap_or(128);
            let groups = custom_groups.unwrap_or(48);
            if m >= large && m.div_ceil(128) * (n / BLOCK) >= groups {
                128
            } else {
                64
            }
        }
    };
    let ks = if d.bits == 4 && d.book == 0 && turned {
        cuda_pre_ks(tree, rule, m, n, k, pre_tile)
    } else {
        1
    };
    let mx8 = tree.compute_at(at) == Compute::Bf16;
    let pre_blk = match (pre_tile, d.bits, d.book, turned) {
        (128, 4, 0, true) => tree
            .knobs
            .get(rule, "t16_min_blocks")
            .or_else(|| tree.knobs.get(rule, "min_blocks")),
        (64, 4, 0, true) => tree
            .knobs
            .get(rule, "t8_min_blocks")
            .or_else(|| tree.knobs.get(rule, "min_blocks")),
        _ => tree.knobs.get(rule, "min_blocks"),
    }
    .map(|v| v as u32);
    let [suh, svh] = d.turn.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
    let p_reads = [d.x].into_iter().chain(suh).map(Read::Node);
    let prep = TrellisPrep {
        m,
        k,
        min_blocks: tree.knobs.get(rule, "prep_min_blocks").map(|v| v as u32),
        turned,
        mx8,
        x_bf16: d.x_bf16,
        rocm: false,
    };
    let pre_len = m.div_ceil(64) * (k / BLOCK) * 8 * 512;
    let prep_draft = Draft::new(prep, p_reads, pre_len);
    let mma = TrellisPreMma {
        m,
        n,
        k,
        ks,
        tile: pre_tile,
        min_blocks: pre_blk,
        bits: d.bits,
        book: d.book,
        turned,
        y_bf16: ks == 1 && epi.y_bf16,
        rocm: false,
        mx8,
        mode: if ks > 1 { 0 } else { epi.mode },
        o_bf16: ks == 1 && epi.o_bf16,
        form: d.form,
    };
    if ks > 1 {
        let red = CudaTrellisRed {
            m,
            n,
            ks,
            turned,
            mode: epi.mode,
            mx8,
            min_blocks: tree.knobs.get(rule, "reduce_min_blocks").map(|v| v as u32),
            y_bf16: epi.y_bf16,
            o_bf16: epi.o_bf16,
        };
        let r_reads = [Read::Draft(1)]
            .into_iter()
            .chain(svh.map(Read::Node))
            .chain(epi.other.map(Read::Node))
            .chain(epi.bias2.map(Read::Node));
        return Ok(vec![
            prep_draft,
            Draft::new(mma, [Read::Draft(0), Read::Node(d.w)], ks * m * n),
            Draft::new(red, r_reads, m * n),
        ]);
    }
    let m_reads = [Read::Draft(0), Read::Node(d.w)]
        .into_iter()
        .chain(svh.map(Read::Node))
        .chain(epi.other.map(Read::Node))
        .chain(epi.bias2.map(Read::Node));
    Ok(vec![prep_draft, Draft::new(mma, m_reads, m * n)])
}

fn trellis_drafts(tree: &Tree, at: T, d: &Dense, epi: Epi) -> Option<Result<Vec<Draft>, String>> {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let rule = if turned {
        "trellis_turned"
    } else {
        "trellis_mma"
    };
    let pre_min_rows = tree.knobs.get(rule, "prefill_min_rows").unwrap_or(33);
    if m >= pre_min_rows {
        return Some(prefill_drafts(tree, at, d, rule, epi));
    }
    if epi.mode != 0 {
        return None;
    }
    let ks = dtr_ks(m, n, k, tree, rule);
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
        y_bf16: ks == 1 && d.y_bf16,
        rocm: false,
        x_bf16: d.x_bf16,
        form: d.form,
    };
    let red = TrellisReduce {
        m,
        n,
        ks,
        min_blocks: tree.knobs.get(rule, "reduce_min_blocks").map(|v| v as u32),
        turned,
        y_bf16: d.y_bf16,
        rocm: false,
    };
    Some(Ok(d.drafts(mma, ks, red)))
}

fn dense_pick(tree: &Tree, at: T) -> Option<Dense> {
    Dense::find_with(tree, at, BLOCK, true)?.ok()
}

/// Rotated or plain dense trellis matmul.
fn eval_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let Some((d, epi)) = match_cuda_site(tree, at, dense_pick) else {
        return Dense::find_with(tree, at, BLOCK, true).and_then(|r| r.err().map(Err));
    };
    trellis_drafts(tree, at, &d, epi)
}

cuda_rule!("trellis_mma", check = eval_trellis);
cuda_rule!("fast_stacked", check = check_stacked);
cuda_rule!("fast_gemv", propose_gemv, why = why_gemv);
