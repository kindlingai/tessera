//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

mod cuda_fuse;
#[cfg(test)]
mod cuda_fuse_test;
#[cfg(test)]
mod device_test;
mod gemv;
pub(crate) mod mm;
mod moe;
mod stacked;
mod stacked_fused;
#[cfg(test)]
mod stacked_test;

use gemv::{propose_gemv, why_gemv};
use stacked::check_stacked;
pub(crate) use tessera_device::{CUDA_LACKS, cuda_rule, f32_only};
use tessera_graph::Order;
use tessera_linear::launch_args;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T, Tree};

const KERNEL: &str = include_str!("linear.cu");
const TRELLIS: &str = include_str!("trellis.cu");
const PREFILL: &str = include_str!("prefill.cu");
const INT4: &str = include_str!("int4.cu");
const BLOCK: usize = 128;
const WIDE_COLS: usize = 32;
const WIDE_THREADS: usize = 128;
const THREADS: usize = 256;
const WARPS: usize = 8;
const DTR_SMEM: usize = 4096 * 4;

macro_rules! cuda_module {
    ($vis:vis $src:ident, $with_vis:vis $with:ident, $name:literal, $text:expr) => {
        $vis fn $src() -> &'static str {
            static CUDA: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
            CUDA.get_or_init(|| {
                tessera_device::flatten_for(tessera_device::Backend::Cuda, $name, $text)
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
cuda_module!(pub(crate) trellis_source, pub(crate) trellis_source_with, "trellis.kernel", TRELLIS);
cuda_module!(pub(crate) prefill_source, pub(crate) prefill_source_with, "prefill.kernel", PREFILL);
cuda_module!(pub(crate) int4_source, pub(crate) int4_source_with, "int4.kernel", INT4);

#[cfg(test)]
pub(crate) fn hip_source() -> &'static str {
    static HIP: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    HIP.get_or_init(|| {
        tessera_device::flatten_for(tessera_device::Backend::Rocm, "linear.kernel", KERNEL)
    })
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

/// The knob rule for a dense trellis matmul.
pub(crate) fn dtr_rule(turned: bool) -> &'static str {
    if turned {
        "trellis_turned"
    } else {
        "trellis_mma"
    }
}

/// A trellis knob that lowers to a `u32` launch bound.
pub(crate) fn dtr_blocks(tree: &Tree, turned: bool, knob: &str) -> Option<u32> {
    tree.knobs.get(dtr_rule(turned), knob).map(|v| v as u32)
}

pub(crate) fn dtr_tile(m: usize) -> usize {
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

pub(crate) fn dtr_needs(bytes: usize) -> Needs {
    Needs {
        bf16_tile: true,
        ..cuda_needs(bytes)
    }
}

/// Defines that trim the prefill module for one entry.
pub(crate) fn prefill_defs(entry: &str, mx8: bool) -> Vec<(&'static str, usize)> {
    let mut defs = Vec::new();
    if entry.starts_with("tessera_mm_mxfp8_") {
        defs.push(("PRE_ONLY_MX8", 1));
    } else if entry.starts_with("tessera_mm_") {
        defs.push(("PRE_ONLY_MM", 1));
    } else {
        if !mx8 {
            defs.push(("DTR_FORCE_FP16", 1));
        }
        if !entry.starts_with("dtr_pre_gen") {
            defs.push(("PRE_ONLY_DTR_K4", 1));
        }
    }
    defs
}

fn trellis_mma_name(t: &TrellisMma) -> &'static str {
    t.form.suffixed(if t.turned {
        "trellis_turned"
    } else {
        "trellis_mma"
    })
}

fn mma_needs(_: &TrellisMma) -> Needs {
    dtr_needs(DTR_SMEM)
}

macro_rules! define_trellis_kernel {
    (
        $(#[$meta:meta])*
        struct $name:ident {
            name: $kname:ident,
            accepts: $accepts:ident,
            emits: $emits:ident,
            needs: $needs:ident,
            source: $source:ident,
            launches: $launches:ident,
            load: $load:ident,
            extra: { $($field:ident : $fty:ty,)* }
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Kernel)]
        #[launch(
            name = $kname(self),
            sizes = [self.m, self.n, self.k],
            work = self.m * self.n * self.k,
            accepts = $accepts,
            emits = $emits,
            needs = $needs(self),
            order = Order::PerConfig,
            source = $source(self),
            launches = $launches,
            load = $load,
            load_key = tessera_lower::load_key((
                (self.m, self.n, self.k, self.ks, self.tile, self.min_blocks),
                (self.bits, self.book, self.turned, self.form.name(), self.y_bf16),
                ($(self.$field,)*),
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
            pub(crate) form: &'static dyn Stored,
            $(pub(crate) $field: $fty,)*
        }

        fn $emits(k: &$name, form: &dyn Stored) -> bool {
            x_accepts(k.y_bf16, form)
        }
    };
}

define_trellis_kernel! {
    /// Dense trellis matmul with warp-per-tile MMA and split-K.
    struct TrellisMma {
        name: trellis_mma_name,
        accepts: mma_accepts,
        emits: mma_emits,
        needs: mma_needs,
        source: dtr_source,
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

fn elem(bf16: bool) -> usize {
    if bf16 { 2 } else { 4 }
}

fn mma_load(t: &TrellisMma, _: usize) -> Option<Load> {
    let scales: &[usize] = match (t.turned, t.ks) {
        (true, 1) => &[t.k, t.n],
        (true, _) => &[t.k],
        (false, _) => &[],
    };
    trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        t.tile,
        elem(t.x_bf16),
        elem(t.y_bf16),
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
    name = "trellis_reduce",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = reduce_accepts,
    emits = reduce_emits,
    needs = dtr_needs(0),
    order = Order::PerConfig,
    source = trellis_source_with(
        "dtr_reduce",
        io_defs(
            self.min_blocks.map(|v| ("DTR_RED_BLK", v as usize)),
            false,
            self.y_bf16,
            false,
        ),
    ),
    launches = reduce_launches,
    load = reduce_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.ks, self.min_blocks, self.turned, self.y_bf16,
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
}

fn reduce_accepts(_: &TrellisReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn reduce_emits(r: &TrellisReduce, form: &dyn Stored) -> bool {
    x_accepts(r.y_bf16, form)
}

pub(crate) fn reduce_load_for([m, n, ks]: [usize; 3], turned: bool, y_bf16: bool) -> Option<Load> {
    let mut reads = vec![(4 * ks * m * n) as f64];
    if turned {
        reads.push((4 * n) as f64);
    }
    Some(Load {
        threads: (n / BLOCK) * THREADS,
        group: THREADS,
        trips: ks as f64,
        reads,
        written: (elem(y_bf16) * m * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_load(r: &TrellisReduce, _: usize) -> Option<Load> {
    reduce_load_for([r.m, r.n, r.ks], r.turned, r.y_bf16)
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

/// Prefill tile for `m` by `n`, or the knob value.
pub(crate) fn dtr_pre_tile(tree: &Tree, turned: bool, m: usize, n: usize) -> usize {
    let rule = dtr_rule(turned);
    if let Some(tile) = tree.knobs.get(rule, "prefill_tile") {
        return if tile >= 128 { 128 } else { 64 };
    }
    let large_rows = tree.knobs.get(rule, "prefill_large_rows");
    let min_groups = tree.knobs.get(rule, "prefill_min_groups");
    if large_rows.is_some() || min_groups.is_some() {
        let large = m >= large_rows.unwrap_or(128)
            && m.div_ceil(128) * (n / BLOCK) >= min_groups.unwrap_or(48);
        return if large { 128 } else { 64 };
    }
    if (m > 64 && n >= 2048) || (m >= 128 && m.div_ceil(128) * (n / BLOCK) >= 48) {
        128
    } else {
        64
    }
}

fn dtr_pre_entry(t: &TrellisPreMma) -> &'static str {
    match (t.tile, t.bits, t.book, t.turned, t.ks > 1, t.rot_out) {
        (128, 4, 0, true, false, true) => "dtr_pre_k4_t16r",
        (64, 4, 0, true, false, true) => "dtr_pre_k4_t8r",
        (128, 4, 0, true, true, _) => "dtr_pre_k4_t16s",
        (128, 4, 0, true, false, false) => "dtr_pre_k4_t16",
        (64, 4, 0, true, true, _) => "dtr_pre_k4_t8s",
        (64, 4, 0, true, false, false) => "dtr_pre_k4_t8",
        (128, _, _, _, _, _) => "dtr_pre_gen16",
        (_, _, _, _, _, _) => "dtr_pre_gen8",
    }
}

fn dtr_pre_source(t: &TrellisPreMma) -> Source {
    let entry = dtr_pre_entry(t);
    let def = match (t.tile, entry.starts_with("dtr_pre_gen")) {
        (128, false) => "DTR_T16_BLK",
        (_, false) => "DTR_T8_BLK",
        (128, true) => "DTR_GEN16_BLK",
        (_, true) => "DTR_GEN8_BLK",
    };
    let base = prefill_defs(entry, t.mx8)
        .into_iter()
        .chain(t.min_blocks.map(|v| (def, v as usize)));
    prefill_source_with(entry, io_defs(base, false, t.y_bf16, false))
}

/// Rotates and quantizes dense inputs once per 64x128 block.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.mx8 { "trellis_prep" } else { "trellis_prep_f16" },
    sizes = [self.m, self.k],
    work = self.m * self.k,
    accepts = prep_accepts,
    emits = f32_only,
    needs = dtr_needs(DTR_SMEM),
    order = Order::PerConfig,
    source = prefill_source_with(
        "dtr_prep",
        x_defs(
            prefill_defs("dtr_prep", self.mx8)
                .into_iter()
                .chain(self.min_blocks.map(|v| ("DTR_PREP_BLK", v as usize))),
            self.x_bf16,
        ),
    ),
    launches = prep_launches,
    load = prep_load,
    load_key = tessera_lower::load_key((
        self.m, self.k, self.min_blocks, self.turned, self.mx8, self.x_bf16,
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
}

fn prep_accepts(p: &TrellisPrep, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => x_accepts(p.x_bf16, form),
        _ => form.name() == "f32",
    }
}

fn prep_load(p: &TrellisPrep, _: usize) -> Option<Load> {
    let groups = (p.k / BLOCK) * p.m.div_ceil(64);
    let mut reads = vec![(elem(p.x_bf16) * p.m * p.k) as f64];
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

fn trellis_pre_name(t: &TrellisPreMma) -> &'static str {
    let base = match (t.turned, t.mx8) {
        (true, true) => "trellis_turned_pf",
        (false, true) => "trellis_mma_pf",
        (true, false) => "trellis_turned_f16",
        (false, false) => "trellis_mma_f16",
    };
    t.form.suffixed(base)
}

fn pre_mma_needs(t: &TrellisPreMma) -> Needs {
    dtr_needs(if t.mx8 { 33_792 } else { 49_152 })
}

define_trellis_kernel! {
    /// Prefill dense trellis matmul from pre-rotated input blocks.
    struct TrellisPreMma {
        name: trellis_pre_name,
        accepts: pre_mma_accepts,
        emits: pre_mma_emits,
        needs: pre_mma_needs,
        source: dtr_pre_source,
        launches: pre_mma_launches,
        load: pre_mma_load,
        extra: {
            mx8: bool,
            mode: u32,
            rot_out: bool,
        }
    }
}

fn pre_mma_accepts(k: &TrellisPreMma, input: usize, form: &dyn Stored) -> bool {
    let other_at = 2 + usize::from(k.turned && k.ks == 1);
    if k.ks == 1 && input >= other_at {
        return gemv::epi_accepts(k.mode, input, other_at, form);
    }
    weight_x_accepts(input, false, k.form, form)
}

fn pre_mma_load(t: &TrellisPreMma, _: usize) -> Option<Load> {
    let scales: &[usize] = match (t.turned, t.ks) {
        (true, 1) => &[t.n],
        _ => &[],
    };
    let x_elem = if t.mx8 { 1 } else { 2 };
    let mut load = trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        t.tile,
        x_elem,
        elem(t.y_bf16),
        scales,
    )?;
    if t.ks == 1 {
        gemv::push_epi_reads(&mut load.reads, t.mode, [t.m, t.n], 4);
        if t.rot_out {
            load.reads.push((4 * t.n) as f64);
            load.written = (x_elem * t.m * t.n) as f64;
        }
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
    let other_at = 2 + usize::from(has_svh);
    let has_o = t.ks == 1 && gemv::has_other(t.mode);
    let other = if has_o {
        step.read(other_at)
    } else {
        step.read(0)
    };
    let bias_at = other_at + usize::from(has_o);
    let bias = if t.ks == 1 && (gemv::has_bias2(t.mode) || t.rot_out) {
        step.read(bias_at)
    } else {
        step.read(0)
    };
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

/// Rotated or plain dense trellis matmul.
fn eval_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    gemv::eval_trellis_for(tree, at)
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

pub(crate) fn x_defs<'a>(
    base: impl IntoIterator<Item = (&'a str, usize)>,
    x_bf16: bool,
) -> Vec<(&'a str, usize)> {
    io_defs(base, x_bf16, false, false)
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

cuda_rule!("trellis_mma", check = eval_trellis);
cuda_rule!("fast_stacked", check = check_stacked);
cuda_rule!("fast_gemv", propose_gemv, why = why_gemv);
