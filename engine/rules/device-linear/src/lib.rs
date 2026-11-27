//! Device-accelerated trellis MMA, streaming decode GEMV, and small-batch matmul rules.

mod cuda_fuse;
#[cfg(test)]
mod cuda_fuse_test;
#[cfg(test)]
mod device_test;
mod gemv;
pub(crate) mod mm;
mod rocm_mm;
#[cfg(test)]
mod rocm_test;
mod rocm_trellis;
mod stacked;
#[cfg(test)]
mod stacked_test;

pub(crate) use gemv::{prefill_source_entry, prefill_source_for};
use gemv::{propose_gemv, why_gemv};
use stacked::check_stacked;
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
pub(crate) use tessera_device::{CUDA_LACKS, ROCM_LACKS, cuda_rule, f32_only, rocm_rule};
use tessera_graph::Order;
use tessera_linear::launch_args;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T, Tree,
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

pub(crate) fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "linear.kernel", KERNEL))
}

pub(crate) fn prefill_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "prefill.kernel", PREFILL))
}

pub(crate) fn prefill_source_with<'a>(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'a str, usize)>,
) -> Source {
    tessera_device::cuda_source_with(prefill_source(), entry, defs)
}

pub(crate) fn int4_source() -> &'static str {
    static INT4_SRC: OnceLock<&'static str> = OnceLock::new();
    INT4_SRC.get_or_init(|| flatten_for(Backend::Cuda, "int4.kernel", INT4))
}

pub(crate) fn int4_source_with<'a>(
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

pub(crate) fn dtr_needs(bytes: usize) -> Needs {
    Needs {
        bf16_tile: true,
        ..cuda_needs(bytes)
    }
}

pub(crate) fn rocm_source(text: &'static str, entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", "", text]),
        entry,
    }
}

pub(crate) fn rocm_needs(group_bytes: usize, group_threads: usize, tile: bool) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads,
        compiler: true,
        tile,
        lacks: ROCM_LACKS,
        ..Needs::default()
    }
}

fn trellis_mma_name(t: &TrellisMma) -> &'static str {
    let base = match (t.rocm, t.turned) {
        (true, true) => "rocm_trellis_turned",
        (true, false) => "rocm_trellis_mma",
        (false, true) => "trellis_turned",
        (false, false) => "trellis_mma",
    };
    t.form.suffixed(base)
}

fn trellis_mma_tile(t: &TrellisMma) -> usize {
    if t.rocm {
        rocm_trellis::small_tile(t.m)
    } else {
        rocm_trellis::dtr_tile(t.m)
    }
}

fn mma_needs(t: &TrellisMma) -> Needs {
    if t.rocm {
        rocm_trellis::small_needs(t.m)
    } else {
        dtr_needs(DTR_SMEM)
    }
}

fn mma_source(t: &TrellisMma) -> Source {
    if t.rocm {
        rocm_source(
            rocm_trellis::rocm_trellis_source(),
            rocm_trellis::small_entry(t),
        )
    } else {
        cuda_source_entry(gemv::cuda_dtr_entry(t))
    }
}

macro_rules! define_trellis_kernel {
    (
        $(#[$meta:meta])*
        struct $name:ident {
            name: $kname:ident,
            accepts: $accepts:ident,
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
            emits = f32_only,
            needs = $needs(self),
            order = Order::PerConfig,
            source = $source(self),
            launches = $launches,
            load = $load,
            unproven,
        )]
        pub(crate) struct $name {
            pub(crate) m: usize,
            pub(crate) n: usize,
            pub(crate) k: usize,
            pub(crate) ks: usize,
            pub(crate) bits: u32,
            pub(crate) book: u32,
            pub(crate) turned: bool,
            pub(crate) rocm: bool,
            pub(crate) form: &'static dyn Stored,
            $(pub(crate) $field: $fty,)*
        }
    };
}

define_trellis_kernel! {
    /// Dense trellis matmul with warp-per-tile MMA and split-K.
    struct TrellisMma {
        name: trellis_mma_name,
        accepts: mma_accepts,
        needs: mma_needs,
        source: mma_source,
        launches: mma_launches,
        load: mma_load,
        extra: {}
    }
}

fn mma_accepts(k: &TrellisMma, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

pub(crate) fn trellis_mma_load(
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
    trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        trellis_mma_tile(t),
        4,
        scales,
    )
}

fn mma_launches(t: &TrellisMma, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(trellis_mma_tile(t)),
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
    emits = f32_only,
    needs = if self.rocm { rocm_needs(0, THREADS, false) } else { dtr_needs(0) },
    order = Order::PerConfig,
    source = if self.rocm {
        rocm_source(rocm_trellis::rocm_trellis_source(), "rocm_dtr_reduce")
    } else {
        cuda_source_entry("dtr_reduce")
    },
    launches = reduce_launches,
    load = reduce_load,
    unproven,
)]
pub(crate) struct TrellisReduce {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) ks: usize,
    pub(crate) turned: bool,
    pub(crate) rocm: bool,
}

fn reduce_accepts(_: &TrellisReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

pub(crate) fn reduce_load_for(m: usize, n: usize, ks: usize, turned: bool) -> Option<Load> {
    let mut reads = vec![(4 * ks * m * n) as f64];
    if turned {
        reads.push((4 * n) as f64);
    }
    Some(Load {
        threads: (n / BLOCK) * THREADS,
        group: THREADS,
        trips: ks as f64,
        reads,
        written: (4 * m * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_load(r: &TrellisReduce, _: usize) -> Option<Load> {
    reduce_load_for(r.m, r.n, r.ks, r.turned)
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

pub(crate) fn dtr_pre_tile(m: usize, n: usize) -> usize {
    if (m > 64 && n >= 2048) || (m >= 128 && m.div_ceil(128) * (n / BLOCK) >= 48) {
        128
    } else {
        64
    }
}

fn dtr_pre_entry(t: &TrellisPreMma) -> &'static str {
    match (
        dtr_pre_tile(t.m, t.n),
        t.bits,
        t.book,
        t.turned,
        t.ks > 1,
        t.rot_out,
    ) {
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
        prefill_source_for("dtr_prep", self.mx8)
    },
    launches = prep_launches,
    load = prep_load,
    shared,
    unproven,
)]
pub(crate) struct TrellisPrep {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) turned: bool,
    pub(crate) rocm: bool,
    pub(crate) mx8: bool,
}

fn prep_accepts(_: &TrellisPrep, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

pub(crate) fn prep_load_for(m: usize, k: usize, turned: bool) -> Option<Load> {
    let groups = (k / BLOCK) * m.div_ceil(64);
    let mut reads = vec![(4 * m * k) as f64];
    if turned {
        reads.push((4 * k) as f64);
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: 1.0,
        reads,
        written: (m * k) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn prep_load(p: &TrellisPrep, _: usize) -> Option<Load> {
    prep_load_for(p.m, p.k, p.turned)
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
    let base = match (t.rocm, t.turned, t.mx8) {
        (true, true, _) => "rocm_trellis_pf",
        (true, false, _) => "rocm_trellis_plain_pf",
        (false, true, true) => "trellis_turned_pf",
        (false, false, true) => "trellis_mma_pf",
        (false, true, false) => "trellis_turned_f16",
        (false, false, false) => "trellis_mma_f16",
    };
    t.form.suffixed(base)
}

fn trellis_pre_tile(t: &TrellisPreMma) -> usize {
    if t.rocm {
        rocm_trellis::pre_tile(t.m, t.n)
    } else {
        dtr_pre_tile(t.m, t.n)
    }
}

fn pre_mma_needs(t: &TrellisPreMma) -> Needs {
    if t.rocm {
        rocm_trellis::pre_needs(t.m, t.n)
    } else {
        dtr_needs(if t.mx8 { 33_792 } else { 49_152 })
    }
}

fn pre_mma_source(t: &TrellisPreMma) -> Source {
    if t.rocm {
        rocm_source(
            rocm_trellis::rocm_trellis_source(),
            rocm_trellis::pre_entry(t),
        )
    } else {
        prefill_source_for(dtr_pre_entry(t), t.mx8)
    }
}

define_trellis_kernel! {
    /// Prefill dense trellis matmul from pre-rotated input blocks.
    struct TrellisPreMma {
        name: trellis_pre_name,
        accepts: pre_mma_accepts,
        needs: pre_mma_needs,
        source: pre_mma_source,
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
    weight_accepts(input, k.form, form)
}

fn pre_mma_load(t: &TrellisPreMma, _: usize) -> Option<Load> {
    let scales: &[usize] = match (t.turned, t.ks) {
        (true, 1) => &[t.n],
        _ => &[],
    };
    let x_elem = if !t.rocm && t.mx8 { 1 } else { 2 };
    let mut load = trellis_mma_load(
        t.form,
        [t.m, t.n, t.k, t.ks],
        trellis_pre_tile(t),
        x_elem,
        scales,
    )?;
    if t.ks == 1 {
        gemv::push_epi_reads(&mut load.reads, t.mode, t.m, t.n);
        if t.rot_out {
            load.reads.push((4 * t.n) as f64);
            load.written = (x_elem * t.m * t.n) as f64;
        }
    }
    Some(load)
}

fn pre_mma_launches(t: &TrellisPreMma, step: &Step) -> Vec<Launch> {
    let tile = trellis_pre_tile(t);
    let shape = Shape::Grid {
        across: (t.n / BLOCK) * t.ks,
        down: t.m.div_ceil(tile),
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
    gemv::eval_trellis_for(tree, at, false)
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

pub(crate) fn cuda_source_entry(entry: &'static str) -> Source {
    cuda_source_with(entry, None::<(&str, usize)>)
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
