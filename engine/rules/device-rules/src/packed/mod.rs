//! Native packed varlen FlashAttention-2 rules for CUDA and ROCm.

use std::sync::OnceLock;
use tessera_device::{
    Backend, CUDA_LACKS, ROCM_LACKS, cuda_rule, f32_only, flatten_for, rocm_rule,
};
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites;
use tessera_lower::{Arg, Attn, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree};

const PACKED_KERNEL: &str = include_str!("packed_gpu.cu");
const FA2_BQ: usize = 64;
const FA2_BK: usize = 64;
const FA2_THREADS: usize = 128;
const FA2_MAX_DIM: usize = 128;

fn packed_hpad(width: usize) -> usize {
    if width <= 16 {
        16
    } else if width <= 64 {
        64
    } else if width <= 80 {
        80
    } else {
        128
    }
}

fn packed_entry(width: usize) -> &'static str {
    match packed_hpad(width) {
        16 => "fa2_packed_d16",
        64 => "fa2_packed_d64",
        80 => "fa2_packed_d80",
        _ => "fa2_packed_d128",
    }
}

pub fn packed_cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "packed.kernel", PACKED_KERNEL))
}

pub fn packed_rocm_source() -> &'static str {
    static ROCM: OnceLock<&'static str> = OnceLock::new();
    ROCM.get_or_init(|| flatten_for(Backend::Rocm, "packed.kernel", PACKED_KERNEL))
}

fn packed_source(rocm: bool, entry: &'static str) -> Source {
    if rocm {
        Source::rocm(packed_rocm_source(), entry)
    } else {
        Source::cuda(packed_cuda_source(), entry)
    }
}

fn packed_bq(width: usize) -> usize {
    match packed_hpad(width) {
        64 | 80 => 256,
        128 => 128,
        _ => FA2_BQ,
    }
}

fn packed_bk(rocm: bool, width: usize) -> usize {
    if rocm && packed_hpad(width) >= 80 {
        32
    } else {
        FA2_BK
    }
}

fn packed_threads(rocm: bool, width: usize) -> usize {
    if rocm {
        match packed_hpad(width) {
            64 | 80 => 512,
            128 => 256,
            _ => FA2_THREADS,
        }
    } else if packed_bq(width) > FA2_BQ {
        256
    } else {
        FA2_THREADS
    }
}

fn packed_smem(rocm: bool, width: usize) -> usize {
    let hpad = packed_hpad(width);
    let bq = packed_bq(width);
    let warps = packed_threads(rocm, width) / 32;
    let bounds = (bq * 2 + warps * 2) * 4;
    if rocm {
        let bk = packed_bk(true, width);
        let bufs = if hpad >= 80 { 2 } else { 1 };
        let kv = bufs * (bk * (hpad + 8) + hpad * bk) * 2;
        kv.max(bounds)
    } else {
        2 * 64 * (hpad + 8) * 2 + bounds
    }
}

/// Native packed varlen FlashAttention-2 kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.rocm { "flash_packed_rocm" } else { "flash_packed_cuda" },
    sizes = self.attn.sizes(),
    work = self.attn.heads * self.attn.queries * self.attn.seen() * self.attn.width,
    accepts = packed_accepts,
    emits = f32_only,
    needs = packed_needs(self),
    order = Order::PerConfig,
    source = packed_source(self.rocm, packed_entry(self.attn.width)),
    launches = packed_launches,
    load = packed_load,
    unproven,
)]
struct PackedAttend {
    attn: Attn,
    scale: f32,
    rocm: bool,
}

fn packed_accepts(_: &PackedAttend, input: usize, form: &dyn Stored) -> bool {
    if input == 3 {
        form.name() == "u32"
    } else {
        form.name() == "f32"
    }
}

fn packed_needs(k: &PackedAttend) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: packed_smem(k.rocm, k.attn.width),
        group_threads: packed_threads(k.rocm, k.attn.width),
        compiler: true,
        lacks: if k.rocm { ROCM_LACKS } else { CUDA_LACKS },
        tile: k.rocm,
        bf16_tile: !k.rocm,
        ..Needs::default()
    }
}

fn packed_load(k: &PackedAttend, len: usize) -> Option<Load> {
    let a = &k.attn;
    let bq = packed_bq(a.width);
    let threads = packed_threads(k.rocm, a.width);
    let blocks = a.queries.div_ceil(bq);
    let groups = blocks * a.heads;
    let seen = a.seen().max(1);
    let q_bytes = (4 * len) as f64;
    let kv_bytes = (4 * seen * a.kv_heads * a.width * blocks) as f64;
    let mut load = Load {
        threads: groups * threads,
        group: threads,
        trips: seen.div_ceil(packed_bk(k.rocm, a.width)) as f64,
        reads: vec![q_bytes, kv_bytes, kv_bytes],
        written: (4 * len) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn packed_launches(k: &PackedAttend, step: &Step) -> Vec<Launch> {
    let a = &k.attn;
    let bq = packed_bq(a.width);
    let shape = Shape::Grid {
        across: a.heads,
        down: a.queries.div_ceil(bq),
        threads: packed_threads(k.rocm, a.width),
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.out(),
        Arg::value(&k.scale),
        Arg::count(a.queries),
        Arg::count(a.heads),
        Arg::count(a.kv_heads),
        Arg::count(a.width),
        Arg::count(usize::from(a.mask.causal())),
    ];
    vec![Launch { args, shape }]
}

fn check_packed(tree: &Tree, at: T, rocm: bool) -> Option<Result<Vec<Draft>, String>> {
    let site: sites::Attend = tree.site(at)?;
    let bounds = site.packed?;
    let a = site.sizes;
    if site.q_form.name() != "f32"
        || a.k_form.name() != "f32"
        || a.v_form.name() != "f32"
        || a.out_form.name() != "f32"
    {
        return Some(Err("packed fa2 requires f32 io".into()));
    }
    if a.width == 0 || a.width > FA2_MAX_DIM || !a.width.is_multiple_of(8) {
        return Some(Err(format!(
            "head width {} (requires 8..={FA2_MAX_DIM}, multiple of 8)",
            a.width
        )));
    }
    if a.queries == 0 || a.kv_heads == 0 || !a.heads.is_multiple_of(a.kv_heads) {
        return Some(Err("packed fa2 requires positive rows and valid gqa".into()));
    }
    let kernel = PackedAttend {
        attn: a,
        scale: site.scale,
        rocm,
    };
    Some(Ok(vec![Draft::new(
        kernel,
        [site.q, site.k, site.v, bounds],
        site.out(),
    )]))
}

fn check_cuda(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    check_packed(tree, at, false)
}

fn check_rocm(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    check_packed(tree, at, true)
}

cuda_rule!("flash_packed_cuda", check = check_cuda);
rocm_rule!("flash_packed_rocm", check = check_rocm);
