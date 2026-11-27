//! Streaming BF16 and MXFP8 Tensor-Core small-batch matmul rule.

pub(super) use super::gemv::pick_divisor;
use super::gemv::{
    MatmulBase, check_mm_base, epi_accepts, has_bias2, has_other, is_bias_bf16, match_cuda_site,
    push_epi_reads, r64_ks,
};
use super::{
    CUDA_LACKS, cuda_source_with, f32_only, int4_source_with, prefill_source_entry,
    prefill_source_with, weight_accepts,
};
mod mx8;

pub(crate) use mx8::propose_mm_mx8;
use tessera_graph::{Act, Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Term, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmKind {
    Bf16,
    Bf16Split,
    Bf16SplitK,
    Bf16Tall,
    Bf16TallSplit,
    Bf16Wide,
    Mxfp8,
    Mxfp8Split,
    Mxfp8Wide,
    Mxfp8WideSplit,
    Mxfp8Tall,
    Mxfp8TallSplit,
    Int4,
    Int4Split,
    Int4Tall,
    Int4TallSplit,
}

impl MmKind {
    fn name(self) -> &'static str {
        match self {
            MmKind::Bf16 | MmKind::Bf16Split | MmKind::Bf16Tall | MmKind::Bf16Wide => {
                "fast_mm_bf16"
            }
            MmKind::Bf16SplitK | MmKind::Bf16TallSplit => "fast_mm_bf16_sk",
            MmKind::Mxfp8 | MmKind::Mxfp8Wide | MmKind::Mxfp8Tall => "fast_mm_mxfp8",
            MmKind::Mxfp8Split | MmKind::Mxfp8WideSplit | MmKind::Mxfp8TallSplit => {
                "fast_mm_mxfp8_sk"
            }
            MmKind::Int4 | MmKind::Int4Tall => "fast_mm_int4",
            MmKind::Int4Split | MmKind::Int4TallSplit => "fast_mm_int4_sk",
        }
    }

    fn entry(self) -> &'static str {
        match self {
            MmKind::Bf16 => "tessera_mm_bf16",
            MmKind::Bf16Split => "tessera_mm_bf16_ks2",
            MmKind::Bf16SplitK => "tessera_mm_bf16_sk",
            MmKind::Bf16Tall => "tessera_mm_bf64",
            MmKind::Bf16TallSplit => "tessera_mm_bf64_sk",
            MmKind::Bf16Wide => "tessera_mm_bf128",
            MmKind::Mxfp8 => "tessera_mm_mxfp8_r8",
            MmKind::Mxfp8Split => "tessera_mm_mxfp8_r8_sk",
            MmKind::Mxfp8Wide => "tessera_mm_mxfp8_r16",
            MmKind::Mxfp8WideSplit => "tessera_mm_mxfp8_r16_sk",
            MmKind::Mxfp8Tall => "tessera_mm_mxfp8_r64",
            MmKind::Mxfp8TallSplit => "tessera_mm_mxfp8_r64_sk",
            MmKind::Int4 => "tessera_mm_int4_r16",
            MmKind::Int4Split => "tessera_mm_int4_r16_sk",
            MmKind::Int4Tall => "tessera_mm_int4_r64",
            MmKind::Int4TallSplit => "tessera_mm_int4_r64_sk",
        }
    }

    fn in_prefill(self) -> bool {
        matches!(
            self,
            MmKind::Bf16Tall | MmKind::Bf16TallSplit | MmKind::Bf16Wide
        )
    }

    fn takes_bias2(self) -> bool {
        self.in_prefill()
            || matches!(
                self,
                MmKind::Int4 | MmKind::Int4Split | MmKind::Int4Tall | MmKind::Int4TallSplit
            )
    }

    fn source(self, min_blocks: Option<u32>) -> tessera_lower::Source {
        if self.in_prefill() {
            return prefill_source_entry(self.entry());
        }
        match mm_blk_def(self) {
            None => int4_source_with(self.entry(), None::<(&str, usize)>),
            Some(def) => cuda_source_with(self.entry(), min_blocks.map(|v| (def, v as usize))),
        }
    }

    fn threads(self) -> usize {
        match self {
            MmKind::Bf16Tall
            | MmKind::Bf16TallSplit
            | MmKind::Bf16Wide
            | MmKind::Mxfp8Tall
            | MmKind::Mxfp8TallSplit
            | MmKind::Int4Tall
            | MmKind::Int4TallSplit => 256,
            _ => 128,
        }
    }

    fn smem(self) -> usize {
        match self {
            MmKind::Bf16 => (2 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Bf16Split | MmKind::Bf16SplitK => (4 * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Bf16Tall | MmKind::Bf16TallSplit => (2 * 64 * 20 + 2 * 64 * 20) * 4,
            MmKind::Bf16Wide => (2 * 128 * 20 + 2 * 128 * 20) * 4,
            MmKind::Mxfp8 | MmKind::Mxfp8Split | MmKind::Mxfp8Wide | MmKind::Mxfp8WideSplit => {
                4 * 16 * 36 * 4
            }
            MmKind::Mxfp8Tall | MmKind::Mxfp8TallSplit => (64 * 136 + 64 * 36) * 4,
            MmKind::Int4 | MmKind::Int4Split => 2 * 16 * 36 * 4,
            MmKind::Int4Tall | MmKind::Int4TallSplit => (128 * 36 + 2 * 64 * 32) * 4,
        }
    }

    fn grid(self, m: usize, n: usize, ks: usize) -> (usize, usize) {
        match self {
            MmKind::Bf16 => (n / 32, m.div_ceil(16)),
            MmKind::Bf16Split | MmKind::Mxfp8Wide | MmKind::Int4 => (n / 16, m.div_ceil(16)),
            MmKind::Bf16SplitK | MmKind::Int4Split => ((n / 16) * ks, m.div_ceil(16)),
            MmKind::Bf16Wide => (n.div_ceil(128), m.div_ceil(128)),
            MmKind::Bf16Tall | MmKind::Mxfp8Tall => (n.div_ceil(64), m.div_ceil(64)),
            MmKind::Bf16TallSplit | MmKind::Mxfp8TallSplit => (n.div_ceil(64) * ks, m.div_ceil(64)),
            MmKind::Mxfp8 => (n / 16, 1),
            MmKind::Mxfp8Split | MmKind::Mxfp8WideSplit => ((n / 16) * ks, 1),
            MmKind::Int4Tall => (n.div_ceil(128), m.div_ceil(64)),
            MmKind::Int4TallSplit => (n.div_ceil(128) * ks, m.div_ceil(64)),
        }
    }

    fn step(self) -> f64 {
        match self {
            MmKind::Bf16 => 64.0,
            MmKind::Bf16Split | MmKind::Bf16SplitK => 128.0,
            MmKind::Bf16Tall | MmKind::Bf16TallSplit | MmKind::Bf16Wide => 32.0,
            MmKind::Int4 | MmKind::Int4Split | MmKind::Int4Tall | MmKind::Int4TallSplit => 256.0,
            MmKind::Mxfp8
            | MmKind::Mxfp8Split
            | MmKind::Mxfp8Wide
            | MmKind::Mxfp8WideSplit
            | MmKind::Mxfp8Tall
            | MmKind::Mxfp8TallSplit => 512.0,
        }
    }

    fn min_knob(self) -> Option<(&'static str, &'static str)> {
        Some(match self {
            MmKind::Bf16 => ("fast_mm_bf16", "min_blocks"),
            MmKind::Bf16Split => ("fast_mm_bf16", "ks2_min_blocks"),
            MmKind::Bf16SplitK => ("fast_mm_bf16_sk", "min_blocks"),
            MmKind::Mxfp8 => ("fast_mm_mxfp8", "min_blocks"),
            MmKind::Mxfp8Split => ("fast_mm_mxfp8_sk", "min_blocks"),
            MmKind::Mxfp8Wide => ("fast_mm_mxfp8", "wide_min_blocks"),
            MmKind::Mxfp8WideSplit => ("fast_mm_mxfp8_sk", "wide_min_blocks"),
            MmKind::Mxfp8Tall => ("fast_mm_mxfp8", "tall_min_blocks"),
            MmKind::Mxfp8TallSplit => ("fast_mm_mxfp8_sk", "tall_min_blocks"),
            MmKind::Bf16Tall
            | MmKind::Bf16TallSplit
            | MmKind::Bf16Wide
            | MmKind::Int4
            | MmKind::Int4Split
            | MmKind::Int4Tall
            | MmKind::Int4TallSplit => {
                return None;
            }
        })
    }

    fn is_split(self) -> bool {
        matches!(
            self,
            MmKind::Bf16SplitK
                | MmKind::Bf16TallSplit
                | MmKind::Mxfp8Split
                | MmKind::Mxfp8WideSplit
                | MmKind::Mxfp8TallSplit
                | MmKind::Int4Split
                | MmKind::Int4TallSplit
        )
    }
}

tessera_lower::knob!("fast_mm_bf16", "split_groups", 0, 2048);
tessera_lower::knob!("fast_mm_bf16", "target_groups", 1, 2048);
tessera_lower::knob!("fast_mm_bf16", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_bf16", "ks2_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_bf16_sk", "splits", 1, 16);
tessera_lower::knob!("fast_mm_bf16_sk", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_bf16_sk", "reduce_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8", "wide_rows", 1, 8);
tessera_lower::knob!("fast_mm_mxfp8", "tall_rows", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8", "target_groups", 1, 2048);
tessera_lower::knob!("fast_mm_mxfp8", "tall_min_groups", 1, 2048);
tessera_lower::knob!("fast_mm_mxfp8", "tall_target_groups", 1, 2048);
tessera_lower::knob!("fast_mm_mxfp8", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8", "wide_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8", "tall_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "splits", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "wide_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "tall_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_sk", "reduce_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_int4_sk", "splits", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_mx8", "min_rows", 16, 512);
tessera_lower::knob!("fast_mm_mxfp8_mx8", "min_groups", 1, 2048);
tessera_lower::knob!("fast_mm_mxfp8_mx8", "target_groups", 1, 2048);
tessera_lower::knob!("fast_mm_mxfp8_mx8", "prep_min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_mx8", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_mx8_sk", "splits", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_mx8_sk", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mm_mxfp8_mx8_sk", "reduce_min_blocks", 1, 16);

fn mm_blk_def(kind: MmKind) -> Option<&'static str> {
    Some(match kind {
        MmKind::Bf16 => "MM_BF16_BLK",
        MmKind::Bf16Split => "MM_BF16_KS2_BLK",
        MmKind::Bf16SplitK => "MM_BF16_SK_BLK",
        MmKind::Mxfp8 => "MM_MX_R8_BLK",
        MmKind::Mxfp8Split => "MM_MX_R8_SK_BLK",
        MmKind::Mxfp8Wide => "MM_MX_R16_BLK",
        MmKind::Mxfp8WideSplit => "MM_MX_R16_SK_BLK",
        MmKind::Mxfp8Tall => "MM_MX_R64_BLK",
        MmKind::Mxfp8TallSplit => "MM_MX_R64_SK_BLK",
        MmKind::Bf16Tall
        | MmKind::Bf16TallSplit
        | MmKind::Bf16Wide
        | MmKind::Int4
        | MmKind::Int4Split
        | MmKind::Int4Tall
        | MmKind::Int4TallSplit => {
            return None;
        }
    })
}

/// The split count a rule set, fitted to `chunks`.
fn set_ks(tree: &Tree, kernel: &str, chunks: usize) -> Option<usize> {
    let want = tree.knobs.get(kernel, "splits")?;
    Some(pick_divisor(chunks, want, want))
}

fn knob(tree: &Tree, rule: &str, name: &str, default: usize) -> usize {
    tree.knobs.get(rule, name).unwrap_or(default)
}

fn bf16_ks(m: usize, n: usize, k: usize, target: usize) -> usize {
    let chunks = k / 128;
    let base = (n / 16) * m.div_ceil(16);
    if chunks < 4 || base >= 192 {
        return 1;
    }
    let cap = if base >= 48 { 6 } else { 16 }.min(chunks / 2);
    let want = target.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

fn mxfp8_ks(n: usize, k: usize, wide: bool, target: usize) -> usize {
    let chunks = k / 512;
    let nb = n / 16;
    let max_nb = if wide { 112 } else { 256 };
    if chunks <= 1 || nb >= max_nb {
        return 1;
    }
    let cap = if nb >= 32 { 6 } else { 12 }.min(chunks);
    let want = target.div_ceil(nb).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

fn mxfp8_tile_ks(
    m: usize,
    n: usize,
    k: usize,
    tile: usize,
    min_groups: usize,
    mid_groups: usize,
    target: usize,
) -> usize {
    let chunks = k / 512;
    let base = n.div_ceil(tile) * m.div_ceil(tile);
    if chunks <= 1 || base >= min_groups {
        return 1;
    }
    let cap = if base >= mid_groups { 4 } else { 12 }.min(chunks);
    let want = target.div_ceil(base).clamp(1, cap);
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
    source = self.kind.source(self.min_blocks),
    launches = mm_launches,
    shape = Shape::grid(self.kind.grid(self.m, self.n, self.ks), self.kind.threads()),
    load = mm_load,
    unproven,
)]
struct FastMm {
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    mode: u32,
    min_blocks: Option<u32>,
    kind: MmKind,
    form: &'static dyn Stored,
}

fn mm_accepts(k: &FastMm, input: usize, form: &dyn Stored) -> bool {
    if input >= 2 {
        return epi_accepts(k.mode, input, 2, form);
    }
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
    push_epi_reads(&mut reads, k.mode, k.m, k.n);
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
    match k.kind {
        MmKind::Bf16 => {
            load.trip.bf16_flops = (2 * 2 * 16 * 32 * 64 / 128) as f64;
            load.trip.loads = 4.0;
            load.trip.width = 16.0;
        }
        MmKind::Bf16Split | MmKind::Bf16SplitK => {
            load.trip.bf16_flops = (2 * 2 * 16 * 16 * 128 / 128) as f64;
            load.trip.loads = 6.0;
            load.trip.width = 16.0;
        }
        MmKind::Bf16Wide => {
            load.trip.bf16_flops = (2 * 128 * 128 * 32 / 256) as f64;
            load.trip.loads = 4.0;
            load.trip.width = 16.0;
        }
        MmKind::Bf16Tall | MmKind::Bf16TallSplit if k.m > 64 => {
            load.trip.bf16_flops = (4 * 64 * 64 * 32 / 256) as f64;
            load.trip.loads = 3.0;
            load.trip.width = 16.0;
        }
        MmKind::Bf16Tall | MmKind::Bf16TallSplit => {}
        MmKind::Mxfp8 | MmKind::Mxfp8Split => {
            load.trip.bf16_flops = (2 * 2 * 16 * 8 * 512 / 128) as f64;
            load.trip.loads = 20.5;
            load.trip.width = 200.0 / 20.5;
            load.trip.e4m3 = 64.0;
        }
        MmKind::Mxfp8Wide | MmKind::Mxfp8WideSplit => {
            load.trip.bf16_flops = (2 * 2 * 16 * 16 * 512 / 128) as f64;
            load.trip.loads = 36.5;
            load.trip.width = 328.0 / 36.5;
            load.trip.e4m3 = 64.0;
        }
        MmKind::Mxfp8Tall | MmKind::Mxfp8TallSplit => {
            load.trip.bf16_flops = (2 * 64 * 64 * 512 / 256) as f64;
            load.trip.loads = 40.25;
            load.trip.width = 16.0;
            load.trip.e4m3 = 256.0;
        }
        MmKind::Int4 | MmKind::Int4Split => {
            load.trip.bf16_flops = (2 * 2 * 16 * 16 * 256 / 128) as f64;
            load.trip.loads = 17.125;
            load.trip.width = 144.0 / 17.125;
        }
        MmKind::Int4Tall | MmKind::Int4TallSplit => {
            load.trip.bf16_flops = (2 * 128 * 64 * 256 / 256) as f64;
            load.trip.loads = 20.5;
            load.trip.width = 16.0;
        }
    }
    Some(load)
}

fn mm_launches(k: &FastMm, step: &Step) -> Vec<Launch> {
    let shape = k.shape().expect("a grid");
    let other = if has_other(k.mode) {
        step.read(2)
    } else {
        step.read(0)
    };
    let mut args = vec![step.read(0), step.read(1), other];
    if k.kind.takes_bias2() {
        let bias = if has_bias2(k.mode) {
            step.read(3)
        } else {
            step.read(0)
        };
        args.push(bias);
    }
    args.extend([
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
    ]);
    if k.kind.is_split() {
        args.push(Arg::count(k.ks));
    }
    vec![Launch { args, shape }]
}

/// Deterministic split-K reduction and matmul epilogue.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.rocm { "rocm_mm_reduce" } else { "fast_mm_reduce" },
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = mm_red_accepts,
    emits = f32_only,
    needs = mm_red_needs(self.rocm),
    order = Order::PerConfig,
    source = if self.rocm {
        super::rocm_source(super::rocm_mm::rocm_linear_source(), "rocm_mm_reduce")
    } else {
        cuda_source_with(
            "tessera_mm_reduce",
            self.min_blocks.map(|v| ("MM_RED_BLK", v as usize)),
        )
    },
    launches = mm_red_launches,
    load = mm_red_load,
    unproven,
)]
pub(super) struct FastMmReduce {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) ks: usize,
    pub(super) mode: u32,
    pub(super) rocm: bool,
    pub(super) min_blocks: Option<u32>,
}

fn mm_red_accepts(r: &FastMmReduce, input: usize, form: &dyn Stored) -> bool {
    if input >= 1 {
        return epi_accepts(r.mode, input, 1, form);
    }
    form.name() == "f32"
}

fn mm_red_needs(rocm: bool) -> Needs {
    if rocm {
        return super::rocm_needs(0, 256, false);
    }
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
    push_epi_reads(&mut reads, r.mode, r.m, r.n);
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
    let other = if has_other(r.mode) {
        step.read(1)
    } else {
        step.read(0)
    };
    let bias = if has_bias2(r.mode) {
        step.read(2)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        other,
        bias,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::value(&r.mode),
    ];
    vec![Launch { args, shape }]
}

pub(super) struct MmSite {
    pub(super) x: T,
    pub(super) w: T,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) ks: usize,
    kind: MmKind,
    pub(super) min_blocks: Option<u32>,
    pub(super) prep_min_blocks: Option<u32>,
    pub(super) reduce_min_blocks: Option<u32>,
    pub(super) form: &'static dyn Stored,
}

pub(super) fn mm_draft<K: Kernel + 'static>(
    mm: K,
    [m, n]: [usize; 2],
    [x, w]: [T; 2],
    ks: usize,
    mode: u32,
    [other, bias2]: [Option<T>; 2],
    rocm: bool,
) -> Vec<Draft> {
    if ks > 1 {
        let red = FastMmReduce {
            m,
            n,
            ks,
            mode,
            rocm,
            min_blocks: None,
        };
        let red_reads = [Read::Draft(0)]
            .into_iter()
            .chain(other.map(Read::Node))
            .chain(bias2.map(Read::Node));
        return vec![
            Draft::new(mm, [x, w], ks * m * n),
            Draft::new(red, red_reads, m * n),
        ];
    }
    let reads = [x, w].into_iter().chain(other).chain(bias2);
    vec![Draft::new(mm, reads, m * n)]
}

impl MmSite {
    pub(super) fn reduce_draft(
        &self,
        prev: usize,
        mode: u32,
        other: Option<T>,
        bias2: Option<T>,
    ) -> Draft {
        let red = FastMmReduce {
            m: self.m,
            n: self.n,
            ks: self.ks,
            mode,
            rocm: false,
            min_blocks: self.reduce_min_blocks,
        };
        let reads = [Read::Draft(prev)]
            .into_iter()
            .chain(other.map(Read::Node))
            .chain(bias2.map(Read::Node));
        Draft::new(red, reads, self.m * self.n)
    }

    fn draft(self, mode: u32, other: Option<T>, bias2: Option<T>) -> Vec<Draft> {
        let ks = if self.kind.is_split() { self.ks } else { 1 };
        let mm = FastMm {
            m: self.m,
            n: self.n,
            k: self.k,
            ks,
            mode: if ks > 1 { 0 } else { mode },
            min_blocks: self.min_blocks,
            kind: self.kind,
            form: self.form,
        };
        if ks > 1 {
            return vec![
                Draft::new(mm, [self.x, self.w], ks * self.m * self.n),
                self.reduce_draft(0, mode, other, bias2),
            ];
        }
        let reads = [self.x, self.w].into_iter().chain(other).chain(bias2);
        vec![Draft::new(mm, reads, self.m * self.n)]
    }
}

/// Validates a multi-row 2D matmul site or explains its rejection.
fn check_mm(tree: &Tree, mat: T) -> Option<Result<MmSite, String>> {
    let base = match check_mm_base(tree, mat)? {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let MatmulBase {
        x,
        w,
        m,
        n,
        k,
        form,
    } = base;
    let split_groups = knob(tree, "fast_mm_bf16", "split_groups", 144);
    let (kind, ks) = match form.name() {
        "bf16"
            if m <= 32
                && n.is_multiple_of(16)
                && k.is_multiple_of(128)
                && (!n.is_multiple_of(32) || (n / 32) * m.div_ceil(16) < split_groups) =>
        {
            let target = knob(tree, "fast_mm_bf16", "target_groups", 320);
            let set = set_ks(tree, "fast_mm_bf16_sk", k / 128);
            let ks = set.unwrap_or_else(|| bf16_ks(m, n, k, target));
            (
                if ks > 1 {
                    MmKind::Bf16SplitK
                } else {
                    MmKind::Bf16Split
                },
                ks,
            )
        }
        "bf16" if m <= 32 && n.is_multiple_of(32) && k.is_multiple_of(64) => (MmKind::Bf16, 1),
        "bf16" if n.is_multiple_of(16) && k.is_multiple_of(16) => {
            let steps = k.div_ceil(32);
            let set = set_ks(tree, "fast_mm_bf16_sk", steps);
            let ks = set.unwrap_or_else(|| r64_ks(m, n, steps));
            if ks > 1 {
                (MmKind::Bf16TallSplit, ks)
            } else {
                (MmKind::Bf16Tall, 1)
            }
        }
        "bf16" => {
            return Some(Err(format!(
                "n={n}, k={k} not aligned for bf16 (needs n%16 & k%16)"
            )));
        }
        "mxfp8" if n.is_multiple_of(16) && k.is_multiple_of(512) => {
            let set = set_ks(tree, "fast_mm_mxfp8_sk", k / 512);
            let wide_rows = knob(tree, "fast_mm_mxfp8", "wide_rows", 8);
            let tall_rows = knob(tree, "fast_mm_mxfp8", "tall_rows", 16);
            if m <= tall_rows {
                let wide = m > wide_rows;
                let target = knob(tree, "fast_mm_mxfp8", "target_groups", 320);
                let ks = set.unwrap_or_else(|| mxfp8_ks(n, k, wide, target));
                let kind = match (wide, ks > 1) {
                    (false, true) => MmKind::Mxfp8Split,
                    (false, false) => MmKind::Mxfp8,
                    (true, true) => MmKind::Mxfp8WideSplit,
                    (true, false) => MmKind::Mxfp8Wide,
                };
                (kind, ks)
            } else {
                let min_g = knob(tree, "fast_mm_mxfp8", "tall_min_groups", 160);
                let target = knob(tree, "fast_mm_mxfp8", "tall_target_groups", 192);
                let ks = set.unwrap_or_else(|| mxfp8_tile_ks(m, n, k, 64, min_g, 64, target));
                (
                    if ks > 1 {
                        MmKind::Mxfp8TallSplit
                    } else {
                        MmKind::Mxfp8Tall
                    },
                    ks,
                )
            }
        }
        "mxfp8" if !n.is_multiple_of(16) => {
            return Some(Err(format!("n={n} is not a multiple of 16 for mxfp8")));
        }
        "mxfp8" => {
            return Some(Err(format!("k={k} is not a multiple of 512 for mxfp8")));
        }
        "int4_g32" if n.is_multiple_of(16) && k.is_multiple_of(256) => {
            let chunks = k / 256;
            let set = set_ks(tree, "fast_mm_int4_sk", chunks);
            if m <= 16 {
                let nb = n / 16;
                let ks = set.unwrap_or_else(|| {
                    if chunks <= 1 || nb >= 112 {
                        1
                    } else {
                        let cap = if nb >= 32 { 6 } else { 12 }.min(chunks);
                        pick_divisor(chunks, 320usize.div_ceil(nb).clamp(1, cap), cap)
                    }
                });
                if ks > 1 {
                    (MmKind::Int4Split, ks)
                } else {
                    (MmKind::Int4, 1)
                }
            } else {
                let base = n.div_ceil(128) * m.div_ceil(64);
                let ks = set.unwrap_or_else(|| {
                    if chunks <= 1 || base >= 96 {
                        1
                    } else {
                        let cap = if base >= 48 { 4 } else { 12 }.min(chunks);
                        pick_divisor(chunks, 144usize.div_ceil(base).clamp(1, cap), cap)
                    }
                });
                if ks > 1 {
                    (MmKind::Int4TallSplit, ks)
                } else {
                    (MmKind::Int4Tall, 1)
                }
            }
        }
        "int4_g32" if !n.is_multiple_of(16) => {
            return Some(Err(format!("n={n} is not a multiple of 16 for int4_g32")));
        }
        "int4_g32" => {
            return Some(Err(format!("k={k} is not a multiple of 256 for int4_g32")));
        }
        other => {
            return Some(Err(format!(
                "weight form {other} (supports bf16, mxfp8, int4_g32)"
            )));
        }
    };
    let min_blocks = kind
        .min_knob()
        .and_then(|(rule, name)| tree.knobs.get(rule, name))
        .map(|v| v as u32);
    let reduce_min_blocks = tree
        .knobs
        .get(kind.name(), "reduce_min_blocks")
        .map(|v| v as u32);
    Some(Ok(MmSite {
        x,
        w,
        m,
        n,
        k,
        ks,
        kind,
        min_blocks,
        prep_min_blocks: None,
        reduce_min_blocks,
        form,
    }))
}

pub(super) fn mm_site(tree: &Tree, mat: T) -> Option<MmSite> {
    check_mm(tree, mat)?.ok()
}

pub(super) trait MmDims {
    fn dims(&self) -> (usize, usize);
}

impl MmDims for MmSite {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

pub(super) fn match_mm_add<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
    allow_bf16: bool,
) -> Option<(S, u32, Option<T>)> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Add) || node.form.name() != "f32" {
        return None;
    }
    let (a, b) = (node.args[0], *node.args.get(1)?);
    for (mat, add) in [(a, b), (b, a)] {
        if mat == add || tree.root(mat) {
            continue;
        }
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        let add_node = tree.node(add);
        let add_elems = tree.elements(add);
        let form = add_node.form.name();
        let mode = if add_elems == m * n && add_node.shape == node.shape && form == "f32" {
            1
        } else if add_elems == n && add_node.shape.last() == Some(&n) && form == "f32" {
            2
        } else if allow_bf16
            && add_elems == n
            && add_node.shape.last() == Some(&n)
            && form == "bf16"
        {
            2 | 16
        } else {
            continue;
        };
        return Some((site, mode, Some(add)));
    }
    None
}

fn match_mm_mul<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>)> {
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
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            return Some((site, 3, Some(other)));
        }
    }
    for (mat, other) in [(a, b), (b, a)] {
        if mat == other || tree.root(mat) || tree.node(other).form.name() != "f32" {
            continue;
        }
        let Some(site) = pick(tree, mat) else {
            continue;
        };
        let (m, n) = site.dims();
        if tree.elements(other) == m * n && tree.node(other).shape == node.shape {
            return Some((site, 4, Some(other)));
        }
    }
    None
}

pub(super) fn match_mm_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>)> {
    if let Some(hit) = match_mm_add(tree, at, pick, false) {
        return Some(hit);
    }
    if let Some(hit) = match_mm_mul(tree, at, pick) {
        return Some(hit);
    }
    let node = tree.node(at);
    if node.term == Term::Stage(Stage::Act(Act::Silu)) && node.form.name() == "f32" {
        let mat = node.args[0];
        if !tree.root(mat)
            && let Some(site) = pick(tree, mat)
        {
            return Some((site, 5, None));
        }
    }
    Some((pick(tree, at)?, 0, None))
}

pub(crate) fn propose_mm(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (mut site, mode, other, bias2) = match_cuda_site(tree, at, mm_site)?;
    if !site.kind.takes_bias2() && !site.kind.is_split() && (is_bias_bf16(mode) || has_bias2(mode))
    {
        if site.form.name() != "bf16" {
            return None;
        }
        let steps = site.k.div_ceil(32);
        let ks =
            set_ks(tree, "fast_mm_bf16_sk", steps).unwrap_or_else(|| r64_ks(site.m, site.n, steps));
        site.ks = ks;
        site.kind = if ks > 1 {
            MmKind::Bf16TallSplit
        } else {
            MmKind::Bf16Tall
        };
    }
    Some(site.draft(mode, other, bias2))
}

fn bf128_site(tree: &Tree, mat: T) -> Option<MmSite> {
    let mut site = mm_site(tree, mat)?;
    if site.form.name() != "bf16"
        || site.m <= 64
        || site.m.div_ceil(128) * site.n.div_ceil(128) < 32
    {
        return None;
    }
    site.kind = MmKind::Bf16Wide;
    site.ks = 1;
    Some(site)
}

pub(crate) fn propose_mm_bf128(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, mode, other, bias2) = match_cuda_site(tree, at, bf128_site)?;
    Some(site.draft(mode, other, bias2))
}

fn why_mm(tree: &Tree, at: T) -> Option<String> {
    check_mm(tree, at)?.err()
}

super::cuda_rule!("fast_mm", propose_mm, why = why_mm);
super::cuda_rule!("fast_mm_bf128", propose_mm_bf128);
super::cuda_rule!("fast_mm_mx8", propose_mm_mx8);
