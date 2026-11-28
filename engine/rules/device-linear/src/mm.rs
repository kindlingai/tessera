//! Streaming BF16 and MXFP8 Tensor-Core small-batch matmul rule.

use super::epi::{
    MatmulBase, MmDims, check_mm_base, epi_accepts, has_bias2, has_other, match_mm_site,
    pick_divisor, push_epi_reads,
};
use super::{
    CUDA_LACKS, cuda_source_with, f32_only, int4_source_with, io_defs, prefill_source_with,
    rocm_needs, weight_io_accepts, x_accepts, x_defs,
};
mod mx8;

use mx8::{Mxfp8Mx8, Mxfp8Prep, mxfp8_pre_len};
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MmKind {
    Bf16,
    Bf16Split,
    Bf16SplitK,
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
            MmKind::Bf16 | MmKind::Bf16Split => "fast_mm_bf16",
            MmKind::Bf16SplitK => "fast_mm_bf16_sk",
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

    fn source(
        self,
        min_blocks: Option<u32>,
        x_bf16: bool,
        y_bf16: bool,
        o_bf16: bool,
    ) -> tessera_lower::Source {
        match mm_blk_def(self) {
            None => int4_source_with(self.entry(), io_defs(None, x_bf16, y_bf16, o_bf16)),
            Some(def) => cuda_source_with(
                self.entry(),
                io_defs(
                    min_blocks.map(|v| (def, v as usize)),
                    x_bf16,
                    y_bf16,
                    o_bf16,
                ),
            ),
        }
    }

    fn threads(self) -> usize {
        match self {
            MmKind::Mxfp8Tall
            | MmKind::Mxfp8TallSplit
            | MmKind::Int4Tall
            | MmKind::Int4TallSplit => 256,
            _ => 128,
        }
    }

    fn smem(self, x_bf16: bool) -> usize {
        let xp = if x_bf16 { 1 } else { 2 };
        match self {
            MmKind::Bf16 => (xp * 16 * 36 + 4 * 8 * 36) * 4,
            MmKind::Bf16Split | MmKind::Bf16SplitK => (xp * 2 * 16 * 36 + 4 * 8 * 36) * 4,
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
            MmKind::Mxfp8 => (n / 16, 1),
            MmKind::Mxfp8Split | MmKind::Mxfp8WideSplit => ((n / 16) * ks, 1),
            MmKind::Mxfp8Tall => (n.div_ceil(64), m.div_ceil(64)),
            MmKind::Mxfp8TallSplit => (n.div_ceil(64) * ks, m.div_ceil(64)),
            MmKind::Int4Tall => (n.div_ceil(128), m.div_ceil(64)),
            MmKind::Int4TallSplit => (n.div_ceil(128) * ks, m.div_ceil(64)),
        }
    }

    fn step(self) -> f64 {
        match self {
            MmKind::Bf16 => 64.0,
            MmKind::Bf16Split | MmKind::Bf16SplitK => 128.0,
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
            MmKind::Int4 | MmKind::Int4Split | MmKind::Int4Tall | MmKind::Int4TallSplit => {
                return None;
            }
        })
    }

    fn is_split(self) -> bool {
        matches!(
            self,
            MmKind::Bf16SplitK
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
        MmKind::Int4 | MmKind::Int4Split | MmKind::Int4Tall | MmKind::Int4TallSplit => {
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
    emits = mm_emits,
    needs = mm_needs(self),
    order = Order::PerConfig,
    source = self.kind.source(self.min_blocks, self.x_bf16, self.y_bf16, self.o_bf16),
    launches = mm_launches,
    shape = Shape::grid(self.kind.grid(self.m, self.n, self.ks), self.kind.threads()),
    load = mm_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.k, self.ks, self.mode, self.min_blocks, self.kind, self.form.name(),
        (self.x_bf16, self.y_bf16, self.o_bf16),
    )),
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
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
    form: &'static dyn Stored,
}

fn mm_accepts(k: &FastMm, input: usize, form: &dyn Stored) -> bool {
    weight_io_accepts(input, k.x_bf16, k.o_bf16, k.form, form)
}

fn mm_emits(k: &FastMm, form: &dyn Stored) -> bool {
    x_accepts(k.y_bf16, form)
}

fn mm_needs(k: &FastMm) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: k.kind.smem(k.x_bf16),
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
    let x_elem = if k.x_bf16 { 2 } else { 4 };
    let o_elem = if k.o_bf16 { 2 } else { 4 };
    let y_elem = if k.y_bf16 { 2 } else { 4 };
    let passes = if k.x_bf16 { 1 } else { 2 };
    let mut reads = vec![k.form.bytes(k.n * k.k) as f64, (x_elem * k.m * k.k) as f64];
    if matches!(k.mode, 1 | 3 | 4) {
        reads.push((o_elem * k.m * k.n) as f64);
    } else if k.mode == 2 {
        reads.push((o_elem * k.n) as f64);
    }
    let mut load = Load {
        threads: groups * threads,
        group: threads,
        trips: k.k as f64 / (k.kind.step() * k.ks as f64),
        reads,
        written: (y_elem * k.ks * k.m * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    match k.kind {
        MmKind::Bf16 => {
            load.trip.bf16_flops = (passes * 2 * 16 * 32 * 64 / 128) as f64;
            load.trip.loads = 4.0;
            load.trip.width = 16.0;
        }
        MmKind::Bf16Split | MmKind::Bf16SplitK => {
            load.trip.bf16_flops = (passes * 2 * 16 * 16 * 128 / 128) as f64;
            load.trip.loads = 6.0;
            load.trip.width = 16.0;
        }
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
    let other = if matches!(k.mode, 1..=4) {
        step.read(2)
    } else {
        step.read(0)
    };
    let mut args = vec![
        step.read(0),
        step.read(1),
        other,
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
    ];
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
    emits = mm_red_emits,
    needs = mm_red_needs(self.rocm),
    order = Order::PerConfig,
    source = if self.rocm {
        tessera_lower::Source::rocm(super::rocm_mm::rocm_linear_source(), "rocm_mm_reduce")
    } else {
        cuda_source_with(
            "tessera_mm_reduce",
            io_defs(
                self.min_blocks.map(|v| ("MM_RED_BLK", v as usize)),
                false,
                self.y_bf16,
                self.o_bf16,
            ),
        )
    },
    launches = mm_red_launches,
    load = mm_red_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.ks, self.mode, self.rocm, self.min_blocks, self.y_bf16, self.o_bf16,
    )),
    unproven,
)]
struct FastMmReduce {
    m: usize,
    n: usize,
    ks: usize,
    mode: u32,
    rocm: bool,
    min_blocks: Option<u32>,
    y_bf16: bool,
    o_bf16: bool,
}

fn mm_red_accepts(r: &FastMmReduce, input: usize, form: &dyn Stored) -> bool {
    if r.rocm && input >= 1 {
        return epi_accepts(r.mode, input, 1, form);
    }
    match input {
        1 => x_accepts(r.o_bf16, form),
        _ => form.name() == "f32",
    }
}

fn mm_red_emits(r: &FastMmReduce, form: &dyn Stored) -> bool {
    x_accepts(r.y_bf16, form)
}

fn mm_red_needs(rocm: bool) -> Needs {
    if rocm {
        return rocm_needs(0, 256, false);
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
    let o_elem = if r.o_bf16 { 2 } else { 4 };
    let y_elem = if r.y_bf16 { 2 } else { 4 };
    let mut reads = vec![(4 * r.ks * r.m * r.n) as f64];
    if r.rocm {
        push_epi_reads(&mut reads, r.mode, r.m, r.n);
    } else if matches!(r.mode, 1 | 3 | 4) {
        reads.push((o_elem * r.m * r.n) as f64);
    } else if r.mode == 2 {
        reads.push((o_elem * r.n) as f64);
    }
    Some(Load {
        threads: (r.m * r.n).div_ceil(256) * 256,
        group: 256,
        trips: r.ks as f64,
        reads,
        written: (y_elem * r.m * r.n) as f64,
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
    let mut args = vec![step.read(0), other];
    if r.rocm {
        let bias = if has_bias2(r.mode) {
            step.read(2)
        } else {
            step.read(0)
        };
        args.push(bias);
    }
    args.extend([
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::value(&r.mode),
    ]);
    vec![Launch { args, shape }]
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
            y_bf16: false,
            o_bf16: false,
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

struct MmSite {
    x: T,
    w: T,
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    kind: MmKind,
    min_blocks: Option<u32>,
    prep_min_blocks: Option<u32>,
    reduce_min_blocks: Option<u32>,
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
    form: &'static dyn Stored,
}

impl MmDims for MmSite {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    fn set_io(&mut self, y_bf16: bool, o_bf16: bool) {
        self.y_bf16 = y_bf16;
        self.o_bf16 = o_bf16;
    }
}

impl MmSite {
    fn reduce_draft(&self, prev: usize, mode: u32, other: Option<T>) -> Draft {
        let red = FastMmReduce {
            m: self.m,
            n: self.n,
            ks: self.ks,
            mode,
            rocm: false,
            min_blocks: self.reduce_min_blocks,
            y_bf16: self.y_bf16,
            o_bf16: self.o_bf16,
        };
        let mut reads = vec![Read::Draft(prev)];
        if let Some(extra) = other {
            reads.push(Read::Node(extra));
        }
        Draft::new(red, reads, self.m * self.n)
    }

    fn draft(self, mode: u32, other: Option<T>) -> Vec<Draft> {
        if self.kind.is_split() {
            let mm = FastMm {
                m: self.m,
                n: self.n,
                k: self.k,
                ks: self.ks,
                mode: 0,
                min_blocks: self.min_blocks,
                kind: self.kind,
                x_bf16: self.x_bf16,
                y_bf16: false,
                o_bf16: false,
                form: self.form,
            };
            return vec![
                Draft::new(mm, [self.x, self.w], self.ks * self.m * self.n),
                self.reduce_draft(0, mode, other),
            ];
        }
        let kernel = FastMm {
            m: self.m,
            n: self.n,
            k: self.k,
            ks: 1,
            mode,
            min_blocks: self.min_blocks,
            kind: self.kind,
            x_bf16: self.x_bf16,
            y_bf16: self.y_bf16,
            o_bf16: self.o_bf16,
            form: self.form,
        };
        let mut reads = vec![self.x, self.w];
        if let Some(extra) = other {
            reads.push(extra);
        }
        vec![Draft::new(kernel, reads, self.m * self.n)]
    }

    fn draft_mx8(self, mode: u32, other: Option<T>) -> Vec<Draft> {
        let prep = Mxfp8Prep {
            m: self.m,
            k: self.k,
            min_blocks: self.prep_min_blocks,
            x_bf16: self.x_bf16,
        };
        let pre_len = mxfp8_pre_len(self.m, self.k);
        if self.ks > 1 {
            let mm = Mxfp8Mx8 {
                m: self.m,
                n: self.n,
                k: self.k,
                ks: self.ks,
                mode: 0,
                min_blocks: self.min_blocks,
                y_bf16: false,
                o_bf16: false,
                form: self.form,
            };
            return vec![
                Draft::new(prep, [Read::Node(self.x)], pre_len),
                Draft::new(
                    mm,
                    [Read::Draft(0), Read::Node(self.w)],
                    self.ks * self.m * self.n,
                ),
                self.reduce_draft(1, mode, other),
            ];
        }
        let mm = Mxfp8Mx8 {
            m: self.m,
            n: self.n,
            k: self.k,
            ks: 1,
            mode,
            min_blocks: self.min_blocks,
            y_bf16: self.y_bf16,
            o_bf16: self.o_bf16,
            form: self.form,
        };
        let mut reads = vec![Read::Draft(0), Read::Node(self.w)];
        if let Some(extra) = other {
            reads.push(Read::Node(extra));
        }
        vec![
            Draft::new(prep, [Read::Node(self.x)], pre_len),
            Draft::new(mm, reads, self.m * self.n),
        ]
    }
}

/// Validates a multi-row 2D matmul site or explains its rejection.
fn check_mm(tree: &Tree, mat: T) -> Option<Result<MmSite, String>> {
    let MatmulBase {
        x,
        w,
        m,
        n,
        k,
        x_bf16,
        y_bf16,
        form,
    } = match check_mm_base(tree, mat, true)? {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    let split_groups = knob(tree, "fast_mm_bf16", "split_groups", 144);
    let (kind, ks) = match form.name() {
        "bf16"
            if n.is_multiple_of(16)
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
        "bf16" if n.is_multiple_of(32) && k.is_multiple_of(64) => (MmKind::Bf16, 1),
        "bf16" => {
            return Some(Err(format!(
                "n={n}, k={k} not aligned for bf16 (needs n%32 & k%64, or n%16 & k%128)"
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
        x_bf16,
        y_bf16,
        o_bf16: false,
        form,
    }))
}

fn mm_site(tree: &Tree, mat: T) -> Option<MmSite> {
    check_mm(tree, mat)?.ok()
}

fn mx8_site(tree: &Tree, mat: T) -> Option<MmSite> {
    let mut site = mm_site(tree, mat)?;
    if site.form.name() != "mxfp8" || site.m < knob(tree, "fast_mm_mxfp8_mx8", "min_rows", 64) {
        return None;
    }
    let chunks = site.k / 512;
    let set = set_ks(tree, "fast_mm_mxfp8_mx8_sk", chunks)
        .or_else(|| set_ks(tree, "fast_mm_mxfp8_sk", chunks));
    let min_g = knob(tree, "fast_mm_mxfp8_mx8", "min_groups", 40);
    let target = knob(tree, "fast_mm_mxfp8_mx8", "target_groups", 64);
    site.ks = set.unwrap_or_else(|| mxfp8_tile_ks(site.m, site.n, site.k, 128, min_g, 24, target));
    let mx_rule = if site.ks > 1 {
        "fast_mm_mxfp8_mx8_sk"
    } else {
        "fast_mm_mxfp8_mx8"
    };
    site.min_blocks = tree.knobs.get(mx_rule, "min_blocks").map(|v| v as u32);
    site.prep_min_blocks = tree
        .knobs
        .get("fast_mm_mxfp8_mx8", "prep_min_blocks")
        .map(|v| v as u32);
    site.reduce_min_blocks = tree
        .knobs
        .get("fast_mm_mxfp8_mx8_sk", "reduce_min_blocks")
        .map(|v| v as u32);
    Some(site)
}

pub(crate) fn propose_mm(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, mode, other) = match_mm_site(tree, at, mm_site, true)?;
    Some(site.draft(mode, other))
}

pub(crate) fn propose_mm_mx8(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, mode, other) = match_mm_site(tree, at, mx8_site, true)?;
    Some(site.draft_mx8(mode, other))
}

fn why_mm(tree: &Tree, at: T) -> Option<String> {
    check_mm(tree, at)?.err()
}

super::cuda_rule!("fast_mm", propose_mm, why = why_mm);
super::cuda_rule!("fast_mm_mx8", propose_mm_mx8);
