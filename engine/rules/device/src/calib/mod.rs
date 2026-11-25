//! Grouped SYRK and deterministic expert sort kernels and CPU references.

use crate::flatten_cuda;
use std::sync::OnceLock;
use tessera_graph::Bf16;
use tessera_graph::{Act, Op, Order, Select, Stage};
use tessera_lower::Backend as LowerBackend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

const KERNEL: &str = include_str!("grouped.cu");

/// Block count for deterministic chunked expert counting sort.
pub(crate) const SORT_BLOCKS: usize = 64;
/// Maximum routed experts supported in shared-memory histograms.
pub(crate) const MAX_EXPERTS: usize = 1024;
/// Tile side length for grouped SYRK and BF16 GEMM.
pub const TILE: usize = 128;

/// Flattened CUDA source for calibration SYRK and BF16 grouped MoE.
pub(crate) fn cuda_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_cuda("calib_grouped.kernel", KERNEL))
}

/// Deterministic stable counting sort matching `calib_sort_*` kernels.
pub fn sort_cpu(ids: &[u32], experts: usize, first: u32) -> (Vec<u32>, Vec<u32>, Vec<i64>) {
    let mut counts = vec![0u32; experts];
    for &raw in ids {
        if raw >= first {
            let e = (raw - first) as usize;
            if e < experts {
                counts[e] += 1;
            }
        }
    }
    let mut offsets = vec![0u32; experts + 1];
    for e in 0..experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let total = offsets[experts] as usize;
    let mut perm = vec![0u32; total];
    let mut cur = offsets[..experts].to_vec();
    for (i, &raw) in ids.iter().enumerate() {
        if raw >= first {
            let e = (raw - first) as usize;
            if e < experts {
                let at = cur[e] as usize;
                perm[at] = i as u32;
                cur[e] += 1;
            }
        }
    }
    let i64_counts: Vec<i64> = counts.into_iter().map(i64::from).collect();
    (offsets, perm, i64_counts)
}

/// Rounds `v` to BF16 and widens back to `f32`.
pub(crate) fn round_bf16(v: f32) -> f32 {
    f32::from(Bf16::from(v))
}

/// Accumulates packed upper-triangle tiles `X_e^T X_e` in fixed order.
pub fn syrk_tiles_cpu(
    x: &[f32],
    perm: &[u32],
    offsets: &[u32],
    dim: usize,
    tile: usize,
    repeat: usize,
    acc: &mut [f64],
) {
    assert!(tile > 0 && dim.is_multiple_of(tile));
    let experts = offsets.len() - 1;
    let blocks = dim / tile;
    let total_tiles = blocks * (blocks + 1) / 2;
    assert_eq!(acc.len(), experts * total_tiles * tile * tile);
    for e in 0..experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        if start == end {
            continue;
        }
        let mut tidx = 0usize;
        for br in 0..blocks {
            for bc in br..blocks {
                let dst = &mut acc[(e * total_tiles + tidx) * tile * tile..][..tile * tile];
                for &pick in &perm[start..end] {
                    let row = (pick as usize) / repeat;
                    let xr = &x[row * dim + br * tile..][..tile];
                    let xc = &x[row * dim + bc * tile..][..tile];
                    for lr in 0..tile {
                        let a = f64::from(round_bf16(xr[lr]));
                        for lc in 0..tile {
                            dst[lr * tile + lc] += a * f64::from(round_bf16(xc[lc]));
                        }
                    }
                }
                tidx += 1;
            }
        }
    }
}

/// Accumulates full symmetric `[dim, dim]` dense Gram `X^T X`.
pub fn syrk_dense_cpu(x: &[f32], rows: usize, dim: usize, acc: &mut [f64]) {
    assert_eq!(x.len(), rows * dim);
    assert_eq!(acc.len(), dim * dim);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        for i in 0..dim {
            let a = f64::from(round_bf16(xr[i]));
            for j in i..dim {
                let prod = a * f64::from(round_bf16(xr[j]));
                acc[i * dim + j] += prod;
                if i != j {
                    acc[j * dim + i] += prod;
                }
            }
        }
    }
}

const CUDA_LACKS: &[LowerBackend] = &[LowerBackend::Metal, LowerBackend::Rocm];
const THREADS: usize = 256;
const MIN_ROWS: usize = 16;
/// `CALIB_BK` and `CALIB_GEMM_STRIDE` in `grouped.cu`.
const BK: usize = 32;
const GEMM_STRIDE: usize = BK + 8;
/// `sizeof(gemm_smem)`: two stages of A and B, then rows.
const SMEM: usize = 2 * 2 * TILE * GEMM_STRIDE * 2 + TILE * 4;

fn is_plain_float(form: &dyn Stored) -> bool {
    matches!(form.name(), "f32" | "bf16")
}

fn bf16_needs(group_bytes: usize, group_threads: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes,
        group_threads,
        compiler: true,
        bf16_tile: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn fixed_cuda(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend(["", cuda_source(), ""]),
        entry,
    }
}

/// Pass 1 of deterministic expert sort: per-block histograms.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "calib_sort_hist",
    sizes = [self.rows, self.count],
    work = self.rows,
    accepts = hist_accepts,
    emits = u32_only,
    needs = bf16_needs(MAX_EXPERTS * 4, THREADS),
    order = Order::PerConfig,
    source = fixed_cuda("calib_sort_hist"),
    launches = hist_launches,
    load = hist_load,
    unproven,
)]
struct SortHist {
    rows: usize,
    count: usize,
}

fn u32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "u32"
}

fn hist_accepts(_: &SortHist, _: usize, form: &dyn Stored) -> bool {
    form.name() == "u32"
}

fn sort_load(group: usize, rows: usize, reads: Vec<f64>, written: f64) -> Option<Load> {
    Some(Load {
        threads: SORT_BLOCKS * group,
        group,
        trips: (rows / SORT_BLOCKS).max(1) as f64,
        reads,
        written,
        flight: 256.0,
        ..Load::default()
    })
}

fn hist_load(k: &SortHist, _: usize) -> Option<Load> {
    sort_load(
        THREADS,
        k.rows,
        vec![(4 * k.rows) as f64],
        (4 * SORT_BLOCKS * k.count) as f64,
    )
}

fn hist_launches(k: &SortHist, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: SORT_BLOCKS,
        down: 1,
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(k.rows),
        Arg::count(k.count),
        Arg::count(0),
        Arg::count(SORT_BLOCKS),
    ];
    vec![Launch { args, shape }]
}

/// Pass 2 of deterministic sort: prefix sums and tile schedule.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "calib_sort_prefix",
    sizes = [self.count, self.max_tiles],
    work = self.count * SORT_BLOCKS,
    accepts = prefix_accepts,
    emits = u32_only,
    needs = bf16_needs(MAX_EXPERTS * 4, THREADS),
    order = Order::PerConfig,
    source = fixed_cuda("calib_sort_prefix"),
    launches = prefix_launches,
    load = prefix_load,
    unproven,
)]
struct SortPrefix {
    count: usize,
    max_tiles: usize,
}

fn prefix_accepts(_: &SortPrefix, _: usize, form: &dyn Stored) -> bool {
    form.name() == "u32"
}

fn prefix_load(k: &SortPrefix, _: usize) -> Option<Load> {
    let words = SORT_BLOCKS * k.count + k.count + 1 + 1 + 2 * k.max_tiles;
    Some(Load {
        threads: THREADS,
        group: THREADS,
        trips: SORT_BLOCKS as f64,
        reads: vec![(4 * SORT_BLOCKS * k.count) as f64],
        written: (4 * words) as f64,
        flight: 128.0,
        ..Load::default()
    })
}

fn prefix_launches(k: &SortPrefix, step: &Step) -> Vec<Launch> {
    let off_start = SORT_BLOCKS * k.count;
    let meta_start = off_start + k.count + 1;
    let zero: u64 = 0;
    let shape = Shape::Grid {
        across: 1,
        down: 1,
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.out(),
        Arg::at(step.write, off_start),
        Arg::value(&zero),
        Arg::at(step.write, meta_start),
        Arg::count(k.count),
        Arg::count(SORT_BLOCKS),
        Arg::count(TILE),
    ];
    vec![Launch { args, shape }]
}

/// Pass 3 of deterministic sort: stable scatter into `perm`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "calib_sort_scatter",
    sizes = [self.rows, self.count],
    work = self.rows,
    accepts = scatter_accepts,
    emits = u32_only,
    needs = bf16_needs(MAX_EXPERTS * 4, 32),
    order = Order::PerConfig,
    source = fixed_cuda("calib_sort_scatter"),
    launches = scatter_launches,
    load = scatter_load,
    unproven,
)]
struct SortScatter {
    rows: usize,
    count: usize,
}

fn scatter_accepts(_: &SortScatter, _: usize, form: &dyn Stored) -> bool {
    form.name() == "u32"
}

fn scatter_load(k: &SortScatter, _: usize) -> Option<Load> {
    sort_load(
        32,
        k.rows,
        vec![(4 * k.rows) as f64, (4 * SORT_BLOCKS * k.count) as f64],
        (4 * k.rows) as f64,
    )
}

fn scatter_launches(k: &SortScatter, step: &Step) -> Vec<Launch> {
    let off_start = SORT_BLOCKS * k.count;
    let shape = Shape::Grid {
        across: SORT_BLOCKS,
        down: 1,
        threads: 32,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        Arg::at(step.reads[1], off_start),
        step.out(),
        Arg::count(k.rows),
        Arg::count(k.count),
        Arg::count(0),
        Arg::count(SORT_BLOCKS),
    ];
    vec![Launch { args, shape }]
}

fn gemm_entry(x_f32: bool, out_f32: bool, fused: bool) -> &'static str {
    match (fused, x_f32, out_f32) {
        (false, false, false) => "pbf16_gemm_bf16_bf16",
        (false, true, true) => "pbf16_gemm_f32_f32",
        (false, false, true) => "pbf16_gemm_bf16_f32",
        (false, true, false) => "pbf16_gemm_f32_bf16",
        (true, false, false) => "pbf16_gu_bf16_bf16",
        (true, true, true) => "pbf16_gu_f32_f32",
        (true, false, true) => "pbf16_gu_bf16_f32",
        (true, true, false) => "pbf16_gu_f32_bf16",
    }
}

/// Grouped BF16 Tensor-Core GEMM over expert-sorted picks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.fused { "pbf16_gu" } else { "pbf16_gemm" },
    sizes = [self.rows, self.n, self.k],
    work = self.rows * self.n * self.k * if self.fused { 2 } else { 1 },
    accepts = gemm_accepts,
    emits = gemm_emits,
    needs = bf16_needs(SMEM, THREADS),
    order = Order::PerConfig,
    source = fixed_cuda(gemm_entry(self.x_form.name() == "f32", self.out_form.name() == "f32", self.fused)),
    launches = gemm_launches,
    load = gemm_load,
    unproven,
)]
struct PickedBf16 {
    rows: usize,
    n: usize,
    k: usize,
    count: usize,
    stride: usize,
    offset: usize,
    repeat: usize,
    max_tiles: usize,
    fused: bool,
    x_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
}

fn gemm_accepts(k: &PickedBf16, input: usize, form: &dyn Stored) -> bool {
    let sort_idx = if k.fused { 3 } else { 2 };
    if input == 0 {
        form.name() == k.x_form.name()
    } else if input < sort_idx {
        form.name() == "bf16"
    } else {
        form.name() == "u32"
    }
}

fn gemm_emits(k: &PickedBf16, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn gemm_load(k: &PickedBf16, _: usize) -> Option<Load> {
    let n_tile = if k.fused { 64 } else { TILE };
    let active = k.count.min(k.rows);
    let est_tiles = k.rows.div_ceil(TILE).max(active);
    let groups = (k.n / n_tile) * est_tiles;
    let w_factor = if k.fused { 2 } else { 1 };
    let reads = vec![
        (w_factor * 2 * active * k.n * k.k) as f64,
        k.x_form.bytes((k.rows / k.repeat) * k.k) as f64,
    ];
    let mut load = Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (k.k / BK) as f64,
        reads,
        written: k.out_form.bytes(k.rows * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn gemm_launches(k: &PickedBf16, step: &Step) -> Vec<Launch> {
    let n_tile = if k.fused { 64 } else { TILE };
    let off_start = SORT_BLOCKS * k.count;
    let meta_start = off_start + k.count + 1;
    let shape = Shape::Grid {
        across: k.n / n_tile,
        down: k.max_tiles,
        threads: THREADS,
    };
    let args = if k.fused {
        vec![
            step.read(0),
            step.read(1),
            step.read(2),
            Arg::at(step.reads[3], off_start),
            step.read(4),
            Arg::at(step.reads[3], meta_start),
            step.out(),
            Arg::count(k.n),
            Arg::count(k.k),
            Arg::count(k.stride),
            Arg::count(k.repeat),
        ]
    } else {
        vec![
            step.read(0),
            step.read(1),
            Arg::at(step.reads[2], off_start),
            step.read(3),
            Arg::at(step.reads[2], meta_start),
            step.out(),
            Arg::count(k.n),
            Arg::count(k.k),
            Arg::count(k.stride),
            Arg::count(k.offset),
            Arg::count(k.repeat),
        ]
    };
    vec![Launch { args, shape }]
}

struct PickSite {
    x: T,
    w: T,
    ids: T,
    rows: usize,
    n: usize,
    k: usize,
    count: usize,
    repeat: usize,
    x_form: &'static dyn Stored,
}

fn match_bf16_pick(tree: &Tree, mat: T, n_align: usize) -> Option<PickSite> {
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, pick) = (node.args[0], node.args[1]);
    let pnode = tree.node(pick);
    if pnode.term != Term::Op(Op::Select(Select::Gather)) || tree.root(pick) {
        return None;
    }
    let (w, ids) = (pnode.args[0], pnode.args[1]);
    if !tree.leaf(w) || tree.node(w).form.name() != "bf16" || tree.node(ids).form.name() != "u32" {
        return None;
    }
    // Picks outside a shared part would leave their rows unwritten.
    if tree.share(w).is_some() {
        return None;
    }
    let x_form = tree.node(x).form;
    if !is_plain_float(x_form) {
        return None;
    }
    let &[count, n, k] = tree.node(w).shape.as_slice() else {
        return None;
    };
    if count == 0 || count > MAX_EXPERTS || !n.is_multiple_of(n_align) || !k.is_multiple_of(BK) {
        return None;
    }
    let rows = tree.elements(ids);
    let x_rows = tree.elements(x) / k;
    if rows < MIN_ROWS || x_rows == 0 || !rows.is_multiple_of(x_rows) {
        return None;
    }
    Some(PickSite {
        x,
        w,
        ids,
        rows,
        n,
        k,
        count,
        repeat: rows / x_rows,
        x_form,
    })
}

fn build_drafts(site: &PickSite, out_form: &'static dyn Stored, w_up: Option<T>) -> Vec<Draft> {
    let max_tiles = site.count + site.rows.div_ceil(TILE);
    let hist = SortHist {
        rows: site.rows,
        count: site.count,
    };
    let prefix = SortPrefix {
        count: site.count,
        max_tiles,
    };
    let scatter = SortScatter {
        rows: site.rows,
        count: site.count,
    };
    let gemm = PickedBf16 {
        rows: site.rows,
        n: site.n,
        k: site.k,
        count: site.count,
        stride: site.n,
        offset: 0,
        repeat: site.repeat,
        max_tiles,
        fused: w_up.is_some(),
        x_form: site.x_form,
        out_form,
    };
    let prefix_len = SORT_BLOCKS * site.count + site.count + 1 + 1 + 2 * max_tiles;
    let mut gemm_reads = vec![Read::Node(site.x), Read::Node(site.w)];
    if let Some(wu) = w_up {
        gemm_reads.push(Read::Node(wu));
    }
    gemm_reads.extend([Read::Draft(1), Read::Draft(2)]);
    vec![
        Draft::new(hist, [Read::Node(site.ids)], SORT_BLOCKS * site.count),
        Draft::new(prefix, [Read::Draft(0)], prefix_len),
        Draft::new(scatter, [Read::Node(site.ids), Read::Draft(1)], site.rows),
        Draft::new(gemm, gemm_reads, site.rows * site.n),
    ]
}

fn match_fused_gu(tree: &Tree, at: T, out_form: &'static dyn Stored) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Stage(Stage::Mul) {
        return None;
    }
    let (a, b) = (node.args[0], *node.args.get(1)?);
    let silu = Term::Stage(Stage::Act(Act::Silu));
    for (cand, other) in [(a, b), (b, a)] {
        if cand == other || tree.root(cand) || tree.root(other) {
            continue;
        }
        let cnode = tree.node(cand);
        if cnode.term != silu {
            continue;
        }
        let gate_mat = cnode.args[0];
        if tree.root(gate_mat) {
            continue;
        }
        let Some(gs) = match_bf16_pick(tree, gate_mat, 64) else {
            continue;
        };
        let Some(us) = match_bf16_pick(tree, other, 64) else {
            continue;
        };
        let same = gs.x == us.x
            && gs.ids == us.ids
            && gs.n == us.n
            && gs.k == us.k
            && gs.count == us.count;
        if same {
            return Some(build_drafts(&gs, out_form, Some(us.w)));
        }
    }
    None
}

/// Proposes grouped BF16 Tensor-Core MoE matmuls on CUDA.
pub(crate) fn propose_picked_bf16(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let out_form = tree.node(at).form;
    if !is_plain_float(out_form) {
        return None;
    }
    if let Some(drafts) = match_fused_gu(tree, at, out_form) {
        return Some(drafts);
    }
    let site = match_bf16_pick(tree, at, TILE)?;
    Some(build_drafts(&site, out_form, None))
}

tessera_lower::rule!("picked_bf16", propose_picked_bf16);
