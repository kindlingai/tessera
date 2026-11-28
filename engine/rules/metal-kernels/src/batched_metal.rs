//! Native Metal coalesced decode and simdgroup_matrix prefill for batched matmuls.

use tessera_graph::Order;
use tessera_kernels::batched::site;
use tessera_lower::{
    Arg, Backend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree,
};

const METAL_THREADS: usize = 128;
const METAL_TILE_ROWS: usize = 8;
const METAL_MMA_MIN_ROWS: usize = 16;
const METAL_MMA_TILE: usize = 32;
const METAL_LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const METAL_SOURCE: &str = include_str!("batched_metal.metal");

const DEC_ENTRIES: [&str; 8] = [
    "bm_dec_m1",
    "bm_dec_m2",
    "bm_dec_m3",
    "bm_dec_m4",
    "bm_dec_m5",
    "bm_dec_m6",
    "bm_dec_m7",
    "bm_dec_m8",
];

/// Native Metal coalesced decode and simdgroup_matrix prefill for batched matmuls.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.mma { "batched_metal_mma" } else { "batched_metal" },
    sizes = [self.batches, self.m, self.n, self.k],
    work = self.batches * self.m * self.n * self.k,
    needs = metal_needs(self.mma),
    order = Order::PerConfig,
    source = metal_source(self),
    launches = metal_launches,
    load = metal_load,
    unproven,
)]
struct BatchedMetal {
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
    x_div: usize,
    mt: usize,
    mma: bool,
}

fn metal_needs(mma: bool) -> Needs {
    Needs {
        lanes: 32,
        group_threads: METAL_THREADS,
        group_bytes: if mma { 9000 } else { 0 },
        tile: mma,
        compiler: true,
        lacks: METAL_LACKS,
        ..Needs::default()
    }
}

fn metal_source(k: &BatchedMetal) -> Source {
    let entry = if k.mma {
        "bm_mma_f32"
    } else {
        DEC_ENTRIES[k.mt - 1]
    };
    Source::metal(METAL_SOURCE, entry)
}

fn metal_cols_per_tg(mt: usize) -> usize {
    let nc = if (2..=4).contains(&mt) { 4 } else { 2 };
    4 * nc
}

fn metal_load(k: &BatchedMetal, _: usize) -> Option<Load> {
    let (across, down, trips, group_bytes) = if k.mma {
        (
            k.n.div_ceil(METAL_MMA_TILE),
            k.m.div_ceil(METAL_MMA_TILE),
            k.k.div_ceil(METAL_MMA_TILE) as f64,
            9000.0,
        )
    } else {
        (
            k.n.div_ceil(metal_cols_per_tg(k.mt)),
            k.m.div_ceil(k.mt),
            k.k.div_ceil(128) as f64,
            0.0,
        )
    };
    let groups = across * down * k.batches;
    let x_rows = (k.batches * k.m) / k.x_div.max(1);
    Some(Load {
        threads: groups * METAL_THREADS,
        group: METAL_THREADS,
        trips,
        reads: vec![
            (4 * x_rows * k.k) as f64,
            (4 * k.batches * k.n * k.k) as f64,
        ],
        written: (4 * k.batches * k.m * k.n) as f64,
        flight: 512.0,
        group_bytes,
        ..Load::default()
    })
}

fn metal_launches(k: &BatchedMetal, step: &Step) -> Vec<Launch> {
    let (across, down) = if k.mma {
        (k.n.div_ceil(METAL_MMA_TILE), k.m.div_ceil(METAL_MMA_TILE))
    } else {
        (k.n.div_ceil(metal_cols_per_tg(k.mt)), k.m.div_ceil(k.mt))
    };
    vec![Launch {
        args: vec![
            step.read(0),
            step.read(1),
            step.out(),
            Arg::count(k.m),
            Arg::count(k.n),
            Arg::count(k.k),
            Arg::count(k.x_div),
            Arg::count(k.batches),
        ],
        shape: Shape::Grid {
            across,
            down: down * k.batches,
            threads: METAL_THREADS,
        },
    }]
}

/// Proposes the Metal kernel when the site fits.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, reads) = site(tree, at)?;
    let (len, n, k, batches) = (site.len, site.n, site.k, site.batches);
    let rows = len / n;
    if batches == 0 || n == 0 || k < 4 {
        return None;
    }
    let m = rows / batches;
    if m == 0 || !rows.is_multiple_of(batches) || !rows.is_multiple_of(site.x_rows) {
        return None;
    }
    let x_div = rows / site.x_rows;
    let (mt, mma) = if m >= METAL_MMA_MIN_ROWS {
        (0, true)
    } else if k.is_multiple_of(4) && k >= 64 {
        (m.min(METAL_TILE_ROWS), false)
    } else {
        return None;
    };
    let kernel = BatchedMetal {
        batches,
        m,
        n,
        k,
        x_div,
        mt,
        mma,
    };
    Some(vec![Draft::new(kernel, reads, len)])
}

tessera_lower::rule!("batched_metal", propose);
