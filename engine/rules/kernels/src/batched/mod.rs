//! Products of two computed values, batch by batch.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, Order};
use tessera_lower::{
    Arg, Backend, ByBackend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree,
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

/// One thread per output. Batches of `w` serve rows.
#[kernel]
fn batched(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    n: u32,
    k: u32,
    x_rows: u32,
    batches: u32,
) {
    if gid() >= len {
        return;
    }
    let row = gid() / n;
    let rows = len / n;
    let x_at = (row / (rows / x_rows)) as u64 * k as u64;
    let w_row = (row / (rows / batches)) * n + gid() % n;
    out[gid() as u64] = x.dot(x_at, w, w_row as u64 * k as u64, k);
}

/// `x[.., k]` times `w[batches, n, k]` transposed.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(batched, threads = self.len, work = self.len * self.k)]
pub(crate) struct Batched {
    pub len: usize,
    pub n: usize,
    pub k: usize,
    pub x_rows: usize,
    pub batches: usize,
}

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
    Source::Fixed {
        text: ByBackend([METAL_SOURCE, "", ""]),
        entry,
    }
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

/// A matmul whose right side is a batch of matrices.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let floats = |t: T| tree.node(t).form.dtype().is_float();
    let shape = &tree.node(w).shape;
    // Gathers of kept rows are ours, not picked weights.
    let wn = tree.node(w);
    let picked = matches!(wn.term, Term::Op(Op::Select(_))) && !tree.appends(wn.args[0]);
    if picked || shape.len() < 3 || !floats(x) || !floats(w) {
        return None;
    }
    let [.., n, k] = shape[..] else {
        return None;
    };
    let batches = tree.elements(w) / (n * k);
    let x_rows = tree.elements(x) / k;
    let len = tree.elements(at);
    let rows = len / n;
    let kernel = Batched {
        len,
        n,
        k,
        x_rows,
        batches,
    };
    let mut drafts = Vec::new();
    let m = rows / batches;
    if batches > 0
        && m > 0
        && rows.is_multiple_of(batches)
        && rows.is_multiple_of(x_rows)
        && n > 0
        && k >= 4
    {
        let x_div = rows / x_rows;
        if m >= METAL_MMA_MIN_ROWS {
            let mma = BatchedMetal {
                batches,
                m,
                n,
                k,
                x_div,
                mt: 0,
                mma: true,
            };
            drafts.push(Draft::new(mma, [x, w], len).choice(0));
        } else if k.is_multiple_of(4) && k >= 64 {
            let mt = m.min(METAL_TILE_ROWS);
            let dec = BatchedMetal {
                batches,
                m,
                n,
                k,
                x_div,
                mt,
                mma: false,
            };
            drafts.push(Draft::new(dec, [x, w], len).choice(0));
        }
    }
    drafts.push(Draft::new(kernel, [x, w], len).choice(1));
    Some(drafts)
}

tessera_lower::rule!("batched", propose);
