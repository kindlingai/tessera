//! Native Metal block-quantized matmul decode for small batches.

use tessera_graph::{Op, Order};
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Step, T, Term, Tree};

const THREADS: usize = 128;
const TILE_ROWS: usize = 4;
const SMALL_N: usize = 2048;
const MMA_SMEM: usize = (8 * 36 + 32 * 36) * 4;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("block_metal.metal");
const GUESS: f64 = 1.0e-13;

const ENTRIES: &[(&str, [&str; 8], Option<&str>)] = &[
    (
        "mxfp8",
        [
            "bm_mxfp8_m1",
            "bm_mxfp8_m2",
            "bm_mxfp8_m3",
            "bm_mxfp8_m4",
            "bm_mxfp8_m5",
            "bm_mxfp8_m6",
            "bm_mxfp8_m7",
            "bm_mxfp8_m8",
        ],
        None,
    ),
    (
        "q8_0",
        [
            "bm_q8_0_m1",
            "bm_q8_0_m2",
            "bm_q8_0_m3",
            "bm_q8_0_m4",
            "bm_q8_0_m5",
            "bm_q8_0_m6",
            "bm_q8_0_m7",
            "bm_q8_0_m8",
        ],
        None,
    ),
    (
        "int4_g32",
        [
            "bm_int4_g32_m1",
            "bm_int4_g32_m2",
            "bm_int4_g32_m3",
            "bm_int4_g32_m4",
            "bm_int4_g32_m5",
            "bm_int4_g32_m6",
            "bm_int4_g32_m7",
            "bm_int4_g32_m8",
        ],
        Some("bm_int4_g32_m1_c2"),
    ),
    (
        "q4_0",
        [
            "bm_q4_0_m1",
            "bm_q4_0_m2",
            "bm_q4_0_m3",
            "bm_q4_0_m4",
            "bm_q4_0_m5",
            "bm_q4_0_m6",
            "bm_q4_0_m7",
            "bm_q4_0_m8",
        ],
        Some("bm_q4_0_m1_c2"),
    ),
];

fn uses_mma(m: usize) -> bool {
    (5..=8).contains(&m)
}

fn cols_per_tg(form: &str, m: usize, n: usize) -> usize {
    if uses_mma(m) {
        return 32;
    }
    match form {
        "int4_g32" | "q4_0" if m == 1 && n <= SMALL_N => 8,
        "int4_g32" | "q4_0" if m == 1 => 32,
        _ => 16,
    }
}

fn row_tiles(m: usize) -> usize {
    if uses_mma(m) {
        1
    } else {
        m.div_ceil(TILE_ROWS)
    }
}

fn entry(form: &str, m: usize, n: usize) -> Option<&'static str> {
    let (_, names, small_m1) = ENTRIES.iter().find(|(name, _, _)| *name == form)?;
    if m == 1
        && n <= SMALL_N
        && let Some(ent) = small_m1
    {
        return Some(ent);
    }
    let idx = if m <= 8 {
        m.checked_sub(1)?
    } else {
        TILE_ROWS - 1
    };
    Some(names[idx])
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.form.suffixed("block_metal"),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = if self.m <= 8 { GUESS } else { tessera_lower::GUESS },
    accepts = accepts,
    emits = crate::f32_only,
    needs = metal_needs(self),
    order = Order::PerConfig,
    source = tessera_lower::Source::metal(SOURCE, self.entry),
    launches = launches,
    load = load,
)]
struct BlockMetal {
    m: usize,
    n: usize,
    k: usize,
    form: &'static dyn Stored,
    entry: &'static str,
}

fn accepts(k: &BlockMetal, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        _ => form.name() == "f32",
    }
}

fn metal_needs(b: &BlockMetal) -> Needs {
    let mma = uses_mma(b.m);
    Needs {
        lanes: 32,
        group_bytes: if mma { MMA_SMEM } else { 0 },
        group_threads: THREADS,
        tile: mma,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn load(b: &BlockMetal, _: usize) -> Option<Load> {
    let down = row_tiles(b.m);
    let cols = cols_per_tg(b.form.name(), b.m, b.n);
    let groups = b.n.div_ceil(cols) * down;
    let mut l = Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: b.k.div_ceil(256) as f64,
        // Weights streamed once per row tile plus activation reads.
        reads: vec![
            (b.form.bytes(b.n * b.k) * down) as f64,
            (b.m * b.k * 4) as f64,
        ],
        written: (4 * b.m * b.n) as f64,
        flight: 512.0,
        ..Load::default()
    };
    // Thirty-two lanes read contiguous 256-element stretches.
    l.trip.fresh = 1.0;
    Some(l)
}

fn launches(b: &BlockMetal, step: &Step) -> Vec<Launch> {
    let cols = cols_per_tg(b.form.name(), b.m, b.n);
    vec![Launch {
        args: vec![
            step.read(0),
            step.read(1),
            step.out(),
            Arg::count(b.m),
            Arg::count(b.n),
            Arg::count(b.k),
        ],
        shape: Shape::Grid {
            across: b.n.div_ceil(cols),
            down: row_tiles(b.m),
            threads: THREADS,
        },
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) || node.form.name() != "f32" {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    if tree.node(x).form.name() != "f32" {
        return None;
    }
    let form = tree.node(w).form;
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    if n == 0 || k == 0 || !k.is_multiple_of(32) {
        return None;
    }
    let m = tree.elements(x) / k;
    let ent = entry(form.name(), m, n)?;
    Some(vec![Draft::new(
        BlockMetal {
            m,
            n,
            k,
            form,
            entry: ent,
        },
        [x, w],
        m * n,
    )])
}

fn why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!("block_metal", propose, why = why, lacks = LACKS);
