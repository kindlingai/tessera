//! Native Metal block-quantized matmul decode for small batches.

use tessera_graph::{Op, Order};
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Step, T, Term, Tree};

const THREADS: usize = 128;
const COLS_PER_TG: usize = 16;
const TILE_ROWS: usize = 4;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("block_metal.metal");
const GUESS: f64 = 1.0e-13;

const ENTRIES: &[(&str, [&str; 4])] = &[
    (
        "mxfp8",
        ["bm_mxfp8_m1", "bm_mxfp8_m2", "bm_mxfp8_m3", "bm_mxfp8_m4"],
    ),
    (
        "q8_0",
        ["bm_q8_0_m1", "bm_q8_0_m2", "bm_q8_0_m3", "bm_q8_0_m4"],
    ),
    (
        "int4_g32",
        [
            "bm_int4_g32_m1",
            "bm_int4_g32_m2",
            "bm_int4_g32_m3",
            "bm_int4_g32_m4",
        ],
    ),
    (
        "q4_0",
        ["bm_q4_0_m1", "bm_q4_0_m2", "bm_q4_0_m3", "bm_q4_0_m4"],
    ),
];

fn entry(form: &str, m: usize) -> Option<&'static str> {
    let (_, names) = ENTRIES.iter().find(|(name, _)| *name == form)?;
    let idx = m.min(TILE_ROWS).checked_sub(1)?;
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
    needs = metal_needs(),
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

fn metal_needs() -> Needs {
    Needs {
        lanes: 32,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn load(b: &BlockMetal, _: usize) -> Option<Load> {
    let down = b.m.div_ceil(TILE_ROWS);
    let groups = b.n.div_ceil(COLS_PER_TG) * down;
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
            across: b.n.div_ceil(COLS_PER_TG),
            down: b.m.div_ceil(TILE_ROWS),
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
    let ent = entry(form.name(), m)?;
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
