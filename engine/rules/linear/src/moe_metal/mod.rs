//! Native Metal coalesced matmul for picked MoE experts.

use tessera_graph::Order;
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::{Matmul, Picked};
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Step, T, Tree};

const THREADS: usize = 128;
const COLS_PER_TG: usize = 16;
const BLOCK: usize = 32;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("moe_metal.metal");
/// Low default guess so Metal prefers coalesced MoE matmul.
const GUESS: f64 = 1.0e-12;

const ENTRIES: &[(&str, &str, bool)] = &[
    ("f32", "pm_f32", false),
    ("bf16", "pm_bf16", false),
    ("f16", "pm_f16", false),
    ("fp8", "pm_fp8", false),
    ("mxfp8", "pm_mxfp8", true),
    ("q8_0", "pm_q8_0", true),
    ("int4_g32", "pm_int4_g32", true),
    ("q4_0", "pm_q4_0", true),
];

fn lookup(form: &str) -> Option<(&'static str, bool)> {
    ENTRIES
        .iter()
        .find(|(name, _, _)| *name == form)
        .map(|(_, entry, block)| (*entry, *block))
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.form.suffixed("moe_metal"),
    sizes = [self.rows, self.n, self.k],
    work = self.rows * self.n * self.k,
    guess = GUESS,
    accepts = moe_accepts,
    emits = crate::f32_only,
    needs = moe_needs(),
    order = Order::PerConfig,
    source = tessera_lower::Source::metal(SOURCE, self.entry),
    launches = moe_launches,
    load = moe_load,
)]
struct MetalMoe {
    rows: usize,
    n: usize,
    k: usize,
    count: usize,
    stride: usize,
    offset: usize,
    repeat: usize,
    first: usize,
    form: &'static dyn Stored,
    entry: &'static str,
}

fn moe_accepts(k: &MetalMoe, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == "f32",
        1 => form.name() == k.form.name(),
        2 => !form.narrow(),
        _ => false,
    }
}

fn moe_needs() -> Needs {
    Needs {
        lanes: 32,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn moe_load(k: &MetalMoe, _: usize) -> Option<Load> {
    let groups = k.n.div_ceil(COLS_PER_TG) * k.rows;
    let mut l = Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: k.k.div_ceil(128) as f64,
        reads: vec![
            k.form.bytes(k.rows * k.n * k.k) as f64,
            (4 * k.rows * k.k) as f64,
        ],
        written: (4 * k.rows * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    };
    // Thirty-two lanes read contiguous stretches along `k`.
    l.trip.fresh = 1.0;
    Some(l)
}

fn moe_launches(k: &MetalMoe, step: &Step) -> Vec<Launch> {
    vec![Launch {
        args: vec![
            step.read(0),
            step.read(1),
            step.read(2),
            step.out(),
            Arg::count(k.rows),
            Arg::count(k.n),
            Arg::count(k.k),
            Arg::count(k.count),
            Arg::count(k.stride),
            Arg::count(k.offset),
            Arg::count(k.repeat),
            Arg::count(k.first),
        ],
        shape: Shape::Grid {
            across: k.n.div_ceil(COLS_PER_TG),
            down: k.rows,
            threads: THREADS,
        },
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: Matmul = tree.site(at)?;
    let Picked {
        count,
        stride,
        offset,
        repeat,
        first,
        ..
    } = site.picked?;
    if site.m == 0 || site.n == 0 || site.k == 0 {
        return None;
    }
    if tree.node(site.x).form.name() != "f32" || tree.node(at).form.name() != "f32" {
        return None;
    }
    let (entry, block) = lookup(site.form.name())?;
    if block && !site.k.is_multiple_of(BLOCK) {
        return None;
    }
    let kernel = MetalMoe {
        rows: site.m,
        n: site.n,
        k: site.k,
        count,
        stride,
        offset,
        repeat,
        first: first.unwrap_or(0),
        form: site.form,
        entry,
    };
    Some(vec![Draft::new(kernel, site.reads(), site.m * site.n)])
}

fn why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!("moe_metal", propose, why = why, lacks = LACKS);
