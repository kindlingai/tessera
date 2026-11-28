//! Native Metal decode and simdgroup_matrix prefill kernels for stacked matmuls.

use super::blocks;
use tessera_graph::{Op, Order};
use tessera_lower::Backend;
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree,
};

const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

fn act_fmt(form: &dyn Stored) -> Option<usize> {
    tessera_lower::form::metal_code(form, false)
}

fn weight_fmt(form: &dyn Stored) -> Option<usize> {
    tessera_lower::form::metal_code(form, true)
}

#[derive(Clone, Copy)]
struct Entry {
    name: &'static str,
    threads: usize,
    smem: usize,
    tile_m: usize,
    tile_n: usize,
    tile: bool,
}

const ENTRIES: [[&str; 4]; 6] = [
    [
        "st_dec_nt_f32",
        "st_dec_nt_bf16",
        "st_dec_nt_f16",
        "st_dec_nt_fp8",
    ],
    [
        "st_dec_t_f32",
        "st_dec_t_bf16",
        "st_dec_t_f16",
        "st_dec_t_fp8",
    ],
    [
        "st_pf16_nt_f32",
        "st_pf16_nt_bf16",
        "st_pf16_nt_f16",
        "st_pf16_nt_fp8",
    ],
    [
        "st_pf16_t_f32",
        "st_pf16_t_bf16",
        "st_pf16_t_f16",
        "st_pf16_t_fp8",
    ],
    [
        "st_pf32_nt_f32",
        "st_pf32_nt_bf16",
        "st_pf32_nt_f16",
        "st_pf32_nt_fp8",
    ],
    [
        "st_pf32_t_f32",
        "st_pf32_t_bf16",
        "st_pf32_t_f16",
        "st_pf32_t_fp8",
    ],
];

fn pick_entry(k: &MetalStacked) -> Entry {
    let fi = weight_fmt(k.w_form).expect("verified weight form");
    let sw = usize::from(k.swapped);
    if k.rows <= 4 {
        return Entry {
            name: ENTRIES[sw][fi],
            threads: 128,
            smem: if k.swapped { 2_048 } else { 0 },
            tile_m: k.rows,
            tile_n: if k.swapped { 256 } else { 8 },
            tile: false,
        };
    }
    let big = k.rows > 16;
    let family = if big { 4 } else { 2 } + sw;
    Entry {
        name: ENTRIES[family][fi],
        threads: if big { 256 } else { 128 },
        smem: if big { 13_824 } else { 6_912 },
        tile_m: if big { 32 } else { 16 },
        tile_n: if big { 64 } else { 32 },
        tile: true,
    }
}

/// Native Metal decode and MMA kernel for stacked matmuls.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.w_form.suffixed(if self.swapped {
        "stacked_metal_t"
    } else {
        "stacked_metal"
    }),
    order = Order::PerConfig,
    sizes = [self.batches, self.rows, self.n, self.k],
    work = self.batches * self.rows * self.n * self.k,
    needs = metal_needs(self),
    accepts = metal_accepts,
    emits = metal_emits,
    source = metal_source(self),
    launches = metal_launches,
    load = metal_load,
    unproven,
)]
struct MetalStacked {
    batches: usize,
    rows: usize,
    n: usize,
    k: usize,
    block: usize,
    first: usize,
    step: usize,
    swapped: bool,
    x_form: &'static dyn Stored,
    w_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
}

fn metal_needs(k: &MetalStacked) -> Needs {
    let e = pick_entry(k);
    Needs {
        lanes: 32,
        group_bytes: e.smem,
        group_threads: e.threads,
        tile: e.tile,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn metal_accepts(k: &MetalStacked, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == k.w_form.name(),
        _ => false,
    }
}

fn metal_emits(k: &MetalStacked, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn metal_source(k: &MetalStacked) -> Source {
    let e = pick_entry(k);
    Source::Fixed {
        text: ByBackend([
            tessera_lower::metal_source!(include_str!("metal.metal")),
            "",
            "",
        ]),
        entry: e.name,
    }
}

fn metal_load(k: &MetalStacked, _: usize) -> Option<Load> {
    let e = pick_entry(k);
    let r_tiles = k.rows.div_ceil(e.tile_m);
    let n_tiles = k.n.div_ceil(e.tile_n);
    let groups = k.batches * r_tiles * n_tiles;
    Some(Load {
        threads: groups * e.threads,
        group: e.threads,
        trips: k.k.div_ceil(32) as f64,
        reads: vec![
            k.x_form.bytes(k.batches * k.rows * k.k) as f64,
            k.w_form.bytes(k.batches * k.n * k.k) as f64,
        ],
        written: k.out_form.bytes(k.batches * k.rows * k.n) as f64,
        flight: 512.0,
        group_bytes: e.smem as f64,
        ..Load::default()
    })
}

fn metal_launches(k: &MetalStacked, step: &Step) -> Vec<Launch> {
    let e = pick_entry(k);
    let r_tiles = if k.rows <= 4 {
        1
    } else {
        k.rows.div_ceil(e.tile_m)
    };
    let n_tiles = k.n.div_ceil(e.tile_n);
    let x_fmt = act_fmt(k.x_form).expect("verified x_form");
    let o_fmt = act_fmt(k.out_form).expect("verified out_form");
    let flags = x_fmt | (o_fmt << 2);
    let args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(k.batches),
        Arg::count(k.rows),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::count(k.block),
        Arg::count(k.first),
        Arg::count(k.step),
        Arg::count(flags),
    ];
    let shape = Shape::Grid {
        across: n_tiles,
        down: k.batches * r_tiles,
        threads: e.threads,
    };
    vec![Launch { args, shape }]
}

/// Returns a Metal draft when the stacked site fits.
pub(super) fn draft(tree: &Tree, at: T) -> Option<Draft> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let b = blocks(tree, w)?;
    let (&[batches, n, k], xs) = (&tree.node(w).shape[..], &tree.node(x).shape[..]) else {
        return None;
    };
    if xs.len() != 3
        || xs[0] != batches
        || xs[2] != k
        || batches == 0
        || xs[1] == 0
        || n == 0
        || k == 0
    {
        return None;
    }
    let x_form = plain(tree.node(x).form);
    let w_form = plain(tree.node(b.leaf).form);
    let out_form = writable(node.form);
    if act_fmt(x_form).is_none() || weight_fmt(w_form).is_none() || act_fmt(out_form).is_none() {
        return None;
    }
    let kernel = MetalStacked {
        batches,
        rows: xs[1],
        n,
        k,
        block: b.rows * b.width,
        first: b.first * b.width,
        step: b.width,
        swapped: b.swapped,
        x_form,
        w_form,
        out_form,
    };
    Some(Draft::new(kernel, [x, b.leaf], tree.elements(at)).choice(1))
}
