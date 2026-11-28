//! Native Metal `simdgroup_matrix` implicit-GEMM 3x3 convolution kernel.

use tessera_graph::{Op, Order};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::sites::Near;
use tessera_lower::{
    Arg, Backend, ByBackend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree,
};

const TAPS: usize = 9;
const TILE: usize = 32;
const THREADS: usize = 128;
const SMEM: usize = 2 * TILE * 36 * 4;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

fn act_fmt(form: &dyn Stored) -> Option<usize> {
    tessera_lower::form::metal_code(form, false)
}

fn weight_fmt(form: &dyn Stored) -> Option<usize> {
    tessera_lower::form::metal_code(form, true)
}

/// Implicit-GEMM 3x3 convolution over `[Seqs, pixels, cin]` using Metal MMA.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "neighbours_metal",
    order = Order::PerConfig,
    sizes = [self.rows, self.pixels, self.cin, self.cout],
    work = self.rows * self.cout * TAPS * self.cin,
    needs = metal_needs(),
    accepts = metal_accepts,
    emits = metal_emits,
    source = metal_source(self),
    launches = metal_launches,
    load = metal_load,
    unproven,
)]
struct NeighboursMetal {
    rows: usize,
    pixels: usize,
    cin: usize,
    cout: usize,
    x_form: &'static dyn Stored,
    w_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
}

fn metal_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: SMEM,
        group_threads: THREADS,
        tile: true,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn metal_accepts(k: &NeighboursMetal, input: usize, form: &dyn Stored) -> bool {
    if input == 0 {
        form.name() == k.x_form.name()
    } else if input == 2 {
        form.name() == k.w_form.name()
    } else {
        !form.narrow()
    }
}

fn metal_emits(k: &NeighboursMetal, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn metal_source(k: &NeighboursMetal) -> Source {
    let entry = if k.cin.is_multiple_of(TILE) {
        "nb_mma_aligned"
    } else {
        "nb_mma_general"
    };
    Source::Fixed {
        text: ByBackend([
            tessera_lower::metal_source!(include_str!("metal.metal")),
            "",
            "",
        ]),
        entry,
    }
}

fn metal_load(k: &NeighboursMetal, _: usize) -> Option<Load> {
    let across = k.cout.div_ceil(TILE);
    let down = k.rows.div_ceil(TILE);
    let groups = across * down;
    let k_total = TAPS * k.cin;
    let x_bytes = k.x_form.bytes(k.rows * k.cin) as f64;
    let w_bytes = k.w_form.bytes(k.cout * k_total) as f64;
    let out_bytes = k.out_form.bytes(k.rows * k.cout) as f64;
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: k_total.div_ceil(TILE) as f64,
        reads: vec![x_bytes, w_bytes],
        written: out_bytes,
        flight: 512.0,
        group_bytes: SMEM as f64,
        ..Load::default()
    })
}

fn metal_launches(k: &NeighboursMetal, step: &Step) -> Vec<Launch> {
    let x_fmt = act_fmt(k.x_form).expect("verified x_form");
    let w_fmt = weight_fmt(k.w_form).expect("verified w_form");
    let o_fmt = act_fmt(k.out_form).expect("verified out_form");
    let flags = x_fmt | (w_fmt << 2) | (o_fmt << 4);
    let shape = Shape::Grid {
        across: k.cout.div_ceil(TILE),
        down: k.rows.div_ceil(TILE),
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.out(),
        Arg::count(k.rows),
        Arg::count(k.pixels),
        Arg::count(k.cin),
        Arg::count(k.cout),
        Arg::count(flags),
    ];
    vec![Launch { args, shape }]
}

pub(super) fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (flat, w) = (node.args[0], node.args[1]);
    let near: Near = tree.site(flat)?;
    if near.taps != TAPS || near.cin == 0 || near.pixels == 0 {
        return None;
    }
    let x_form = plain(tree.node(near.src).form);
    let w_form = plain(tree.node(w).form);
    let out_form = writable(node.form);
    act_fmt(x_form)?;
    weight_fmt(w_form)?;
    act_fmt(out_form)?;
    let len = tree.elements(at);
    let rows = tree.elements(near.src) / near.cin;
    let cout = len / rows.max(1);
    if rows == 0 || cout == 0 {
        return None;
    }
    let k = NeighboursMetal {
        rows,
        pixels: near.pixels,
        cin: near.cin,
        cout,
        x_form,
        w_form,
        out_form,
    };
    Some(vec![Draft::new(k, [near.src, near.ids, w], len).choice(1)])
}
