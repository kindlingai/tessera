//! Streaming decode GEMV for BF16, MXFP8 and INT4 weights.

use super::{THREADS, WARPS, cuda_needs, cuda_source_entry, f32_only, weight_accepts};
use tessera_graph::{Act, Op, Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemvKind {
    Bf16,
    Bf16Narrow,
    Mxfp8,
    Int4,
}

impl GemvKind {
    fn entry(self) -> &'static str {
        match self {
            GemvKind::Bf16 => "tessera_gemv_bf16",
            GemvKind::Bf16Narrow => "tessera_gemv_bf16_narrow",
            GemvKind::Mxfp8 => "tessera_gemv_mxfp8",
            GemvKind::Int4 => "tessera_gemv_int4",
        }
    }

    fn name(self) -> &'static str {
        match self {
            GemvKind::Bf16 | GemvKind::Bf16Narrow => "fast_gemv_bf16",
            GemvKind::Mxfp8 => "fast_gemv_mxfp8",
            GemvKind::Int4 => "fast_gemv_int4_g32",
        }
    }

    fn smem(self) -> usize {
        match self {
            GemvKind::Bf16 => 0,
            GemvKind::Bf16Narrow => WARPS * 4,
            GemvKind::Mxfp8 => 512 * 4 + WARPS * 268 * 4,
            GemvKind::Int4 => 512 * 4 + WARPS * 296 * 4,
        }
    }

    fn groups(self, n: usize) -> usize {
        match self {
            GemvKind::Int4 => (n / 2).div_ceil(WARPS),
            GemvKind::Bf16Narrow => n,
            GemvKind::Bf16 | GemvKind::Mxfp8 => n.div_ceil(WARPS),
        }
    }
}

/// Streaming decode GEMV for BF16, MXFP8, and INT4_G32 weights.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.name(),
    sizes = [1, self.n, self.k],
    work = self.n * self.k,
    accepts = gemv_accepts,
    emits = f32_only,
    needs = gemv_needs(self),
    order = Order::PerConfig,
    source = gemv_source(self),
    launches = gemv_launches,
    load = gemv_load,
    unproven,
)]
struct FastGemv {
    n: usize,
    k: usize,
    mode: u32,
    a: f32,
    b: f32,
    kind: GemvKind,
    form: &'static dyn Stored,
}

fn gemv_accepts(k: &FastGemv, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn gemv_needs(k: &FastGemv) -> Needs {
    cuda_needs(k.kind.smem())
}

fn gemv_source(k: &FastGemv) -> Source {
    cuda_source_entry(k.kind.entry())
}

fn gemv_load(k: &FastGemv, _: usize) -> Option<Load> {
    let groups = k.kind.groups(k.n);
    let step = match k.kind {
        GemvKind::Bf16 => 256.0,
        GemvKind::Bf16Narrow => 2048.0,
        GemvKind::Mxfp8 | GemvKind::Int4 => 1024.0,
    };
    let mut load = Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: k.k as f64 / step,
        reads: vec![k.form.bytes(k.n * k.k) as f64, (4 * k.k) as f64],
        written: (4 * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn gemv_launches(k: &FastGemv, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.kind.groups(k.n),
        down: 1,
        threads: THREADS,
    };
    let mut args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(k.n),
        Arg::count(k.k),
    ];
    if matches!(k.kind, GemvKind::Bf16 | GemvKind::Bf16Narrow) {
        args.extend([Arg::value(&k.mode), Arg::value(&k.a), Arg::value(&k.b)]);
    }
    vec![Launch { args, shape }]
}

fn gemv_epilogue(tree: &Tree, mut at: T) -> Option<(T, u32, f32, f32)> {
    let (mut mode, mut a, mut b) = (0u32, 1.0f32, 1.0f32);
    if let Term::Stage(Stage::Scale(s)) = tree.node(at).term {
        let prev = tree.node(at).args[0];
        if !tree.root(prev) && tree.node(prev).term == Term::Stage(Stage::Act(Act::Sigmoid)) {
            b = s;
            at = prev;
        }
    }
    match tree.node(at).term {
        Term::Stage(Stage::Act(Act::Silu)) if b == 1.0 => (mode, at) = (1, tree.node(at).args[0]),
        Term::Stage(Stage::Act(Act::Sigmoid)) => (mode, at) = (2, tree.node(at).args[0]),
        _ if b != 1.0 => return None,
        _ => {}
    }
    if mode != 0
        && !tree.root(at)
        && let Term::Stage(Stage::Scale(s)) = tree.node(at).term
    {
        (a, at) = (s, tree.node(at).args[0]);
    }
    (mode == 0 || !tree.root(at)).then_some((at, mode, a, b))
}

pub(super) fn propose_gemv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    if tree.node(at).form.name() != "f32" {
        return None;
    }
    let (mat, mode, a, b) = gemv_epilogue(tree, at)?;
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) || node.form.name() != "f32" {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    if !tree.leaf(w) || tree.node(x).form.name() != "f32" {
        return None;
    }
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    if tree.elements(x) != k || n == 0 {
        return None;
    }
    let form = tree.node(w).form;
    let kind = match form.name() {
        "bf16" if k.is_multiple_of(8) && n <= 64 && k >= 1024 => GemvKind::Bf16Narrow,
        "bf16" if k.is_multiple_of(8) => GemvKind::Bf16,
        "mxfp8" if mode == 0 && k.is_multiple_of(1024) => GemvKind::Mxfp8,
        "int4_g32" if mode == 0 && k.is_multiple_of(1024) && n.is_multiple_of(2) => GemvKind::Int4,
        _ => return None,
    };
    let kernel = FastGemv {
        n,
        k,
        mode,
        a,
        b,
        kind,
        form,
    };
    Some(vec![Draft::new(kernel, [x, w], n)])
}
