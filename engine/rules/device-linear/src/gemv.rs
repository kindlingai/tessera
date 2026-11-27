//! Streaming decode GEMV for BF16, MXFP8 and INT4 weights.

use super::{WARPS, cuda_needs, cuda_source_with, f32_only, trip, weight_accepts};
use tessera_graph::{Act, Op, Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Draft, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T, Term, Tree,
};

tessera_lower::knob!("fast_gemv_bf16", "narrow_max_n", 1, 1024);
tessera_lower::knob!("fast_gemv_bf16", "narrow_min_k", 256, 131072);
tessera_lower::knob!("fast_gemv_bf16", "warps", 1, 32);
tessera_lower::knob!("fast_gemv_bf16", "min_blocks", 1, 16);
tessera_lower::knob!("fast_gemv_mxfp8", "min_blocks", 1, 16);
tessera_lower::knob!("fast_gemv_int4_g32", "min_blocks", 1, 16);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

    fn blk_def(self) -> &'static str {
        match self {
            GemvKind::Bf16 | GemvKind::Bf16Narrow => "GEMV_BF16_BLK",
            GemvKind::Mxfp8 => "GEMV_MXFP8_BLK",
            GemvKind::Int4 => "GEMV_INT4_BLK",
        }
    }

    fn smem(self, warps: usize) -> usize {
        match self {
            GemvKind::Bf16 => 0,
            GemvKind::Bf16Narrow => warps * 4,
            GemvKind::Mxfp8 => 512 * 4 + warps * 268 * 4,
            GemvKind::Int4 => 512 * 4 + warps * 296 * 4,
        }
    }

    /// One lane's trip: 8 bf16 weights, or a staged chunk.
    fn trip(self) -> Ops {
        match self {
            GemvKind::Bf16 | GemvKind::Bf16Narrow => trip(8, 8, 3, 48),
            GemvKind::Mxfp8 => Ops {
                shared: 24.0,
                e4m3: 32.0,
                ..trip(40, 48, 3, 48)
            },
            GemvKind::Int4 => Ops {
                shared: 40.0,
                ..trip(80, 160, 3, 48)
            },
        }
    }

    fn groups(self, n: usize, warps: usize) -> usize {
        match self {
            GemvKind::Int4 => (n / 2).div_ceil(warps),
            GemvKind::Bf16Narrow => n,
            GemvKind::Bf16 | GemvKind::Mxfp8 => n.div_ceil(warps),
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
    shape = Shape::Grid {
        across: self.kind.groups(self.n, self.warps),
        down: 1,
        threads: self.warps * 32,
    },
    load = gemv_load,
    load_key = tessera_lower::load_key((
        self.n, self.k, self.mode, self.a.to_bits(), self.b.to_bits(),
        self.warps, self.custom_warps, self.min_blocks, self.kind, self.form.name(),
    )),
    unproven,
)]
struct FastGemv {
    n: usize,
    k: usize,
    mode: u32,
    a: f32,
    b: f32,
    warps: usize,
    custom_warps: bool,
    min_blocks: Option<u32>,
    kind: GemvKind,
    form: &'static dyn Stored,
}

fn gemv_accepts(k: &FastGemv, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
}

fn gemv_needs(k: &FastGemv) -> Needs {
    Needs {
        group_threads: k.warps * 32,
        ..cuda_needs(k.kind.smem(k.warps))
    }
}

fn gemv_source(k: &FastGemv) -> Source {
    let mut defs = Vec::new();
    if k.custom_warps {
        defs.push(("GEMV_BF16_WARPS", k.warps));
    }
    if let Some(mb) = k.min_blocks {
        defs.push((k.kind.blk_def(), mb as usize));
    }
    cuda_source_with(k.kind.entry(), defs)
}

fn gemv_load(k: &FastGemv, _: usize) -> Option<Load> {
    let groups = k.kind.groups(k.n, k.warps);
    let threads = k.warps * 32;
    let step = match k.kind {
        GemvKind::Bf16 => 256.0,
        GemvKind::Bf16Narrow => (k.warps * 256) as f64,
        GemvKind::Mxfp8 | GemvKind::Int4 => 1024.0,
    };
    Some(Load {
        threads: groups * threads,
        group: threads,
        trips: k.k as f64 / step,
        trip: k.kind.trip(),
        reads: vec![k.form.bytes(k.n * k.k) as f64, (4 * k.k) as f64],
        written: (4 * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn gemv_launches(k: &FastGemv, step: &Step) -> Vec<Launch> {
    let shape = k.shape().expect("a grid");
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

fn check_gemv(
    tree: &Tree,
    mat: T,
    mode: u32,
    a: f32,
    b: f32,
) -> Option<Result<Vec<Draft>, String>> {
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    if tree.elements(x) != k || n == 0 {
        return None;
    }
    if node.form.name() != "f32" {
        return Some(Err(format!(
            "output form {} (requires f32)",
            node.form.name()
        )));
    }
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    if tree.node(x).form.name() != "f32" {
        return Some(Err(format!(
            "input form {} (requires f32)",
            tree.node(x).form.name()
        )));
    }
    let form = tree.node(w).form;
    let narrow_max_n = tree
        .knobs
        .get("fast_gemv_bf16", "narrow_max_n")
        .unwrap_or(64);
    let narrow_min_k = tree
        .knobs
        .get("fast_gemv_bf16", "narrow_min_k")
        .unwrap_or(1024);
    let kind = match form.name() {
        "bf16" if k.is_multiple_of(8) && n <= narrow_max_n && k >= narrow_min_k => {
            GemvKind::Bf16Narrow
        }
        "bf16" if k.is_multiple_of(256) => GemvKind::Bf16,
        // A part trip leaves warp lanes idle: lanes is faster.
        "bf16" if k.is_multiple_of(8) => return None,
        "bf16" => return Some(Err(format!("k={k} is not a multiple of 8 for bf16"))),
        "mxfp8" if mode == 0 && k.is_multiple_of(1024) => GemvKind::Mxfp8,
        "mxfp8" if mode != 0 => return None,
        "mxfp8" => return Some(Err(format!("k={k} is not a multiple of 1024 for mxfp8"))),
        "int4_g32" if mode == 0 && k.is_multiple_of(1024) && n.is_multiple_of(2) => GemvKind::Int4,
        "int4_g32" if mode != 0 => return None,
        "int4_g32" if !k.is_multiple_of(1024) => {
            return Some(Err(format!("k={k} is not a multiple of 1024 for int4_g32")));
        }
        "int4_g32" => return Some(Err(format!("n={n} is not even for int4_g32"))),
        other => {
            return Some(Err(format!(
                "weight form {other} (supports bf16, mxfp8, int4_g32)"
            )));
        }
    };
    let (warps, custom_warps) = match (kind, tree.knobs.get("fast_gemv_bf16", "warps")) {
        (GemvKind::Bf16 | GemvKind::Bf16Narrow, Some(w)) => (w, true),
        _ => (WARPS, false),
    };
    let min_blocks = tree.knobs.get(kind.name(), "min_blocks").map(|v| v as u32);
    let kernel = FastGemv {
        n,
        k,
        mode,
        a,
        b,
        warps,
        custom_warps,
        min_blocks,
        kind,
        form,
    };
    Some(Ok(vec![Draft::new(kernel, [x, w], n)]))
}

pub(super) fn propose_gemv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    if tree.node(at).form.name() != "f32" {
        return None;
    }
    let (mat, mode, a, b) = gemv_epilogue(tree, at)?;
    check_gemv(tree, mat, mode, a, b)?.ok()
}

pub(super) fn why_gemv(tree: &Tree, at: T) -> Option<String> {
    check_gemv(tree, at, 0, 1.0, 1.0)?.err()
}
