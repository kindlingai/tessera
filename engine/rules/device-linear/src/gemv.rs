//! Streaming decode GEMV for BF16, MXFP8 and INT4 weights.

use super::mm::{MmDims, match_mm_add, match_mm_site};
use super::{
    BLOCK, THREADS, TrellisMma, TrellisReduce, WARPS, cuda_needs, cuda_source_with, dtr_blocks,
    dtr_needs, dtr_pre_tile, dtr_rule, dtr_tile, io_defs, prefill_defs, prefill_source_with,
    reduce_load_for, rocm_needs, rocm_source, trip, weight_x_accepts, x_accepts,
};
use tessera_graph::{Act, Op, Order, Stage};
use tessera_linear::Dense;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Compute, Draft, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T, Term, Tree,
};

tessera_lower::knob!("fast_gemv_bf16", "narrow_max_n", 1, 1024);
tessera_lower::knob!("fast_gemv_bf16", "narrow_min_k", 256, 131072);
tessera_lower::knob!("fast_gemv_bf16", "warps", 1, 32);
tessera_lower::knob!("fast_gemv_bf16", "min_blocks", 1, 16);
tessera_lower::knob!("fast_gemv_mxfp8", "min_blocks", 1, 16);
tessera_lower::knob!("fast_gemv_int4_g32", "min_blocks", 1, 16);

pub(super) fn pick_divisor(chunks: usize, want: usize, cap: usize) -> usize {
    for d in want..=cap {
        if chunks.is_multiple_of(d) {
            return d;
        }
    }
    for d in (1..want).rev() {
        if chunks.is_multiple_of(d) {
            return d;
        }
    }
    1
}

pub(super) fn r64_ks(m: usize, n: usize, chunks: usize) -> usize {
    let base = n.div_ceil(64) * m.div_ceil(64);
    if chunks < 4 || base >= 96 {
        return 1;
    }
    let cap = if base >= 32 { 4 } else { 8 }.min(chunks / 2);
    let want = 96usize.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

pub(super) fn has_other(mode: u32) -> bool {
    matches!(mode & 15, 1..=4 | 6..=8)
}

pub(super) fn has_bias2(mode: u32) -> bool {
    (mode & 15) == 8
}

pub(super) fn is_bias(mode: u32) -> bool {
    matches!(mode & 15, 2 | 6 | 7)
}

pub(super) fn is_bias_bf16(mode: u32) -> bool {
    (mode & 16) != 0
}

pub(super) fn epi_accepts(mode: u32, input: usize, other_at: usize, form: &dyn Stored) -> bool {
    let bias_slot = if is_bias(mode) {
        Some(other_at)
    } else if has_bias2(mode) {
        Some(other_at + 1)
    } else {
        None
    };
    if bias_slot == Some(input) && is_bias_bf16(mode) {
        return form.name() == "bf16";
    }
    form.name() == "f32"
}

/// Epilogue reads, with `o_elem` bytes per `other` element.
pub(super) fn push_epi_reads(reads: &mut Vec<f64>, mode: u32, [m, n]: [usize; 2], o_elem: usize) {
    let base = mode & 15;
    if matches!(base, 1 | 3 | 4 | 8) {
        reads.push((o_elem * m * n) as f64);
    }
    if matches!(base, 2 | 6 | 7) {
        let elem = if is_bias_bf16(mode) { 2 } else { o_elem };
        reads.push((elem * n) as f64);
    }
    if base == 8 {
        let elem = if is_bias_bf16(mode) { 2 } else { 4 };
        reads.push((elem * n) as f64);
    }
}

pub(super) struct MatmulBase {
    pub(super) x: T,
    pub(super) w: T,
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) x_bf16: bool,
    pub(super) y_bf16: bool,
    pub(super) form: &'static dyn Stored,
}

/// Whether an f32 or bf16 activation is bf16.
pub(super) fn plain_bf16(what: &str, form: &dyn Stored) -> Result<bool, String> {
    match form.name() {
        "f32" => Ok(false),
        "bf16" => Ok(true),
        name => Err(format!("{what} form {name} (requires f32 or bf16)")),
    }
}

/// Checks common 2D multi-row matmul preconditions.
pub(super) fn check_mm_base(tree: &Tree, mat: T) -> Option<Result<MatmulBase, String>> {
    let node = tree.node(mat);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let [n, k] = tree.node(w).shape[..] else {
        return None;
    };
    if n == 0 || k == 0 {
        return None;
    }
    let elems = tree.elements(x);
    if !elems.is_multiple_of(k) {
        return None;
    }
    let m = elems / k;
    if m < 2 {
        return None;
    }
    let y_bf16 = match plain_bf16("output", node.form) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    let x_bf16 = match plain_bf16("input", tree.node(x).form) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(MatmulBase {
        x,
        w,
        m,
        n,
        k,
        x_bf16,
        y_bf16,
        form: tree.node(w).form,
    }))
}

pub(super) fn match_cuda_site<S: MmDims>(
    tree: &Tree,
    at: T,
    pick: fn(&Tree, T) -> Option<S>,
) -> Option<(S, u32, Option<T>, Option<T>)> {
    let node = tree.node(at);
    if node.form.name() == "f32" {
        if let Term::Stage(Stage::Act(act @ (Act::GeluTanh | Act::Gelu))) = node.term {
            let add = node.args[0];
            if !tree.root(add)
                && let Some((site, m_add, bias)) = match_mm_add(tree, add, pick, true)
                && (m_add & 15) == 2
            {
                let base = if act == Act::GeluTanh { 6 } else { 7 };
                return Some((site, base | (m_add & 16), bias, None));
            }
        }
        if node.term == Term::Stage(Stage::Add)
            && let Some(&b) = node.args.get(1)
        {
            let a = node.args[0];
            for (inner, resid) in [(a, b), (b, a)] {
                if inner == resid || tree.root(inner) || tree.node(resid).form.name() != "f32" {
                    continue;
                }
                if let Some((site, m_add, bias)) = match_mm_add(tree, inner, pick, true)
                    && (m_add & 15) == 2
                {
                    let (m, n) = site.dims();
                    if tree.elements(resid) == m * n && tree.node(resid).shape == node.shape {
                        return Some((site, 8 | (m_add & 16), Some(resid), bias));
                    }
                }
            }
        }
    }
    if let Some((site, m_add, other)) = match_mm_add(tree, at, pick, true) {
        return Some((site, m_add, other, None));
    }
    let (site, mode, other) = match_mm_site(tree, at, pick)?;
    Some((site, mode, other, None))
}

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
    emits = gemv_emits,
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
        self.x_bf16, self.y_bf16,
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
    x_bf16: bool,
    y_bf16: bool,
    form: &'static dyn Stored,
}

fn gemv_accepts(k: &FastGemv, input: usize, form: &dyn Stored) -> bool {
    weight_x_accepts(input, k.x_bf16, k.form, form)
}

fn gemv_emits(k: &FastGemv, form: &dyn Stored) -> bool {
    x_accepts(k.y_bf16, form)
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
    if k.x_bf16 {
        defs.push(("X_BF16", 1));
    }
    if k.y_bf16 {
        defs.push(("Y_BF16", 1));
    }
    cuda_source_with(k.kind.entry(), defs)
}

fn gemv_load(k: &FastGemv, _: usize) -> Option<Load> {
    let groups = k.kind.groups(k.n, k.warps);
    let threads = k.warps * 32;
    let x_elem = if k.x_bf16 { 2 } else { 4 };
    let y_elem = if k.y_bf16 { 2 } else { 4 };
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
        reads: vec![k.form.bytes(k.n * k.k) as f64, (x_elem * k.k) as f64],
        written: (y_elem * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

pub(super) fn gemv_launch(
    shape: Shape,
    n: usize,
    k: usize,
    act: Option<(u32, f32, f32)>,
    step: &Step,
) -> Vec<Launch> {
    let mut args = vec![
        step.read(0),
        step.read(1),
        step.out(),
        Arg::count(n),
        Arg::count(k),
    ];
    if let Some((mode, a, b)) = act {
        args.extend([Arg::value(&mode), Arg::value(&a), Arg::value(&b)]);
    }
    vec![Launch { args, shape }]
}

fn gemv_launches(k: &FastGemv, step: &Step) -> Vec<Launch> {
    let act = matches!(k.kind, GemvKind::Bf16 | GemvKind::Bf16Narrow).then_some((k.mode, k.a, k.b));
    gemv_launch(k.shape().expect("a grid"), k.n, k.k, act, step)
}

pub(super) fn gemv_epilogue(tree: &Tree, mut at: T) -> Option<(T, u32, f32, f32)> {
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
    y_bf16: bool,
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
    if let Err(e) = plain_bf16("output", node.form) {
        return Some(Err(e));
    }
    if !tree.leaf(w) {
        return Some(Err("weight is not a direct leaf".into()));
    }
    let x_bf16 = match plain_bf16("input", tree.node(x).form) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
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
        x_bf16,
        y_bf16,
        form,
    };
    Some(Ok(vec![Draft::new(kernel, [x, w], n)]))
}

pub(super) type GemvCheck = fn(&Tree, T, u32, f32, f32, bool) -> Option<Result<Vec<Draft>, String>>;

pub(super) fn gemv_propose(tree: &Tree, at: T, check: GemvCheck) -> Option<Vec<Draft>> {
    let y_bf16 = plain_bf16("output", tree.node(at).form).ok()?;
    let (mat, mode, a, b) = gemv_epilogue(tree, at)?;
    check(tree, mat, mode, a, b, y_bf16)?.ok()
}

pub(super) fn propose_gemv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    gemv_propose(tree, at, check_gemv)
}

pub(super) fn why_gemv(tree: &Tree, at: T) -> Option<String> {
    let y_bf16 = tree.node(at).form.name() == "bf16";
    check_gemv(tree, at, 0, 1.0, 1.0, y_bf16)?.err()
}

/// Split-K trellis reduction with output rotation and epilogue.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.rocm { "rocm_trellis_reduce_epi" } else { "trellis_reduce_epi" },
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = cred_accepts,
    emits = cred_emits,
    needs = if self.rocm {
        rocm_needs(if (self.mode & 32) != 0 { 2048 } else { 0 }, THREADS, false)
    } else {
        dtr_needs(if (self.mode & 32) != 0 { 2048 } else { 0 })
    },
    order = Order::PerConfig,
    source = if self.rocm {
        rocm_source(
            super::rocm_trellis::rocm_trellis_source(),
            "rocm_dtr_reduce_epi",
        )
    } else {
        let base = prefill_defs("dtr_reduce_epi", (self.mode & 256) != 0);
        prefill_source_with("dtr_reduce_epi", io_defs(base, false, self.y_bf16, false))
    },
    launches = cred_launches,
    load = cred_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.ks, self.turned, self.mode, self.rocm, self.y_bf16,
    )),
    unproven,
)]
pub(super) struct CudaTrellisRed {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) ks: usize,
    pub(super) turned: bool,
    pub(super) mode: u32,
    pub(super) rocm: bool,
    pub(super) y_bf16: bool,
}

fn cred_emits(r: &CudaTrellisRed, form: &dyn Stored) -> bool {
    x_accepts(r.y_bf16, form)
}

fn cred_accepts(r: &CudaTrellisRed, input: usize, form: &dyn Stored) -> bool {
    let other_at = 1 + usize::from(r.turned);
    if input >= other_at {
        return epi_accepts(r.mode, input, other_at, form);
    }
    form.name() == "f32"
}

fn cred_load(r: &CudaTrellisRed, _: usize) -> Option<Load> {
    let mut load = reduce_load_for([r.m, r.n, r.ks], r.turned, r.y_bf16)?;
    push_epi_reads(&mut load.reads, r.mode, [r.m, r.n], 4);
    if (r.mode & 32) != 0 {
        load.reads.push((4 * r.n) as f64);
        let x_elem = if (r.mode & 256) != 0 { 1 } else { 2 };
        load.written = (x_elem * r.m * r.n) as f64;
    }
    Some(load)
}

fn cred_launches(r: &CudaTrellisRed, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: r.n / BLOCK,
        down: r.m.div_ceil(8),
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let other_at = 1 + usize::from(r.turned);
    let has_o = has_other(r.mode);
    let other = if has_o {
        step.read(other_at)
    } else {
        step.read(0)
    };
    let bias_at = other_at + usize::from(has_o);
    let bias = if has_bias2(r.mode) || (r.mode & 32) != 0 {
        step.read(bias_at)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        svh,
        other,
        bias,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::count(usize::from(r.turned)),
        Arg::value(&r.mode),
    ];
    vec![Launch { args, shape }]
}

fn cuda_small_ks(tree: &Tree, d: &Dense) -> usize {
    let (m, n, ibs) = (d.m, d.n, d.k / BLOCK);
    let rule = dtr_rule(d.turned());
    if let Some(want) = tree.knobs.get(rule, "splits") {
        let cap = ibs.clamp(1, 16);
        return pick_divisor(ibs, want.clamp(1, cap), cap);
    }
    let split_max_rows = tree.knobs.get(rule, "split_max_rows").unwrap_or(32);
    if m > split_max_rows || ibs <= 1 {
        return 1;
    }
    let large = tree.knobs.get(rule, "target_large");
    let target = if m <= 2 {
        tree.knobs.get(rule, "target_small").unwrap_or(288)
    } else if m <= 16 {
        large.unwrap_or(128)
    } else {
        large.unwrap_or(96)
    };
    let cap = if m <= 16 { 16 } else { 8 }.min(ibs);
    let want = target.div_ceil(n / BLOCK).clamp(1, cap);
    pick_divisor(ibs, want, cap)
}

pub(super) fn cuda_pre_ks(tree: &Tree, d: &Dense) -> usize {
    let (m, n, ibs) = (d.m, d.n, d.k / BLOCK);
    if ibs < 8 {
        return 1;
    }
    if let Some(want) = tree.knobs.get("trellis_pre_sk", "splits") {
        return pick_divisor(ibs, want, want.min(ibs / 2));
    }
    let tile = dtr_pre_tile(tree, d.turned(), m, n);
    let base = (n / BLOCK) * m.div_ceil(tile);
    if base >= 96 {
        return 1;
    }
    let max_k = if base <= 16 {
        8
    } else if tile == 128 {
        4
    } else {
        6
    };
    let cap = max_k.min(ibs / 2);
    pick_divisor(ibs, 96usize.div_ceil(base).clamp(1, cap), cap)
}

tessera_lower::knob!("trellis_pre_sk", "splits", 1, 16);

impl MmDims for Dense {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

pub(super) fn dense_pick(tree: &Tree, at: T) -> Option<Dense> {
    Dense::find(tree, at, BLOCK)?.ok()
}

fn small_drafts(tree: &Tree, d: &Dense, rocm: bool) -> Result<Vec<Draft>, String> {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let (ks, tile) = if rocm {
        let ks = super::rocm_trellis::dtr_ks_for(m, n, k, 4, [80, 40]);
        (ks, super::rocm_trellis::small_tile(m))
    } else {
        let tile = match tree.knobs.get(dtr_rule(turned), "tile_rows") {
            Some(v @ (8 | 16 | 32)) => v,
            Some(v) => return Err(format!("tile_rows={v} (requires 8, 16, or 32)")),
            None => dtr_tile(m),
        };
        (cuda_small_ks(tree, d), tile)
    };
    let blocks = |knob| (!rocm).then(|| dtr_blocks(tree, turned, knob)).flatten();
    let mma = TrellisMma {
        m,
        n,
        k,
        ks,
        tile,
        min_blocks: blocks("min_blocks"),
        bits: d.bits,
        book: d.book,
        turned,
        rocm,
        y_bf16: ks == 1 && d.y_bf16,
        form: d.form,
        x_bf16: d.x_bf16,
    };
    let red = TrellisReduce {
        m,
        n,
        ks,
        min_blocks: blocks("reduce_min_blocks"),
        turned,
        rocm,
        y_bf16: d.y_bf16,
    };
    Ok(d.drafts(mma, ks, red))
}

pub(super) fn eval_trellis_for(
    tree: &Tree,
    at: T,
    rocm: bool,
) -> Option<Result<Vec<Draft>, String>> {
    let (d, mode, other, bias2) = match match_cuda_site(tree, at, dense_pick) {
        Some(hit) => hit,
        None => match Dense::find_with(tree, at, BLOCK, !rocm)? {
            Ok(d) => (d, 0, None, None),
            Err(e) => return Some(Err(e)),
        },
    };
    let pre_min_rows = match rocm {
        true => None,
        false => tree.knobs.get(dtr_rule(d.turned()), "prefill_min_rows"),
    };
    if d.m < pre_min_rows.unwrap_or(33) {
        return (mode == 0).then(|| small_drafts(tree, &d, rocm));
    }
    let mx8 = !rocm && tree.compute_at(at) == Compute::Bf16;
    Some(Ok(super::cuda_fuse::plan_trellis_pre(
        tree, at, d, mode, other, bias2, mx8, rocm,
    )))
}
