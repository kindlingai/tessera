//! RDNA 3/3.5 ROCm GEMV and WMMA GEMM rules.

use super::gemv::{
    MatmulBase, check_mm_base, epi_accepts, gemv_launch, gemv_propose, has_bias2, has_other,
    match_cuda_site, push_epi_reads,
};
use super::mm::{MmDims, mm_draft, pick_divisor};
use super::{THREADS, WARPS, f32_only, rocm_needs, rocm_rule, rocm_source, weight_x_accepts};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
use tessera_graph::{Op, Order};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Ops, Shape, Step, T, Term, Tree};

const ROCM_LINEAR: &str = include_str!("rocm_linear.cu");
const MM_THREADS: usize = 128;

pub(crate) fn rocm_linear_source() -> &'static str {
    static ROCM: OnceLock<&'static str> = OnceLock::new();
    ROCM.get_or_init(|| flatten_for(Backend::Rocm, "rocm_linear.kernel", ROCM_LINEAR))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RocmGemvKind {
    Bf16,
    Bf16Narrow,
    Int4,
    Mxfp8,
}

impl RocmGemvKind {
    fn entry(self) -> &'static str {
        match self {
            RocmGemvKind::Bf16 => "rocm_gemv_bf16",
            RocmGemvKind::Bf16Narrow => "rocm_gemv_bf16_narrow",
            RocmGemvKind::Int4 => "rocm_gemv_int4",
            RocmGemvKind::Mxfp8 => "rocm_gemv_mxfp8",
        }
    }

    fn name(self) -> &'static str {
        match self {
            RocmGemvKind::Bf16 | RocmGemvKind::Bf16Narrow => "rocm_gemv_bf16",
            RocmGemvKind::Int4 => "rocm_gemv_int4_g32",
            RocmGemvKind::Mxfp8 => "rocm_gemv_mxfp8",
        }
    }

    fn smem(self) -> usize {
        match self {
            RocmGemvKind::Bf16 => 0,
            RocmGemvKind::Bf16Narrow => WARPS * 4,
            RocmGemvKind::Int4 => 2 * 144 * 4,
            RocmGemvKind::Mxfp8 => (2 * 264 + 256) * 4,
        }
    }

    fn threads(self) -> usize {
        match self {
            RocmGemvKind::Bf16Narrow => THREADS,
            _ => 64,
        }
    }

    fn groups(self, n: usize) -> usize {
        match self {
            RocmGemvKind::Bf16Narrow => n,
            _ => n.div_ceil(16),
        }
    }

    fn trip(self) -> Ops {
        let base = Ops {
            loads: 1.0,
            width: 16.0,
            fresh: 1.0,
            ..Ops::default()
        };
        match self {
            RocmGemvKind::Bf16 | RocmGemvKind::Bf16Narrow => Ops {
                float: 8.0,
                int: 4.0,
                ..base
            },
            RocmGemvKind::Int4 => Ops {
                float: 32.0,
                int: 32.0,
                shared: 16.0,
                ..base
            },
            RocmGemvKind::Mxfp8 => Ops {
                float: 32.0,
                int: 16.0,
                shared: 16.0,
                e4m3: 32.0,
                ..base
            },
        }
    }
}

/// RDNA 3/3.5 streaming decode GEMV kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.name(),
    sizes = [1, self.n, self.k],
    work = self.n * self.k,
    accepts = gemv_accepts,
    emits = f32_only,
    needs = rocm_needs(self.kind.smem(), self.kind.threads(), false),
    order = Order::PerConfig,
    source = rocm_source(rocm_linear_source(), self.kind.entry()),
    launches = gemv_launches,
    shape = Shape::Grid { across: self.kind.groups(self.n), down: 1, threads: self.kind.threads() },
    load = gemv_load,
    unproven,
)]
struct RocmGemv {
    n: usize,
    k: usize,
    mode: u32,
    a: f32,
    b: f32,
    kind: RocmGemvKind,
    form: &'static dyn Stored,
}

fn gemv_accepts(k: &RocmGemv, input: usize, form: &dyn Stored) -> bool {
    weight_x_accepts(input, false, k.form, form)
}

fn gemv_load(k: &RocmGemv, _: usize) -> Option<Load> {
    let step = match k.kind {
        RocmGemvKind::Bf16 => 256.0,
        RocmGemvKind::Bf16Narrow => 2048.0,
        RocmGemvKind::Int4 | RocmGemvKind::Mxfp8 => 1024.0,
    };
    let group = k.kind.threads();
    Some(Load {
        threads: k.kind.groups(k.n) * group,
        group,
        trips: (k.k as f64 / step).max(1.0),
        trip: k.kind.trip(),
        reads: vec![k.form.bytes(k.n * k.k) as f64, (4 * k.k) as f64],
        written: (4 * k.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn gemv_launches(k: &RocmGemv, step: &Step) -> Vec<Launch> {
    let act = matches!(k.kind, RocmGemvKind::Bf16 | RocmGemvKind::Bf16Narrow)
        .then_some((k.mode, k.a, k.b));
    gemv_launch(k.shape().expect("a grid"), k.n, k.k, act, step)
}

fn check_gemv(
    tree: &Tree,
    mat: T,
    mode: u32,
    a: f32,
    b: f32,
    _y_bf16: bool,
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
    if node.form.name() != "f32" || tree.node(x).form.name() != "f32" || !tree.leaf(w) {
        return Some(Err("rocm_gemv requires f32 io and leaf weight".into()));
    }
    let form = tree.node(w).form;
    let kind = match form.name() {
        "bf16" if k.is_multiple_of(8) && n <= 64 && k >= 1024 => RocmGemvKind::Bf16Narrow,
        "bf16" if k.is_multiple_of(8) => RocmGemvKind::Bf16,
        "bf16" => return Some(Err(format!("k={k} is not a multiple of 8 for bf16"))),
        "int4_g32" if mode == 0 && k.is_multiple_of(32) => RocmGemvKind::Int4,
        "int4_g32" if mode != 0 => return None,
        "int4_g32" => return Some(Err(format!("k={k} is not a multiple of 32 for int4_g32"))),
        "mxfp8" if mode == 0 && k.is_multiple_of(32) => RocmGemvKind::Mxfp8,
        "mxfp8" if mode != 0 => return None,
        "mxfp8" => return Some(Err(format!("k={k} is not a multiple of 32 for mxfp8"))),
        other => {
            return Some(Err(format!(
                "weight form {other} (supports bf16, int4_g32, mxfp8)"
            )));
        }
    };
    let kernel = RocmGemv {
        n,
        k,
        mode,
        a,
        b,
        kind,
        form,
    };
    Some(Ok(vec![Draft::new(kernel, [x, w], n)]))
}

fn propose_gemv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    gemv_propose(tree, at, check_gemv)
}

fn why_gemv(tree: &Tree, at: T) -> Option<String> {
    check_gemv(tree, at, 0, 1.0, 1.0, false)?.err()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RocmMmForm {
    Bf16,
    Int4,
    Mxfp8,
}

impl RocmMmForm {
    fn entry(self, bm: usize, split: bool) -> &'static str {
        match (self, bm, split) {
            (RocmMmForm::Bf16, 16, false) => "rocm_mm_bf16_r16",
            (RocmMmForm::Bf16, 16, true) => "rocm_mm_bf16_r16_sk",
            (RocmMmForm::Bf16, 64, false) => "rocm_mm_bf16_r64",
            (RocmMmForm::Bf16, 64, true) => "rocm_mm_bf16_r64_sk",
            (RocmMmForm::Bf16, _, false) => "rocm_mm_bf16_r128",
            (RocmMmForm::Bf16, _, true) => "rocm_mm_bf16_r128_sk",
            (RocmMmForm::Int4, 16, false) => "rocm_mm_int4_r16",
            (RocmMmForm::Int4, 16, true) => "rocm_mm_int4_r16_sk",
            (RocmMmForm::Int4, 64, false) => "rocm_mm_int4_r64",
            (RocmMmForm::Int4, 64, true) => "rocm_mm_int4_r64_sk",
            (RocmMmForm::Int4, _, false) => "rocm_mm_int4_r128",
            (RocmMmForm::Int4, _, true) => "rocm_mm_int4_r128_sk",
            (RocmMmForm::Mxfp8, 16, false) => "rocm_mm_mxfp8_r16",
            (RocmMmForm::Mxfp8, 16, true) => "rocm_mm_mxfp8_r16_sk",
            (RocmMmForm::Mxfp8, 64, false) => "rocm_mm_mxfp8_r64",
            (RocmMmForm::Mxfp8, 64, true) => "rocm_mm_mxfp8_r64_sk",
            (RocmMmForm::Mxfp8, _, false) => "rocm_mm_mxfp8_r128",
            (RocmMmForm::Mxfp8, _, true) => "rocm_mm_mxfp8_r128_sk",
        }
    }

    fn name(self, split: bool) -> &'static str {
        match (self, split) {
            (RocmMmForm::Bf16, false) => "rocm_mm_bf16",
            (RocmMmForm::Bf16, true) => "rocm_mm_bf16_sk",
            (RocmMmForm::Int4, false) => "rocm_mm_int4_g32",
            (RocmMmForm::Int4, true) => "rocm_mm_int4_g32_sk",
            (RocmMmForm::Mxfp8, false) => "rocm_mm_mxfp8",
            (RocmMmForm::Mxfp8, true) => "rocm_mm_mxfp8_sk",
        }
    }

    fn smem(self, bm: usize) -> usize {
        match bm {
            16 => 10 * 1024,
            64 => 12 * 1024,
            _ => 22 * 1024,
        }
    }
}

/// RDNA 3/3.5 WMMA GEMM kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.name(self.ks > 1),
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = mm_accepts,
    emits = f32_only,
    needs = rocm_needs(
        self.kind.smem(mm_bm(self.kind, self.m, self.n)),
        mm_threads(self.kind, self.m, self.n),
        true,
    ),
    order = Order::PerConfig,
    source = rocm_source(
        rocm_linear_source(),
        self.kind.entry(mm_bm(self.kind, self.m, self.n), self.ks > 1),
    ),
    launches = mm_launches,
    shape = Shape::grid(
        mm_grid(self.kind, self.m, self.n, self.ks),
        mm_threads(self.kind, self.m, self.n),
    ),
    load = mm_load,
    unproven,
)]
struct RocmMm {
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    mode: u32,
    kind: RocmMmForm,
    form: &'static dyn Stored,
}

fn mm_bm(kind: RocmMmForm, m: usize, n: usize) -> usize {
    if m <= 16 {
        16
    } else if !matches!(kind, RocmMmForm::Mxfp8)
        && m >= 128
        && n >= 128
        && m.div_ceil(128) * n.div_ceil(128) >= 32
    {
        128
    } else {
        64
    }
}

fn mm_bn(kind: RocmMmForm, m: usize, n: usize) -> usize {
    if mm_bm(kind, m, n) == 128 { 128 } else { 64 }
}

fn mm_threads(kind: RocmMmForm, m: usize, n: usize) -> usize {
    if mm_bm(kind, m, n) == 128 {
        THREADS
    } else {
        MM_THREADS
    }
}

fn mm_grid(kind: RocmMmForm, m: usize, n: usize, ks: usize) -> (usize, usize) {
    (
        n.div_ceil(mm_bn(kind, m, n)) * ks,
        m.div_ceil(mm_bm(kind, m, n)),
    )
}

fn mm_accepts(k: &RocmMm, input: usize, form: &dyn Stored) -> bool {
    if k.ks == 1 && input >= 2 {
        return epi_accepts(k.mode, input, 2, form);
    }
    weight_x_accepts(input, false, k.form, form)
}

fn mm_load(k: &RocmMm, _: usize) -> Option<Load> {
    let (across, down) = mm_grid(k.kind, k.m, k.n, k.ks);
    let threads = mm_threads(k.kind, k.m, k.n);
    let mut reads = vec![k.form.bytes(k.n * k.k) as f64, (4 * k.m * k.k) as f64];
    if k.ks == 1 {
        push_epi_reads(&mut reads, k.mode, [k.m, k.n], 4);
    }
    let mut load = Load {
        threads: across * down * threads,
        group: threads,
        trips: (k.k.div_ceil(32) as f64 / k.ks as f64).max(1.0),
        reads,
        written: (4 * k.ks * k.m * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn mm_launches(k: &RocmMm, step: &Step) -> Vec<Launch> {
    let shape = k.shape().expect("a grid");
    let other = if k.ks == 1 && has_other(k.mode) {
        step.read(2)
    } else {
        step.read(0)
    };
    let bias = if k.ks == 1 && has_bias2(k.mode) {
        step.read(3)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        other,
        bias,
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
        Arg::count(k.ks),
    ];
    vec![Launch { args, shape }]
}

struct RocmSite {
    x: T,
    w: T,
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    kind: RocmMmForm,
    form: &'static dyn Stored,
}

impl MmDims for RocmSite {
    fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

impl RocmSite {
    fn draft(self, mode: u32, other: Option<T>, bias2: Option<T>) -> Vec<Draft> {
        let mm = RocmMm {
            m: self.m,
            n: self.n,
            k: self.k,
            ks: self.ks,
            mode: if self.ks > 1 { 0 } else { mode },
            kind: self.kind,
            form: self.form,
        };
        mm_draft(
            mm,
            [self.m, self.n],
            [self.x, self.w],
            self.ks,
            mode,
            [other, bias2],
            true,
        )
    }
}

fn rocm_mm_ks(kind: RocmMmForm, m: usize, n: usize, k: usize) -> usize {
    let chunks = k.div_ceil(32);
    let base = n.div_ceil(mm_bn(kind, m, n)) * m.div_ceil(mm_bm(kind, m, n));
    if chunks < 32 || base >= 32 {
        return 1;
    }
    let cap = if base >= 16 { 4 } else { 8 }.min(chunks / 4);
    let want = 48usize.div_ceil(base).clamp(1, cap);
    pick_divisor(chunks, want, cap)
}

fn check_mm(tree: &Tree, mat: T) -> Option<Result<RocmSite, String>> {
    let MatmulBase {
        x,
        w,
        m,
        n,
        k,
        x_bf16,
        y_bf16,
        form,
    } = match check_mm_base(tree, mat)? {
        Ok(b) => b,
        Err(e) => return Some(Err(e)),
    };
    if x_bf16 || y_bf16 {
        return Some(Err("rocm_mm requires f32 io".into()));
    }
    let kind = match form.name() {
        "bf16" if k.is_multiple_of(16) => RocmMmForm::Bf16,
        "bf16" => return Some(Err(format!("k={k} is not a multiple of 16 for bf16"))),
        "int4_g32" if k.is_multiple_of(32) => RocmMmForm::Int4,
        "int4_g32" => return Some(Err(format!("k={k} is not a multiple of 32 for int4_g32"))),
        "mxfp8" if k.is_multiple_of(32) => RocmMmForm::Mxfp8,
        "mxfp8" => return Some(Err(format!("k={k} is not a multiple of 32 for mxfp8"))),
        other => {
            return Some(Err(format!(
                "weight form {other} (supports bf16, int4_g32, mxfp8)"
            )));
        }
    };
    let ks = rocm_mm_ks(kind, m, n, k);
    Some(Ok(RocmSite {
        x,
        w,
        m,
        n,
        k,
        ks,
        kind,
        form,
    }))
}

fn mm_site(tree: &Tree, mat: T) -> Option<RocmSite> {
    check_mm(tree, mat)?.ok()
}

pub(crate) fn propose_rocm_mm(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, mode, other, bias2) = match_cuda_site(tree, at, mm_site)?;
    Some(site.draft(mode, other, bias2))
}

fn why_mm(tree: &Tree, at: T) -> Option<String> {
    check_mm(tree, at)?.err()
}

rocm_rule!("rocm_gemv", propose_gemv, why = why_gemv);
rocm_rule!("rocm_mm", propose_rocm_mm, why = why_mm);
