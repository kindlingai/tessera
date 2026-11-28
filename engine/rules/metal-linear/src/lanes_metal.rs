//! Small-batch Metal GEMV for small-K, small-N and wide-N shapes.

use tessera_codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_linear::{Product, Variant, place};
use tessera_lower::form::{BF16_FORM, F32_FORM, Load as Stored};
use tessera_lower::kernels::finish;
use tessera_lower::{Backend, Draft, GUESS, Kernel, Launch, Load, Needs, Shape, Step, T, Tree};

/// Backends this rule skips (Metal / portable GPU only).
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

/// Maximum rows handled per launch.
const MAX_ROWS: usize = 8;

/// Elements per wide load.
const WIDE: u32 = 4;

/// Lanes per column for small-K shapes (`k <= SMALL_K`).
const SUB: u32 = 8;

/// Threadgroup size for small-K mode (4 SIMD groups = 16 columns).
const SUB_THREADS: u32 = 128;

/// Maximum `k` for sub-warp mode.
const SMALL_K: usize = 128;

/// Maximum `n` for cooperative TG-per-column mode.
const SMALL_N: usize = 512;

/// Columns computed per SIMD group in wide-N mode.
const WIDE_COLS: u32 = 4;

/// SIMD groups per threadgroup in wide-N mode.
const WIDE_SIMDS: u32 = 2;

/// Threadgroup size for wide-N mode.
const WIDE_THREADS: u32 = WIDE_SIMDS * LANES as u32;

/// Metal GEMV schedule for `(n, k)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// 8 lanes per column, 128 threads per group.
    SmallK,
    /// One group per column, shared warp reduction.
    SmallN { simds: u32 },
    /// 4 columns per SIMD group, reusing `x`.
    WideN { pair: bool },
}

impl Mode {
    fn tag(self) -> u8 {
        match self {
            Self::SmallK => 0,
            Self::SmallN { simds: 2 } => 1,
            Self::SmallN { .. } => 2,
            Self::WideN { pair: false } => 3,
            Self::WideN { pair: true } => 4,
        }
    }

    fn threads(self) -> usize {
        match self {
            Self::SmallK => SUB_THREADS as usize,
            Self::SmallN { simds } => simds as usize * LANES,
            Self::WideN { .. } => WIDE_THREADS as usize,
        }
    }

    fn group_bytes(self) -> usize {
        match self {
            Self::SmallN { simds } => 4 * simds.max(4) as usize,
            _ => 0,
        }
    }

    fn groups_per_row(self, n: usize) -> usize {
        match self {
            Self::SmallK => n.div_ceil((SUB_THREADS / SUB) as usize),
            Self::SmallN { .. } => n,
            Self::WideN { .. } => n.div_ceil((WIDE_SIMDS * WIDE_COLS) as usize),
        }
    }

    fn total_threads(self, m: usize, n: usize) -> Option<usize> {
        self.groups_per_row(n)
            .checked_mul(m)?
            .checked_mul(self.threads())
    }

    fn k_step(self) -> usize {
        match self {
            Self::SmallK => (SUB * WIDE) as usize,
            Self::SmallN { simds } => simds as usize * LANES * WIDE as usize,
            Self::WideN { pair } => LANES * if pair { 2 * WIDE } else { WIDE } as usize,
        }
    }

    fn shape(self, m: usize, n: usize) -> Shape {
        Shape::Grid {
            across: self.groups_per_row(n) * m,
            down: 1,
            threads: self.threads(),
        }
    }
}

/// Single-column GEMV: sub-warp or cooperative group.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn gemv_col(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    #[form(out_form)] out: BufMut<Dyn>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] simds: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let mut row = 0u32;
    let mut col = 0u32;
    let mut s = 0.0f32;
    if simds == 0 {
        let sub_lane = lane() % SUB;
        let mut blk = gid() / SUB_THREADS;
        if m > 1 {
            row = blk % m as u32;
            blk /= m as u32;
        }
        col = blk * (SUB_THREADS / SUB) + (lid() / SUB);
        if col >= n {
            return;
        }
        let at = col as u64 * k as u64;
        let mut x_base = 0u64;
        if m > 1 {
            x_base = row as u64 * k as u64;
        }
        for i in (sub_lane * WIDE..k).by(SUB * WIDE).unroll(4) {
            let c = w.load4(at + i as u64);
            let v = x.load4(x_base + i as u64);
            s += v.x * c.x + v.y * c.y + v.z * c.z + v.w * c.w;
        }
        s += shfl_xor(s, 4);
        s += shfl_xor(s, 2);
        s += shfl_xor(s, 1);
        if sub_lane != 0 {
            return;
        }
    } else {
        let mut smem = shared::<f32>(simds.max(4));
        let threads = simds * LANES as u32;
        let mut blk = gid() / threads;
        if m > 1 {
            row = blk % m as u32;
            blk /= m as u32;
        }
        col = blk;
        if col >= n {
            return;
        }
        let at = col as u64 * k as u64;
        let mut x_base = 0u64;
        if m > 1 {
            x_base = row as u64 * k as u64;
        }
        for i in (lid() * WIDE..k).by(threads * WIDE).unroll(4) {
            let c = w.load4(at + i as u64);
            let v = x.load4(x_base + i as u64);
            s += v.x * c.x + v.y * c.y + v.z * c.z + v.w * c.w;
        }
        s = sum32(s);
        if lane() == 0 {
            smem[lid() / LANES as u32] = s;
        }
        barrier();
        if lid() != 0 {
            return;
        }
        s = 0.0f32;
        for j in 0..simds {
            s += smem[j];
        }
    }
}

/// 4 columns per SIMD group, reusing `x`.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn gemv_wide_n(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(form)] w: Buf<Dyn>,
    #[form(out_form)] out: BufMut<Dyn>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] pair: bool,
    #[build] x_form: &'static dyn Stored,
    #[build] form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let mut row = 0u32;
    let mut blk = gid() / WIDE_THREADS;
    if m > 1 {
        row = blk % m as u32;
        blk /= m as u32;
    }
    let simd = blk * WIDE_SIMDS + lid() / LANES as u32;
    let col0 = simd * WIDE_COLS;
    if col0 >= n {
        return;
    }
    let mut col1 = col0;
    if col0 + 1 < n {
        col1 = col0 + 1;
    }
    let mut col2 = col0;
    if col0 + 2 < n {
        col2 = col0 + 2;
    }
    let mut col3 = col0;
    if col0 + 3 < n {
        col3 = col0 + 3;
    }
    let at0 = col0 as u64 * k as u64;
    let at1 = col1 as u64 * k as u64;
    let at2 = col2 as u64 * k as u64;
    let at3 = col3 as u64 * k as u64;
    let mut x_base = 0u64;
    if m > 1 {
        x_base = row as u64 * k as u64;
    }
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    if pair {
        // Two contiguous load4s per lane coalesce into 128-bit loads.
        for i in (lane() * (2 * WIDE)..k)
            .by(LANES as u32 * 2 * WIDE)
            .unroll(2)
        {
            let v0 = x.load4(x_base + i as u64);
            let v1 = x.load4(x_base + (i + WIDE) as u64);
            let w0a = w.load4(at0 + i as u64);
            let w0b = w.load4(at0 + (i + WIDE) as u64);
            s0 += v0.x * w0a.x + v0.y * w0a.y + v0.z * w0a.z + v0.w * w0a.w;
            s0 += v1.x * w0b.x + v1.y * w0b.y + v1.z * w0b.z + v1.w * w0b.w;
            let w1a = w.load4(at1 + i as u64);
            let w1b = w.load4(at1 + (i + WIDE) as u64);
            s1 += v0.x * w1a.x + v0.y * w1a.y + v0.z * w1a.z + v0.w * w1a.w;
            s1 += v1.x * w1b.x + v1.y * w1b.y + v1.z * w1b.z + v1.w * w1b.w;
            let w2a = w.load4(at2 + i as u64);
            let w2b = w.load4(at2 + (i + WIDE) as u64);
            s2 += v0.x * w2a.x + v0.y * w2a.y + v0.z * w2a.z + v0.w * w2a.w;
            s2 += v1.x * w2b.x + v1.y * w2b.y + v1.z * w2b.z + v1.w * w2b.w;
            let w3a = w.load4(at3 + i as u64);
            let w3b = w.load4(at3 + (i + WIDE) as u64);
            s3 += v0.x * w3a.x + v0.y * w3a.y + v0.z * w3a.z + v0.w * w3a.w;
            s3 += v1.x * w3b.x + v1.y * w3b.y + v1.z * w3b.z + v1.w * w3b.w;
        }
    } else {
        for i in (lane() * WIDE..k).by(LANES as u32 * WIDE).unroll(4) {
            let v = x.load4(x_base + i as u64);
            let w0 = w.load4(at0 + i as u64);
            s0 += v.x * w0.x + v.y * w0.y + v.z * w0.z + v.w * w0.w;
            let w1 = w.load4(at1 + i as u64);
            s1 += v.x * w1.x + v.y * w1.y + v.z * w1.z + v.w * w1.w;
            let w2 = w.load4(at2 + i as u64);
            s2 += v.x * w2.x + v.y * w2.y + v.z * w2.z + v.w * w2.w;
            let w3 = w.load4(at3 + i as u64);
            s3 += v.x * w3.x + v.y * w3.y + v.z * w3.z + v.w * w3.w;
        }
    }
    s0 = sum32(s0);
    s1 = sum32(s1);
    s2 = sum32(s2);
    s3 = sum32(s3);
    if lane() != 0 {
        return;
    }
    let mut vals = local::<f32>(WIDE_COLS as usize);
    vals[0] = s0;
    vals[1] = s1;
    vals[2] = s2;
    vals[3] = s3;
}

/// Small-batch Metal GEMV across small-K, small-N and wide-N regimes.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("lanes_metal"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    needs = lanes_metal_needs(self.mode),
    order = Order::PerConfig,
    source = self.product.source(self.tree(), "lanes_metal"),
    launches = lanes_metal_launches,
    shape = self.mode.shape(self.product.m, self.product.n),
    load = lanes_metal_load,
    guess = GUESS,
)]
pub(crate) struct LanesMetal {
    pub product: Product,
    mode: Mode,
}

impl LanesMetal {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let Product {
            m,
            x_form,
            form,
            out_form,
            ..
        } = self.product;
        let mode = self.mode;
        let build = || match mode {
            Mode::SmallK => gemv_col_ir(m, 0, x_form, form, out_form),
            Mode::SmallN { simds } => gemv_col_ir(m, simds, x_form, form, out_form),
            Mode::WideN { pair } => gemv_wide_n_ir(m, pair, x_form, form, out_form),
        };
        let key: Variant = (
            "lanes_metal",
            mode.tag() as usize,
            x_form.name(),
            form.name(),
            out_form.name(),
            0,
            m,
        );
        let mut body = self.product.tree(key, build);
        match mode {
            Mode::SmallK | Mode::SmallN { .. } => {
                let at = if m > 1 {
                    place(Expr::var("row"), Expr::var("col"))
                } else {
                    Expr::cast(Ty::U64, Expr::var("col"))
                };
                body.stmts
                    .extend(finish(&self.product.after, Expr::var("s"), at));
            }
            Mode::WideN { .. } => {
                let col = Expr::bin(BinOp::Add, Expr::var("col0"), Expr::var("c"));
                let at = if m > 1 {
                    place(Expr::var("row"), col.clone())
                } else {
                    Expr::cast(Ty::U64, col.clone())
                };
                let val = Expr::index(Expr::var("vals"), Expr::var("c"));
                let then = finish(&self.product.after, val, at);
                body.stmts.push(Stmt::For {
                    var: "c".into(),
                    from: Expr::Int(0),
                    to: Expr::Int(i64::from(WIDE_COLS)),
                    step: Expr::Int(1),
                    unroll: 0,
                    body: vec![Stmt::If {
                        cond: Expr::bin(BinOp::Lt, col, Expr::var("n")),
                        then,
                        els: Vec::new(),
                    }],
                });
            }
        }
        body
    }
}

fn lanes_metal_needs(mode: Mode) -> Needs {
    Needs {
        lanes: LANES,
        group_bytes: mode.group_bytes(),
        group_threads: mode.threads(),
        lacks: LACKS,
        ..Needs::default()
    }
}

fn lanes_metal_load(k: &LanesMetal, _: usize) -> Option<Load> {
    let Product {
        m,
        n,
        k: depth,
        form,
        out_form,
        ..
    } = k.product;
    let group = k.mode.threads();
    let threads = k.mode.total_threads(m, n)?;
    let (trips, flight) = match k.mode {
        Mode::SmallK => (
            depth.div_ceil((SUB * WIDE) as usize) as f64,
            (form.bytes(WIDE as usize) * 4) as f64,
        ),
        Mode::SmallN { simds } => (
            depth.div_ceil(simds as usize * LANES * WIDE as usize) as f64,
            (form.bytes(WIDE as usize) * 4) as f64,
        ),
        Mode::WideN { pair } => {
            let elems = if pair { 2 * WIDE } else { WIDE } as usize;
            let cols = WIDE_COLS as usize;
            (
                depth.div_ceil(LANES * elems) as f64,
                (form.bytes(elems) * cols * 2) as f64,
            )
        }
    };
    Some(Load {
        threads,
        group,
        trips,
        trip: k.tree().trip(),
        reads: k.product.read_bytes(),
        written: out_form.bytes(m * n) as f64,
        flight,
        group_bytes: k.mode.group_bytes() as f64,
    })
}

fn lanes_metal_launches(k: &LanesMetal, step: &Step) -> Vec<Launch> {
    let Product { m, n, k: depth, .. } = k.product;
    k.product.launch(step, &[n, depth], k.mode.shape(m, n))
}

fn pick_mode(n: usize, k: usize) -> Mode {
    if k <= SMALL_K {
        Mode::SmallK
    } else if n <= SMALL_N {
        let simds = if n <= 128 { 8 } else { 2 };
        Mode::SmallN { simds }
    } else {
        Mode::WideN {
            pair: k.is_multiple_of((2 * WIDE) as usize),
        }
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    if product.picked.is_some()
        || product.m == 0
        || product.m > MAX_ROWS
        || !product.k.is_multiple_of(WIDE as usize)
    {
        return None;
    }
    let name = product.form.name();
    if name != BF16_FORM.name() && name != F32_FORM.name() {
        return None;
    }
    let mode = pick_mode(product.n, product.k);
    if u32::try_from(product.n).is_err()
        || product
            .k
            .checked_add(mode.k_step() * 4)
            .is_none_or(|end| end > u32::MAX as usize)
        || mode
            .total_threads(product.m, product.n)
            .is_none_or(|t| t > u32::MAX as usize)
    {
        return None;
    }
    let len = product.outputs();
    Some(vec![Draft::new(LanesMetal { product, mode }, reads, len)])
}

tessera_lower::rule!("lanes_metal", propose);
