//! A matmul on the matrix unit, staged through group memory.

use crate::product::{Product, Variant, place};
use tessera_codegen::ir::{BinOp, Body, Expr, Stmt};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::{BF16_FORM, Load as Stored, plain};
use tessera_lower::kernels::finish;
use tessera_lower::sites::Near;
use tessera_lower::{
    Compute, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Tree, knob,
};

knob!("tiled", "band", 1, 64);
knob!("tiled", "block_m", 32, 128);
knob!("tiled", "block_n", 32, 128);

/// How a block of the output is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Rows, columns and depth of one group's block.
    pub block: (usize, usize, usize),
    /// How SIMD groups split the block's rows and columns.
    pub simds: (usize, usize),
    /// Its cost curve's name. The weight form adds a suffix.
    pub name: &'static str,
    /// The same, reading neighbours by id.
    pub near: &'static str,
}

/// Every matmul's schedules. `w16` SIMD groups own 16 by 16.
pub const SCHEDULES: [Schedule; 7] = [
    Schedule::new((64, 64, 32), (2, 2), "tiled_64x64", "near_64x64"),
    Schedule::new((32, 64, 32), (1, 2), "tiled_32x64", "near_32x64"),
    Schedule::new((128, 64, 32), (4, 2), "tiled_128x64", "near_128x64"),
    Schedule::new((64, 128, 32), (2, 4), "tiled_64x128", "near_64x128"),
    Schedule::new((64, 64, 32), (4, 4), "tiled_64x64_w16", "near_64x64_w16"),
    Schedule::new((32, 32, 32), (2, 2), "tiled_32x32_w16", "near_32x32_w16"),
    Schedule::new(
        (64, 64, 32),
        (2, 4),
        "tiled_64x64_w32x16",
        "near_64x64_w32x16",
    ),
];

/// Schedules from bf16 parts. SIMD groups own multiples of 16.
pub const PARTS: [Schedule; 2] = [
    Schedule::new((64, 64, 32), (2, 4), "parts_64x64", ""),
    Schedule::new((128, 128, 32), (2, 4), "parts_128x128", ""),
];

/// One-pass bf16 schedules. SIMD groups own multiples of 16.
pub const BF16: [Schedule; 2] = [
    Schedule::new((64, 64, 32), (2, 4), "bf16_64x64", ""),
    Schedule::new((128, 128, 32), (2, 4), "bf16_128x128", ""),
];

/// Slabs of `x` in bf16. Three hold an f32 exactly.
const SLABS: u32 = 3;

/// Floats of `x` a thread holds between blocks, at most.
const FLOATS: usize = 16;

/// Words of `w` a thread holds between blocks, at most.
const WORDS: usize = 8;

impl Schedule {
    const fn new(
        block: (usize, usize, usize),
        simds: (usize, usize),
        name: &'static str,
        near: &'static str,
    ) -> Self {
        Self {
            block,
            simds,
            name,
            near,
        }
    }

    fn threads(&self) -> usize {
        self.simds.0 * self.simds.1 * LANES
    }

    /// The rows and columns one SIMD group owns.
    fn simd(&self) -> (usize, usize) {
        (self.block.0 / self.simds.0, self.block.1 / self.simds.1)
    }
}

/// Floats of padding after each staged row, against bank conflicts.
const PAD: u32 = 4;

/// Elements one wide load brings.
const WIDE: u32 = 4;

/// Pixels a 3x3 kernel reads.
const TAPS: u32 = 9;

/// Row blocks in one band when `x` outweighs `w`.
const BAND: u32 = 8;

/// Stages `span` rows of `src` from `r0`, `k0`, zero outside.
#[kernel]
fn stage<F: Elems<Elem = f32>>(
    dst: &mut Shared<f32>,
    #[form(form)] src: Buf<F>,
    r0: u32,
    k0: u32,
    limit: u32,
    k: u32,
    #[build] span: u32,
    #[build] bk: u32,
    #[build] threads: u32,
    #[build] wide: bool,
    #[build] form: &'static dyn Stored,
) {
    if wide {
        // Groups of four along `k` need one test per load.
        for i in (lid()..span * (bk / WIDE)).by(threads) {
            let r = i / (bk / WIDE);
            let c = (i % (bk / WIDE)) * WIDE;
            let at = r * (bk + PAD) + c;
            if r0 + r < limit && k0 + c < k {
                let v = src.load4((r0 + r) as u64 * k as u64 + (k0 + c) as u64);
                dst[at] = v.x;
                dst[at + 1] = v.y;
                dst[at + 2] = v.z;
                dst[at + 3] = v.w;
            } else {
                dst[at] = 0.0;
                dst[at + 1] = 0.0;
                dst[at + 2] = 0.0;
                dst[at + 3] = 0.0;
            }
        }
    } else {
        for i in (lid()..span * bk).by(threads) {
            let r = i / bk;
            let c = i % bk;
            if r0 + r < limit && k0 + c < k {
                dst[r * (bk + PAD) + c] = src[(r0 + r) as u64 * k as u64 + (k0 + c) as u64];
            } else {
                dst[r * (bk + PAD) + c] = 0.0;
            }
        }
    }
}

/// Stages rows of neighbours, one tap: `bk` divides `cin`.
#[kernel]
fn stage_near<F: Elems<Elem = f32>>(
    dst: &mut Shared<f32>,
    #[form(form)] src: Buf<F>,
    ids: Buf<U32>,
    r0: u32,
    k0: u32,
    limit: u32,
    pixels: u32,
    cin: u32,
    #[build] span: u32,
    #[build] bk: u32,
    #[build] threads: u32,
    #[build] form: &'static dyn Stored,
) {
    let tap = k0 / cin;
    let c0 = k0 % cin;
    for i in (lid()..span * (bk / WIDE)).by(threads) {
        let r = i / (bk / WIDE);
        let c = (i % (bk / WIDE)) * WIDE;
        let at = r * (bk + PAD) + c;
        let row = r0 + r;
        let p = row % pixels;
        let mut id = pixels;
        if row < limit {
            id = ids[(p * TAPS + tap) as u64];
        }
        if id < pixels {
            let v = src.load4((row - p + id) as u64 * cin as u64 + (c0 + c) as u64);
            dst[at] = v.x;
            dst[at + 1] = v.y;
            dst[at + 2] = v.z;
            dst[at + 3] = v.w;
        } else {
            dst[at] = 0.0;
            dst[at + 1] = 0.0;
            dst[at + 2] = 0.0;
            dst[at + 3] = 0.0;
        }
    }
}

/// This thread's words of a block of bf16 `src`.
#[kernel]
fn fetch_bf16(
    regs: &mut [u32; WORDS],
    #[form(form)] src: Buf<Dyn>,
    r0: u32,
    k0: u32,
    limit: u32,
    k: u32,
    #[build] count: u32,
    #[build] bk: u32,
    #[build] threads: u32,
    #[build] form: &'static dyn Stored,
) {
    for j in 0..count {
        let i = lid() + j * threads;
        let r = i / (bk / 2);
        let c = (i % (bk / 2)) * 2;
        regs[j as usize] = 0u32;
        if r0 + r < limit && k0 + c < k {
            regs[j as usize] = src.word(((r0 + r) * k + k0 + c) * 2);
        }
    }
}

/// Puts what `fetch_bf16` read in group memory.
#[kernel]
fn stash_bf16(
    dst: &mut Shared<u32>,
    regs: &mut [u32; WORDS],
    #[build] count: u32,
    #[build] bk: u32,
    #[build] threads: u32,
) {
    for j in 0..count {
        let i = lid() + j * threads;
        dst[(i / (bk / 2)) * (bk / 2 + PAD) + i % (bk / 2)] = regs[j as usize];
    }
}

/// This thread's floats of a block of `src`.
#[kernel]
fn fetch_rows(
    regs: &mut [f32; FLOATS],
    #[form(form)] src: Buf<Dyn>,
    r0: u32,
    k0: u32,
    limit: u32,
    k: u32,
    #[build] count: u32,
    #[build] bk: u32,
    #[build] threads: u32,
    #[build] form: &'static dyn Stored,
) {
    for j in 0..count / 4 {
        let i = lid() + j * threads;
        let r = i / (bk / WIDE);
        let c = (i % (bk / WIDE)) * WIDE;
        // Past the block it reads 0. `zero()` leaves it unset.
        let mut v = F32x4::new(0.0, 0.0, 0.0, 0.0);
        if r0 + r < limit && k0 + c < k {
            v = src.load4((r0 + r) as u64 * k as u64 + (k0 + c) as u64);
        }
        regs[(4 * j) as usize] = v.x;
        regs[(4 * j + 1) as usize] = v.y;
        regs[(4 * j + 2) as usize] = v.z;
        regs[(4 * j + 3) as usize] = v.w;
    }
}

/// Stores `fetch_rows` floats as `parts` bf16 slabs.
#[kernel]
fn stash_parts(
    dst: &mut Shared<u32>,
    regs: &mut [f32; FLOATS],
    #[build] count: u32,
    #[build] span: u32,
    #[build] bk: u32,
    #[build] threads: u32,
    #[build] parts: u32,
) {
    let words = bk / 2 + PAD;
    for j in 0..count / 2 {
        let i = lid() + (j / 2) * threads;
        let at = (i / (bk / WIDE)) * words + (i % (bk / WIDE)) * 2 + j % 2;
        let mut lo = regs[(2 * j) as usize];
        let mut hi = regs[(2 * j + 1) as usize];
        // Each part rounds what the parts before it left over.
        for p in 0..parts {
            let d = bf16_pair(hi, lo);
            dst[p * span * words + at] = d;
            lo -= f32::from_bits(d << 16);
            hi -= f32::from_bits(d & 0xffff0000);
        }
    }
}

/// A tiled kernel. The caller gives staging and multiply.
macro_rules! tiled_kernel {
    (
        $(#[$doc:meta])* $name:ident,
        [$x:ident, $a:ident, $row0:ident, $k0:ident, $m:ident, $k:ident],
        [$bm:ident, $bk:ident, $threads:ident, $wide:ident],
        [$w:ident, $b:ident, $col0:ident, $n:ident, $bn:ident, $wm0:ident, $wn0:ident, $acc:ident, $x_form:ident, $form:ident, $out_form:ident],
        [$($bufs:tt)*],
        [$($scalars:tt)*],
        [$($builds:tt)*],
        [$($slabs:tt)*],
        [$($stage:tt)*],
        [$($multiply:tt)*]
    ) => {
        $(#[$doc])*
        #[kernel]
        fn $name(
            #[form($x_form)] $x: Buf<Dyn>,
            #[form($form)] $w: Buf<Dyn>,
            $($bufs)*
            #[form($out_form)] out: BufMut<Dyn>,
            $m: u32,
            $n: u32,
            $k: u32,
            $($scalars)*
            #[build] $bm: u32,
            #[build] $bn: u32,
            #[build] $bk: u32,
            #[build] wm: u32,
            #[build] wn: u32,
            #[build] simds_n: u32,
            #[build] $threads: u32,
            #[build] $wide: bool,
            #[build] wbits: u32,
            #[build] band_cfg: u32,
            $($builds)*
            #[build] $x_form: &'static dyn Stored,
            #[build] $form: &'static dyn Stored,
            #[build] $out_form: &'static dyn Stored,
        ) {
            let simd = lid() / LANES as u32;
            let group = gid() / $threads;
            // Down a band, then across: groups share a weight block.
            let tiles_m = ($m + $bm - 1) / $bm;
            let mut band = band_cfg;
            if $n * wbits >= 32 * $m {
                band = tiles_m;
            }
            let per_band = band * (($n + $bn - 1) / $bn);
            let first = (group / per_band) * band;
            let mut height = band;
            if tiles_m - first < band {
                height = tiles_m - first;
            }
            let $row0 = (first + (group % per_band) % height) * $bm;
            let $col0 = ((group % per_band) / height) * $bn;
            let $wm0 = (simd / simds_n) * wm;
            let $wn0 = (simd % simds_n) * wn;
            $($slabs)*
            let mut $acc = tiles(wm, wn);
            for $k0 in (0..$k).by($bk) {
                $($stage)*
                barrier();
                $($multiply)*
                barrier();
            }
        }
    };
}

tiled_kernel!(
    /// Stages a block of `x` and `w`, then multiplies.
    tiled,
    [x, a_s, row0, k0, m, k],
    [bm, bk, threads, wide],
    [w, b_s, col0, n, bn, wm0, wn0, acc, x_form, form, out_form],
    [],
    [],
    [],
    [
        let mut a_s = shared::<f32>(bm * (bk + PAD));
        let mut b_s = shared::<f32>(bn * (bk + PAD));
    ],
    [
        stage(&mut a_s, x, row0, k0, m, k, bm, bk, threads, wide, x_form);
        stage(&mut b_s, w, col0, k0, n, k, bn, bk, threads, wide, form);
    ],
    [acc.block(&a_s, wm0 * (bk + PAD), &b_s, wn0 * (bk + PAD), bk + PAD, bk);]
);

tiled_kernel!(
    /// `tiled`, its rows each pixel's neighbours, gathered while staged.
    tiled_near,
    [x, a_s, row0, k0, m, k],
    [bm, bk, threads, wide],
    [w, b_s, col0, n, bn, wm0, wn0, acc, x_form, form, out_form],
    [ids: Buf<U32>,],
    [pixels: u32, cin: u32,],
    [],
    [
        let mut a_s = shared::<f32>(bm * (bk + PAD));
        let mut b_s = shared::<f32>(bn * (bk + PAD));
    ],
    [
        stage_near(&mut a_s, x, ids, row0, k0, m, pixels, cin, bm, bk, threads, x_form);
        stage(&mut b_s, w, col0, k0, n, k, bn, bk, threads, wide, form);
    ],
    [acc.block(&a_s, wm0 * (bk + PAD), &b_s, wn0 * (bk + PAD), bk + PAD, bk);]
);

tiled_kernel!(
    /// `tiled` on bf16 matrix units, in one or three passes.
    tiled_parts,
    [x, a_s, row0, k0, m, k],
    [bm, bk, threads, _wide],
    [w, b_s, col0, n, bn, wm0, wn0, acc, x_form, form, out_form],
    [],
    [],
    [#[build] parts: u32, #[build] direct_x: bool, #[build] direct_w: bool,],
    [
        let mut a_s = shared::<u32>(parts * bm * (bk / 2 + PAD));
        let mut b_s = shared::<u32>(bn * (bk / 2 + PAD));
        // The next block's loads fly while this one multiplies.
        let mut xr = [0.0f32; FLOATS];
        let mut xw = [0u32; WORDS];
        let mut wr = [0.0f32; FLOATS];
        let mut ww = [0u32; WORDS];
        if direct_x {
            fetch_bf16(&mut xw, x, row0, 0, m, k, bm * bk / 2 / threads, bk, threads, x_form);
        } else {
            fetch_rows(&mut xr, x, row0, 0, m, k, bm * bk / threads, bk, threads, x_form);
        }
        if direct_w {
            fetch_bf16(&mut ww, w, col0, 0, n, k, bn * bk / 2 / threads, bk, threads, form);
        } else {
            fetch_rows(&mut wr, w, col0, 0, n, k, bn * bk / threads, bk, threads, form);
        }
    ],
    [
        if direct_x {
            stash_bf16(&mut a_s, &mut xw, bm * bk / 2 / threads, bk, threads);
        } else {
            stash_parts(&mut a_s, &mut xr, bm * bk / threads, bm, bk, threads, parts);
        }
        if direct_w {
            stash_bf16(&mut b_s, &mut ww, bn * bk / 2 / threads, bk, threads);
        } else {
            stash_parts(&mut b_s, &mut wr, bn * bk / threads, bn, bk, threads, 1);
        }
    ],
    [
        if k0 + bk < k {
            if direct_x {
                fetch_bf16(&mut xw, x, row0, k0 + bk, m, k, bm * bk / 2 / threads, bk, threads, x_form);
            } else {
                fetch_rows(&mut xr, x, row0, k0 + bk, m, k, bm * bk / threads, bk, threads, x_form);
            }
            if direct_w {
                fetch_bf16(&mut ww, w, col0, k0 + bk, n, k, bn * bk / 2 / threads, bk, threads, form);
            } else {
                fetch_rows(&mut wr, w, col0, k0 + bk, n, k, bn * bk / threads, bk, threads, form);
            }
        }
        acc.parts(
        &a_s,
        wm0 * (bk / 2 + PAD),
        bm * (bk / 2 + PAD),
        parts,
        &b_s,
        wn0 * (bk / 2 + PAD),
        bk / 2 + PAD,
        bk
    );]
);

/// `x[m, k]` times `w[n, k]` transposed, by tiles.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed(self.name()),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    needs = tiled_needs(&self.schedule, self.parts),
    // Each device's tiles sum in their own order.
    order = Order::PerConfig,
    source = self.product.source(self.tree(), "tiled"),
    launches = tiled_launches,
    shape = tiled_shape(self),
    load = tiled_load,
    load_key = self.product.key_with((
        self.schedule.block,
        self.schedule.simds,
        self.schedule.name,
        self.schedule.near,
        self.parts,
        self.band,
    )),
    unproven,
)]
pub struct Tiled {
    pub product: Product,
    pub schedule: Schedule,
    /// Slabs of `x` in bf16, or zero for f32.
    pub parts: u32,
    pub band: u32,
}

impl Tiled {
    fn name(&self) -> &'static str {
        match self.product.near {
            Some(_) => self.schedule.near,
            None => self.schedule.name,
        }
    }

    /// Where the schedule sits in its list.
    fn at(&self) -> usize {
        let list: &[Schedule] = match self.parts {
            0 => &SCHEDULES,
            1 => &BF16,
            _ => &PARTS,
        };
        let at = list.iter().position(|s| *s == self.schedule);
        at.expect("a listed schedule")
    }

    fn groups(&self) -> (usize, usize) {
        let (bm, bn, _) = self.schedule.block;
        (self.product.m.div_ceil(bm), self.product.n.div_ceil(bn))
    }

    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let Product {
            m,
            n,
            k,
            x_form,
            form,
            out_form,
            ..
        } = self.product;
        let wide = k.is_multiple_of(WIDE as usize);
        let near = self.product.near.is_some();
        let direct_x = self.parts == 1 && bf16_fits(x_form, m * k);
        let direct_w = bf16_fits(form, n * k);
        let build = || {
            let s = &self.schedule;
            let (bm, bn, bk) = (s.block.0 as u32, s.block.1 as u32, s.block.2 as u32);
            let (wm, wn) = (s.simd().0 as u32, s.simd().1 as u32);
            let (simds_n, threads) = (s.simds.1 as u32, s.threads() as u32);
            let (wbits, parts, band) = (weight_bits(form), self.parts, self.band);
            match (near, parts) {
                (true, _) => tiled_near_ir(
                    bm, bn, bk, wm, wn, simds_n, threads, wide, wbits, band, x_form, form, out_form,
                ),
                (false, 0) => tiled_ir(
                    bm, bn, bk, wm, wn, simds_n, threads, wide, wbits, band, x_form, form, out_form,
                ),
                _ => tiled_parts_ir(
                    bm, bn, bk, wm, wn, simds_n, threads, wide, wbits, band, parts, direct_x,
                    direct_w, x_form, form, out_form,
                ),
            }
        };
        let kind = match (near, self.parts) {
            (true, _) => "tiled_near",
            (false, 0) => "tiled",
            (false, 1) => "tiled_bf16",
            _ => "tiled_parts",
        };
        let flags = u8::from(wide)
            | (u8::from(direct_x) << 1)
            | (u8::from(direct_w) << 2)
            | ((self.parts as u8) << 3);
        let key: Variant = (
            kind,
            self.at(),
            x_form.name(),
            form.name(),
            out_form.name(),
            flags,
            self.band as usize,
        );
        let mut body = self.product.tree(key, build);
        body.stmts.push(self.stores());
        body
    }

    /// Each sum through the stages, into its place.
    fn stores(&self) -> Stmt {
        let add = |a, b| Expr::bin(BinOp::Add, a, b);
        let gr = add(add(Expr::var("row0"), Expr::var("wm0")), Expr::var("r"));
        let gc = add(add(Expr::var("col0"), Expr::var("wn0")), Expr::var("c"));
        let o = place(Expr::var("gr"), Expr::var("gc"));
        let then = finish(&self.product.after, Expr::var("v"), o);
        let inside = Expr::bin(
            BinOp::And,
            Expr::bin(BinOp::Lt, Expr::var("gr"), Expr::var("m")),
            Expr::bin(BinOp::Lt, Expr::var("gc"), Expr::var("n")),
        );
        let body = vec![
            Stmt::Let {
                name: "gr".into(),
                init: gr,
            },
            Stmt::Let {
                name: "gc".into(),
                init: gc,
            },
            Stmt::If {
                cond: inside,
                then,
                els: Vec::new(),
            },
        ];
        Stmt::TileEach {
            acc: "acc".into(),
            body,
        }
    }
}

/// Whether `form` of `elems` fits direct bf16 word loads.
fn bf16_fits(form: &dyn Stored, elems: usize) -> bool {
    form.name() == BF16_FORM.name() && (form.bytes(elems) as u64) < (1 << 32)
}

/// Bits per weight, rounded down.
fn weight_bits(form: &dyn Stored) -> u32 {
    (form.bytes(1024) * 8 / 1024) as u32
}

/// Row blocks per band. The kernel computes the same count.
fn band(m: usize, n: usize, bm: usize, form: &dyn Stored, cfg: u32) -> usize {
    let tiles_m = m.div_ceil(bm);
    match n * weight_bits(form) as usize >= 32 * m {
        true => tiles_m,
        false => (cfg as usize).min(tiles_m),
    }
}

fn tiled_load(t: &Tiled, _: usize) -> Option<Load> {
    let Product {
        m,
        n,
        k,
        x_form,
        form,
        out_form,
        ..
    } = t.product;
    let (gm, gn) = t.groups();
    let (bm, _, bk) = t.schedule.block;
    // `x` stays cached. `w` streams once per band.
    let rows = match t.product.near {
        Some(near) => x_form.bytes(m * near.cin) as f64,
        None => x_form.bytes(m * k) as f64,
    };
    let mut reads = vec![rows];
    let bands = gm.div_ceil(band(m, n, bm, form, t.band));
    reads.extend((0..bands).map(|_| form.bytes(n * k) as f64));
    reads.extend(
        t.product
            .near
            .map(|near| (4 * near.pixels * near.taps) as f64),
    );
    for other in &t.product.other_forms {
        reads.push(other.bytes(m * n) as f64);
    }
    Some(Load {
        threads: gm * gn * t.schedule.threads(),
        group: t.schedule.threads(),
        trips: (k.div_ceil(bk)) as f64,
        trip: t.tree().trip(),
        reads,
        written: out_form.bytes(m * n) as f64,
        flight: 16.0,
        ..Load::default()
    })
}

fn tiled_shape(t: &Tiled) -> Shape {
    let (gm, gn) = t.groups();
    Shape::Grid {
        across: gm * gn,
        down: 1,
        threads: t.schedule.threads(),
    }
}

fn tiled_launches(t: &Tiled, step: &Step) -> Vec<Launch> {
    let Product { m, n, k, .. } = t.product;
    let shape = tiled_shape(t);
    match t.product.near {
        Some(near) => t
            .product
            .launch(step, &[m, n, k, near.pixels, near.cin], shape),
        None => t.product.launch(step, &[m, n, k], shape),
    }
}

/// Staged slabs, in floats or bf16 pairs.
fn tiled_needs(schedule: &Schedule, parts: u32) -> Needs {
    let (bm, bn, bk) = schedule.block;
    let (pad, parts) = (PAD as usize, parts as usize);
    let group_bytes = match parts {
        0 => 4 * (bm + bn) * (bk + pad),
        _ => 4 * (parts * bm + bn) * (bk / 2 + pad),
    };
    Needs {
        lanes: LANES,
        group_bytes,
        group_threads: schedule.threads(),
        tile: true,
        bf16_tile: parts > 0,
        ..Needs::default()
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    if product.picked.is_some() {
        return None;
    }
    let band = tree.knobs.get("tiled", "band").map_or(BAND, |v| v as u32);
    let block_m = tree.knobs.get("tiled", "block_m");
    let block_n = tree.knobs.get("tiled", "block_n");
    let matches_block = |s: &Schedule| {
        block_m.is_none_or(|m| s.block.0 == m) && block_n.is_none_or(|n| s.block.1 == n)
    };
    // Each schedule is one way, so the cheapest wins.
    let draft = |product: &Product, reads: &[Read], (choice, schedule): (usize, &Schedule)| {
        let tiled = Tiled {
            product: product.clone(),
            schedule: *schedule,
            parts: 0,
            band,
        };
        Draft::new(tiled, reads.to_vec(), product.outputs()).choice(choice)
    };
    let mut drafts: Vec<Draft> = (SCHEDULES.iter().enumerate())
        .filter(|(_, s)| matches_block(s))
        .map(|c| draft(&product, &reads, c))
        .collect();
    let direct_x = bf16_fits(product.x_form, product.m * product.k);
    let direct_w = bf16_fits(product.form, product.n * product.k);
    // bf16 weights multiply as stored, against `x` split exactly.
    if direct_w && product.k.is_multiple_of(WIDE as usize) {
        for (i, schedule) in PARTS.iter().enumerate() {
            if !matches_block(schedule) {
                continue;
            }
            let tiled = Tiled {
                product: product.clone(),
                schedule: *schedule,
                parts: SLABS,
                band,
            };
            let choice = 2 * SCHEDULES.len() + i;
            drafts.push(Draft::new(tiled, reads.to_vec(), product.outputs()).choice(choice));
        }
    }
    // One-pass bf16 tiles when precision allows or both are bf16.
    let aligned = if direct_x && direct_w {
        product.k.is_multiple_of(2)
    } else {
        product.k.is_multiple_of(WIDE as usize)
    };
    if aligned && (tree.compute_at(at) == Compute::Bf16 || (direct_x && direct_w)) {
        let base = 2 * SCHEDULES.len() + PARTS.len();
        for (i, schedule) in BF16.iter().enumerate() {
            if !matches_block(schedule) {
                continue;
            }
            let tiled = Tiled {
                product: product.clone(),
                schedule: *schedule,
                parts: 1,
                band,
            };
            drafts.push(Draft::new(tiled, reads.to_vec(), product.outputs()).choice(base + i));
        }
    }
    // Neighbours gathered while staged: the nine-fold input is never made.
    if let Some((near, reads)) = gathered(tree, &product, &reads) {
        let product = Product {
            x_form: plain(tree.node(near.src).form),
            near: Some(near),
            ..product.clone()
        };
        let ways = (SCHEDULES.iter().enumerate())
            .filter(|(_, s)| matches_block(s))
            .map(|(i, s)| (SCHEDULES.len() + i, s));
        drafts.extend(ways.map(|c| draft(&product, &reads, c)));
    }
    (!drafts.is_empty()).then_some(drafts)
}

/// The input's neighbours, and reads that skip the gather.
fn gathered(tree: &Tree, product: &Product, reads: &[Read]) -> Option<(Near, Vec<Read>)> {
    let Read::Node(x) = reads[0] else {
        return None;
    };
    let near: Near = tree.site(x)?;
    let fits = |s: &Schedule| near.cin.is_multiple_of(s.block.2);
    if near.taps != TAPS as usize || !SCHEDULES.iter().all(fits) || product.picked.is_some() {
        return None;
    }
    let mut reads = reads.to_vec();
    reads[0] = Read::Node(near.src);
    reads.insert(2, Read::Node(near.ids));
    Some((near, reads))
}

tessera_lower::rule!("tiled", propose);
