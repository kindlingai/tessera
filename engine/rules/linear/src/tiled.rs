//! A matmul on the matrix unit, staged through group memory.

use crate::product::{Product, Variant, place};
use sensai_codegen::form::F32_FORM;
use sensai_codegen::ir::{BinOp, Body, Expr, Stmt};
use sensai_codegen::prelude::*;
use sensai_graph::Order;
use sensai_lower::form::Load as Stored;
use sensai_lower::kernels::finish;
use sensai_lower::sites::Near;
use sensai_lower::{Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Tree};

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
    Schedule {
        block: (64, 64, 32),
        simds: (2, 2),
        name: "tiled_64x64",
        near: "near_64x64",
    },
    Schedule {
        block: (32, 64, 32),
        simds: (1, 2),
        name: "tiled_32x64",
        near: "near_32x64",
    },
    Schedule {
        block: (128, 64, 32),
        simds: (4, 2),
        name: "tiled_128x64",
        near: "near_128x64",
    },
    Schedule {
        block: (64, 128, 32),
        simds: (2, 4),
        name: "tiled_64x128",
        near: "near_64x128",
    },
    Schedule {
        block: (64, 64, 32),
        simds: (4, 4),
        name: "tiled_64x64_w16",
        near: "near_64x64_w16",
    },
    Schedule {
        block: (32, 32, 32),
        simds: (2, 2),
        name: "tiled_32x32_w16",
        near: "near_32x32_w16",
    },
    Schedule {
        block: (64, 64, 32),
        simds: (2, 4),
        name: "tiled_64x64_w32x16",
        near: "near_64x64_w32x16",
    },
];

impl Schedule {
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

/// A tiled kernel. The caller names what its staging reads.
macro_rules! tiled_kernel {
    (
        $(#[$doc:meta])* $name:ident,
        [$x:ident, $a:ident, $row0:ident, $k0:ident, $m:ident, $k:ident],
        [$bm:ident, $bk:ident, $threads:ident, $wide:ident],
        [$($bufs:tt)*],
        [$($scalars:tt)*],
        [$($stage:tt)*]
    ) => {
        $(#[$doc])*
        #[kernel]
        fn $name(
            $x: Buf<F32>,
            #[form(form)] w: Buf<Dyn>,
            $($bufs)*
            out: BufMut<F32>,
            $m: u32,
            n: u32,
            $k: u32,
            $($scalars)*
            #[build] $bm: u32,
            #[build] bn: u32,
            #[build] $bk: u32,
            #[build] wm: u32,
            #[build] wn: u32,
            #[build] simds_n: u32,
            #[build] $threads: u32,
            #[build] $wide: bool,
            #[build] form: &'static dyn Stored,
        ) {
            let simd = lid() / LANES as u32;
            let group = gid() / $threads;
            let $row0 = (group / ((n + bn - 1) / bn)) * $bm;
            let col0 = (group % ((n + bn - 1) / bn)) * bn;
            let wm0 = (simd / simds_n) * wm;
            let wn0 = (simd % simds_n) * wn;
            let mut $a = shared::<f32>($bm * ($bk + PAD));
            let mut b_s = shared::<f32>(bn * ($bk + PAD));
            let mut acc = tiles(wm, wn);
            for $k0 in (0..$k).by($bk) {
                $($stage)*;
                stage(&mut b_s, w, col0, $k0, n, $k, bn, $bk, $threads, $wide, form);
                barrier();
                acc.block(&$a, wm0 * ($bk + PAD), &b_s, wn0 * ($bk + PAD), $bk + PAD, $bk);
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
    [],
    [],
    [stage(&mut a_s, x, row0, k0, m, k, bm, bk, threads, wide, F32_FORM)]
);

tiled_kernel!(
    /// `tiled`, its rows each pixel's neighbours, gathered while staged.
    tiled_near,
    [x, a_s, row0, k0, m, k],
    [bm, bk, threads, wide],
    [ids: Buf<U32>,],
    [pixels: u32, cin: u32,],
    [stage_near(&mut a_s, x, ids, row0, k0, m, pixels, cin, bm, bk, threads, F32_FORM)]
);

/// `x[m, k]` times `w[n, k]` transposed, by tiles.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed(self.name()),
    sizes = self.product.sizes(),
    work = self.product.work(),
    accepts = self.product.accepts(input, form),
    needs = tiled_needs(&self.schedule),
    // Each device's tiles sum in their own order.
    order = Order::PerConfig,
    source = self.product.source(self.tree(), "tiled"),
    launches = tiled_launches,
    load = tiled_load,
    unproven,
)]
pub struct Tiled {
    pub product: Product,
    pub schedule: Schedule,
}

impl Tiled {
    fn name(&self) -> &'static str {
        match self.product.near {
            Some(_) => self.schedule.near,
            None => self.schedule.name,
        }
    }

    fn groups(&self) -> (usize, usize) {
        let (bm, bn, _) = self.schedule.block;
        (self.product.m.div_ceil(bm), self.product.n.div_ceil(bn))
    }

    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let Product { k, form, .. } = self.product;
        let wide = k.is_multiple_of(WIDE as usize);
        let near = self.product.near.is_some();
        let build = || {
            let s = &self.schedule;
            let (bm, bn, bk) = (s.block.0 as u32, s.block.1 as u32, s.block.2 as u32);
            let (wm, wn) = (s.simd().0 as u32, s.simd().1 as u32);
            let (simds_n, threads) = (s.simds.1 as u32, s.threads() as u32);
            match near {
                true => tiled_near_ir(bm, bn, bk, wm, wn, simds_n, threads, wide, form),
                false => tiled_ir(bm, bn, bk, wm, wn, simds_n, threads, wide, form),
            }
        };
        let at = SCHEDULES.iter().position(|s| *s == self.schedule);
        let kind = if near { "tiled_near" } else { "tiled" };
        let key: Variant = (kind, at.expect("a listed schedule"), form.name(), wide);
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

fn tiled_load(t: &Tiled, _: usize) -> Option<Load> {
    let Product { m, n, k, form, .. } = t.product;
    let (mf, nf) = (m as f64, n as f64);
    let (gm, gn) = t.groups();
    let (_, _, bk) = t.schedule.block;
    // Neighbouring groups share `x` rows. `w` streams per row.
    let rows = match t.product.near {
        Some(near) => 4.0 * mf * near.cin as f64,
        None => 4.0 * mf * k as f64,
    };
    let mut reads = vec![rows];
    reads.extend((0..gm).map(|_| form.bytes(n * k) as f64));
    reads.extend(
        t.product
            .near
            .map(|near| (4 * near.pixels * near.taps) as f64),
    );
    reads.extend((0..t.product.after.others()).map(|_| 4.0 * mf * nf));
    Some(Load {
        threads: gm * gn * t.schedule.threads(),
        group: t.schedule.threads(),
        trips: (k.div_ceil(bk)) as f64,
        trip: t.tree().trip(),
        reads,
        written: 4.0 * mf * nf,
        flight: 16.0,
        ..Load::default()
    })
}

fn tiled_launches(t: &Tiled, step: &Step) -> Vec<Launch> {
    let Product { m, n, k, .. } = t.product;
    let (gm, gn) = t.groups();
    let shape = Shape::Grid {
        across: gm * gn,
        down: 1,
        threads: t.schedule.threads(),
    };
    match t.product.near {
        Some(near) => t
            .product
            .launch(step, &[m, n, k, near.pixels, near.cin], shape),
        None => t.product.launch(step, &[m, n, k], shape),
    }
}

/// Two staged slabs of `bk + PAD` floats a row.
fn tiled_needs(schedule: &Schedule) -> Needs {
    let (bm, bn, bk) = schedule.block;
    Needs {
        lanes: LANES,
        group_bytes: 4 * (bm + bn) * (bk + PAD as usize),
        group_threads: schedule.threads(),
        tile: true,
        ..Needs::default()
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    if product.picked.is_some() {
        return None;
    }
    // Each schedule is one way, so the cheapest wins.
    let draft = |product: &Product, reads: &[Read], (choice, schedule): (usize, &Schedule)| {
        let tiled = Tiled {
            product: product.clone(),
            schedule: *schedule,
        };
        Draft::new(tiled, reads.to_vec(), product.outputs()).choice(choice)
    };
    let mut drafts: Vec<Draft> = (SCHEDULES.iter().enumerate())
        .map(|c| draft(&product, &reads, c))
        .collect();
    // Neighbours gathered while staged: the nine-fold input is never made.
    if let Some((near, reads)) = gathered(tree, &product, &reads) {
        let product = Product {
            near: Some(near),
            ..product.clone()
        };
        let ways = SCHEDULES
            .iter()
            .enumerate()
            .map(|(i, s)| (SCHEDULES.len() + i, s));
        drafts.extend(ways.map(|c| draft(&product, &reads, c)));
    }
    Some(drafts)
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

sensai_lower::rule!("tiled", propose);
