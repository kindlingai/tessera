//! A matmul on the matrix unit, staged through group memory.

use crate::lanes::{After, finishing};
use codegen::form::F32_FORM;
use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{add_operands, operands, stage_stmts};
use lower::sites;
use lower::{
    Arg, Draft, Form as Stored, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree,
};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;

/// How a block of the output is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Rows, columns and depth of one group's block.
    pub block: (usize, usize, usize),
    /// How SIMD groups split the block's rows and columns.
    pub simds: (usize, usize),
    /// Its cost curve's name. The weight form adds a suffix.
    pub name: &'static str,
}

/// Every matmul's schedules. `w16` SIMD groups own 16 by 16.
pub const SCHEDULES: [Schedule; 6] = [
    Schedule {
        block: (64, 64, 32),
        simds: (2, 2),
        name: "tiled_64x64",
    },
    Schedule {
        block: (32, 64, 32),
        simds: (1, 2),
        name: "tiled_32x64",
    },
    Schedule {
        block: (128, 64, 32),
        simds: (4, 2),
        name: "tiled_128x64",
    },
    Schedule {
        block: (64, 128, 32),
        simds: (2, 4),
        name: "tiled_64x128",
    },
    Schedule {
        block: (64, 64, 32),
        simds: (4, 4),
        name: "tiled_64x64_w16",
    },
    Schedule {
        block: (32, 32, 32),
        simds: (2, 2),
        name: "tiled_32x32_w16",
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

thread_local! {
    /// The kernel per schedule, form and staging, built once.
    static TREES: RefCell<HashMap<(usize, &'static str, bool), Body>> =
        RefCell::new(HashMap::new());
}

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

/// Stages a block of `x` and `w`, then multiplies.
#[kernel]
fn tiled(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    #[build] bm: u32,
    #[build] bn: u32,
    #[build] bk: u32,
    #[build] wm: u32,
    #[build] wn: u32,
    #[build] simds_n: u32,
    #[build] threads: u32,
    #[build] wide: bool,
    #[build] form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    let group = gid() / threads;
    let row0 = (group / ((n + bn - 1) / bn)) * bm;
    let col0 = (group % ((n + bn - 1) / bn)) * bn;
    let wm0 = (simd / simds_n) * wm;
    let wn0 = (simd % simds_n) * wn;
    let mut a_s = shared::<f32>(bm * (bk + PAD));
    let mut b_s = shared::<f32>(bn * (bk + PAD));
    let mut acc = tiles(wm, wn);
    for k0 in (0..k).by(bk) {
        stage(&mut a_s, x, row0, k0, m, k, bm, bk, threads, wide, F32_FORM);
        stage(&mut b_s, w, col0, k0, n, k, bn, bk, threads, wide, form);
        barrier();
        acc.block(&a_s, wm0 * (bk + PAD), &b_s, wn0 * (bk + PAD), bk + PAD, bk);
        barrier();
    }
}

/// `x[m, k]` times `w[n, k]` transposed, by tiles.
#[derive(Clone, Debug)]
pub struct Tiled {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub schedule: Schedule,
    pub after: After,
}

impl Tiled {
    fn groups(&self) -> (usize, usize) {
        let (bm, bn, _) = self.schedule.block;
        (self.m.div_ceil(bm), self.n.div_ceil(bn))
    }

    fn others(&self) -> usize {
        match self.after.stages.is_empty() {
            true => 0,
            false => operands(&self.after.stages),
        }
    }

    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let wide = self.k.is_multiple_of(WIDE as usize);
        let build = || {
            let s = &self.schedule;
            let (bm, bn, bk) = (s.block.0 as u32, s.block.1 as u32, s.block.2 as u32);
            let (wm, wn) = (s.simd().0 as u32, s.simd().1 as u32);
            let (simds_n, threads) = (s.simds.1 as u32, s.threads() as u32);
            tiled_ir(bm, bn, bk, wm, wn, simds_n, threads, wide, self.form)
        };
        let at = SCHEDULES.iter().position(|s| *s == self.schedule);
        let key = (at.expect("a listed schedule"), self.form.name(), wide);
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        add_operands(&mut body, 2, self.others());
        body.stmts.push(self.stores());
        body
    }

    /// Each sum through the stages, into its place.
    fn stores(&self) -> Stmt {
        let After { stages, sum } = &self.after;
        let u64 = |e| Expr::cast(Ty::U64, e);
        let add = |a, b| Expr::bin(BinOp::Add, a, b);
        let gr = add(add(Expr::var("row0"), Expr::var("wm0")), Expr::var("r"));
        let gc = add(add(Expr::var("col0"), Expr::var("wn0")), Expr::var("c"));
        let o = Expr::bin(BinOp::Mul, u64(Expr::var("gr")), u64(Expr::var("n")));
        let o = add(o, u64(Expr::var("gc")));
        // The chain's reads in order, the sum among them.
        let read = |j: usize| match j.cmp(sum) {
            Ordering::Equal => Expr::var("s"),
            Ordering::Less => Expr::index(Expr::var(&format!("a{j}")), o.clone()),
            Ordering::Greater => Expr::index(Expr::var(&format!("a{}", j - 1)), o.clone()),
        };
        // The stages run on `v`, so the sum moves first.
        let mut then = vec![
            Stmt::Let {
                name: "s".into(),
                init: Expr::var("v"),
            },
            Stmt::Assign {
                target: Expr::var("v"),
                op: None,
                value: read(0),
            },
        ];
        then.extend(stage_stmts(stages, &|j| read(j + 1)));
        then.push(Stmt::Assign {
            target: Expr::index(Expr::var("out"), o.clone()),
            op: None,
            value: Expr::var("v"),
        });
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

impl Kernel for Tiled {
    fn name(&self) -> &'static str {
        self.form.suffixed(self.schedule.name)
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// Each device's tiles sum in their own order.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m as f64, self.n as f64, self.k as f64);
        let (gm, gn) = self.groups();
        let (_, _, bk) = self.schedule.block;
        // Each operand is re-read once per group across it.
        let mut reads = vec![
            4.0 * m * k * gn as f64,
            self.form.bytes(self.n * self.k) as f64 * gm as f64,
        ];
        reads.extend((0..self.others()).map(|_| 4.0 * m * n));
        Some(Load {
            threads: gm * gn * self.schedule.threads(),
            group: self.schedule.threads(),
            trips: (self.k.div_ceil(bk)) as f64,
            trip: self.tree().trip(),
            reads,
            written: 4.0 * m * n,
            flight: 16.0,
        })
    }

    fn source(&self) -> Source {
        let mut body = self.tree();
        body.entry = format!("tiled_{}_{:x}", self.form.name(), hash(&body.text().body));
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let (gm, gn) = self.groups();
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.m, self.n, self.k].map(Arg::count));
        vec![Launch {
            args,
            shape: Shape::Grid {
                across: gm * gn,
                down: 1,
                threads: self.schedule.threads(),
            },
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }

    /// Two staged slabs of `bk + PAD` floats a row.
    fn needs(&self) -> Needs {
        let (bm, bn, bk) = self.schedule.block;
        Needs {
            lanes: LANES,
            group_bytes: 4 * (bm + bn) * (bk + PAD as usize),
            group_threads: self.schedule.threads(),
            ..Needs::default()
        }
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, after, others) = finishing(tree, at)?;
    let sites::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = product;
    // Each schedule is one way, so the cheapest wins.
    let draft = |(choice, schedule): (usize, &Schedule)| {
        let tiled = Tiled {
            m,
            n,
            k,
            form,
            schedule: *schedule,
            after: after.clone(),
        };
        let reads = [Read::Node(x), Read::Node(w)].into_iter();
        Draft::new(tiled, reads.chain(others.iter().copied()), m * n).choice(choice)
    };
    Some(SCHEDULES.iter().enumerate().map(draft).collect())
}

lower::rule!("tiled", propose);
