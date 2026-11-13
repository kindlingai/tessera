//! A matmul on the matrix unit, staged through group memory.

use crate::lanes::{After, finishing};
use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Param, Stmt, Ty};
use codegen::prelude::*;
use graph::{DType, Order};
use lower::kernels::{operands, stage_stmts};
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, Stored, T, Tree};
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
    /// Its cost curves' names, for f32 and bf16 weights.
    pub names: [&'static str; 2],
}

/// Every matmul's schedules. `w16` SIMD groups own 16 by 16.
pub const SCHEDULES: [Schedule; 6] = [
    Schedule {
        block: (64, 64, 32),
        simds: (2, 2),
        names: ["tiled_64x64", "tiled_64x64_bf16"],
    },
    Schedule {
        block: (32, 64, 32),
        simds: (1, 2),
        names: ["tiled_32x64", "tiled_32x64_bf16"],
    },
    Schedule {
        block: (128, 64, 32),
        simds: (4, 2),
        names: ["tiled_128x64", "tiled_128x64_bf16"],
    },
    Schedule {
        block: (64, 128, 32),
        simds: (2, 4),
        names: ["tiled_64x128", "tiled_64x128_bf16"],
    },
    Schedule {
        block: (64, 64, 32),
        simds: (4, 4),
        names: ["tiled_64x64_w16", "tiled_64x64_w16_bf16"],
    },
    Schedule {
        block: (32, 32, 32),
        simds: (2, 2),
        names: ["tiled_32x32_w16", "tiled_32x32_w16_bf16"],
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
    static TREES: RefCell<HashMap<(usize, bool, bool), Body>> = RefCell::new(HashMap::new());
}

/// Stages a block of `x` and `w`, then multiplies.
#[kernel]
fn tiled<F: Form<Elem = f32>>(
    x: Buf<F32>,
    w: Buf<F>,
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
        if wide {
            // Groups of four along `k` need one test per load.
            for i in (lid()..bm * (bk / WIDE)).by(threads) {
                let r = i / (bk / WIDE);
                let c = (i % (bk / WIDE)) * WIDE;
                let at = r * (bk + PAD) + c;
                if row0 + r < m && k0 + c < k {
                    let v = x.load4((row0 + r) as u64 * k as u64 + (k0 + c) as u64);
                    a_s[at] = v.x;
                    a_s[at + 1] = v.y;
                    a_s[at + 2] = v.z;
                    a_s[at + 3] = v.w;
                } else {
                    a_s[at] = 0.0;
                    a_s[at + 1] = 0.0;
                    a_s[at + 2] = 0.0;
                    a_s[at + 3] = 0.0;
                }
            }
            for i in (lid()..bn * (bk / WIDE)).by(threads) {
                let r = i / (bk / WIDE);
                let c = (i % (bk / WIDE)) * WIDE;
                let at = r * (bk + PAD) + c;
                if col0 + r < n && k0 + c < k {
                    let v = w.load4((col0 + r) as u64 * k as u64 + (k0 + c) as u64);
                    b_s[at] = v.x;
                    b_s[at + 1] = v.y;
                    b_s[at + 2] = v.z;
                    b_s[at + 3] = v.w;
                } else {
                    b_s[at] = 0.0;
                    b_s[at + 1] = 0.0;
                    b_s[at + 2] = 0.0;
                    b_s[at + 3] = 0.0;
                }
            }
        } else {
            for i in (lid()..bm * bk).by(threads) {
                let r = i / bk;
                let c = i % bk;
                if row0 + r < m && k0 + c < k {
                    a_s[r * (bk + PAD) + c] = x[(row0 + r) as u64 * k as u64 + (k0 + c) as u64];
                } else {
                    a_s[r * (bk + PAD) + c] = 0.0;
                }
            }
            for i in (lid()..bn * bk).by(threads) {
                let r = i / bk;
                let c = i % bk;
                if col0 + r < n && k0 + c < k {
                    b_s[r * (bk + PAD) + c] = w[(col0 + r) as u64 * k as u64 + (k0 + c) as u64];
                } else {
                    b_s[r * (bk + PAD) + c] = 0.0;
                }
            }
        }
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
    pub form: DType,
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
            match self.form {
                DType::Bf16 => tiled_ir::<Bf16>(bm, bn, bk, wm, wn, simds_n, threads, wide),
                _ => tiled_ir::<F32>(bm, bn, bk, wm, wn, simds_n, threads, wide),
            }
        };
        let at = SCHEDULES.iter().position(|s| *s == self.schedule);
        let key = (
            at.expect("a listed schedule"),
            self.form == DType::Bf16,
            wide,
        );
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        let operand = |a: usize| Param::Buffer {
            name: format!("a{a}"),
            dtype: DType::F32,
            writes: false,
        };
        // The operands sit between the weights and the output.
        body.params.splice(2..2, (0..self.others()).map(operand));
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
        self.schedule.names[(self.form == DType::Bf16) as usize]
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
            self.form.bytes() as f64 * n * k * gm as f64,
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

    fn accepts(&self, input: usize, form: DType) -> bool {
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
