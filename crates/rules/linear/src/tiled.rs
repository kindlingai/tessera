//! A matmul on the matrix unit, staged through group memory.

use crate::lanes::{After, finishing};
use codegen::{Gen, LANES, hash};
use graph::{DType, Order};
use lower::kernels::{operands, stage_lines};
use lower::{Arg, Draft, Kernel, Launch, Load, Read, Shape, Site, Source, Step, Stored, T, Tree};
use std::cmp::Ordering;

/// How a block of the output is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Rows, columns and depth of one group's block.
    pub block: (usize, usize, usize),
    /// How the group's warps split the block's rows and columns.
    pub warps: (usize, usize),
    /// Its cost curves' names, for f32 and bf16 weights.
    pub names: [&'static str; 2],
}

/// Every matmul's schedules. `w16` warps own 16 by 16.
pub const SCHEDULES: [Schedule; 6] = [
    Schedule {
        block: (64, 64, 32),
        warps: (2, 2),
        names: ["tiled_64x64", "tiled_64x64_bf16"],
    },
    Schedule {
        block: (32, 64, 32),
        warps: (1, 2),
        names: ["tiled_32x64", "tiled_32x64_bf16"],
    },
    Schedule {
        block: (128, 64, 32),
        warps: (4, 2),
        names: ["tiled_128x64", "tiled_128x64_bf16"],
    },
    Schedule {
        block: (64, 128, 32),
        warps: (2, 4),
        names: ["tiled_64x128", "tiled_64x128_bf16"],
    },
    Schedule {
        block: (64, 64, 32),
        warps: (4, 4),
        names: ["tiled_64x64_w16", "tiled_64x64_w16_bf16"],
    },
    Schedule {
        block: (32, 32, 32),
        warps: (2, 2),
        names: ["tiled_32x32_w16", "tiled_32x32_w16_bf16"],
    },
];

impl Schedule {
    fn threads(&self) -> usize {
        self.warps.0 * self.warps.1 * LANES
    }

    /// The rows and columns one warp owns.
    fn warp(&self) -> (usize, usize) {
        (self.block.0 / self.warps.0, self.block.1 / self.warps.1)
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

/// Floats of padding after each staged row, against bank conflicts.
const PAD: usize = 4;

/// Elements one wide load brings.
const WIDE: usize = 4;

/// One operand's slab and where its rows come from.
struct Slab<'a> {
    slab: &'a str,
    src: &'a str,
    dtype: DType,
    first: &'a str,
    count: &'a str,
    rows: usize,
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

    /// Stages a block of `x` and `w`, multiplies, then stores.
    fn write(&self) -> Gen {
        let (bm, bn, bk) = self.schedule.block;
        let (wm, wn) = self.schedule.warp();
        let (threads, stride) = (self.schedule.threads(), bk + PAD);
        let mut g = Gen::new("tiled");
        g.floats("x", false);
        g.buffer("w", self.form.ty(), false);
        for a in 0..self.others() {
            g.floats(&format!("a{a}"), false);
        }
        g.floats("out", true);
        g.scalar("m", "u32");
        g.scalar("n", "u32");
        g.scalar("k", "u32");
        g.lane();
        g.line(format!("u32 warp = lid / {LANES};"));
        g.line(format!("u32 group = gid / {threads};"));
        g.line(format!(
            "u32 row0 = (group / ((n + {bn} - 1) / {bn})) * {bm};"
        ));
        g.line(format!(
            "u32 col0 = (group % ((n + {bn} - 1) / {bn})) * {bn};"
        ));
        g.line(format!(
            "u32 wm = (warp / {}) * {wm};",
            self.schedule.warps.1
        ));
        g.line(format!(
            "u32 wn = (warp % {}) * {wn};",
            self.schedule.warps.1
        ));
        g.line(format!("f_group(float, As, {});", bm * stride));
        g.line(format!("f_group(float, Bs, {});", bn * stride));
        g.tile_acc("acc", wm, wn);
        let a = Slab {
            slab: "As",
            src: "x",
            dtype: DType::F32,
            first: "row0",
            count: "m",
            rows: bm,
        };
        let b = Slab {
            slab: "Bs",
            src: "w",
            dtype: self.form,
            first: "col0",
            count: "n",
            rows: bn,
        };
        g.repeat("k0", "0", "k", &bk.to_string(), 1, |g| {
            self.stage(g, &a);
            self.stage(g, &b);
            g.line("f_barrier();");
            let a = format!("As + wm * {stride}");
            let b = format!("Bs + wn * {stride}");
            g.tile_block("acc", &a, &b, &stride.to_string(), [wm, wn, bk]);
            g.line("f_barrier();");
        });
        g.tile_each("acc", wm, wn, |g| {
            g.line("u32 gr = row0 + wm + r, gc = col0 + wn + c;");
            g.block("if (gr < m && gc < n)", |g| self.store(g));
        });
        g
    }

    /// Threads fill the slab's rows, zero past the edges.
    fn stage(&self, g: &mut Gen, s: &Slab) {
        let Slab {
            slab,
            src,
            dtype,
            first,
            count,
            rows,
        } = *s;
        let (_, _, bk) = self.schedule.block;
        let (threads, stride) = (self.schedule.threads(), bk + PAD);
        let at = format!("(u64)({first} + r) * k + k0 + c");
        if !self.k.is_multiple_of(WIDE) {
            g.line(format!(
                "for (u32 i = lid; i < {rows} * {bk}; i += {threads}) {{"
            ));
            g.line(format!("    u32 r = i / {bk}, c = i % {bk};"));
            g.line(format!(
                "    bool in = {first} + r < {count} && k0 + c < k;"
            ));
            let load = dtype.load(src, &at);
            g.line(format!(
                "    {slab}[r * {stride} + c] = in ? {load} : 0.0f;"
            ));
            g.line("}");
            return;
        }
        // Groups of four along `k` need one test per load.
        let per = bk / WIDE;
        g.line(format!(
            "for (u32 i = lid; i < {rows} * {per}; i += {threads}) {{"
        ));
        g.line(format!("    u32 r = i / {per}, c = (i % {per}) * {WIDE};"));
        g.block(
            &format!("    if ({first} + r < {count} && k0 + c < k)"),
            |g| {
                let v = g.four(dtype, &format!("{src} + {at}"));
                for (e, v) in v.iter().enumerate() {
                    g.line(format!("    {slab}[r * {stride} + c + {e}] = {v};"));
                }
            },
        );
        g.block("    else", |g| {
            for e in 0..WIDE {
                g.line(format!("    {slab}[r * {stride} + c + {e}] = 0.0f;"));
            }
        });
        g.line("}");
    }

    /// The sum `v` through the stages, into its place.
    fn store(&self, g: &mut Gen) {
        let After { stages, sum } = &self.after;
        let o = "((u64)gr * n + gc)";
        if stages.is_empty() {
            return g.line(format!("out[{o}] = v;"));
        }
        // The chain's reads in order, the sum among them.
        g.line("float s = v;");
        let read = |j: usize| match j.cmp(sum) {
            Ordering::Equal => "s".to_string(),
            Ordering::Less => format!("a{j}[{o}]"),
            Ordering::Greater => format!("a{}[{o}]", j - 1),
        };
        g.line(format!("{{ float v = {};", read(0)));
        for line in stage_lines(stages, &|j| read(j + 1)).lines() {
            g.line(line.trim_start().to_string());
        }
        g.line(format!("out[{o}] = v; }}"));
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
            trip: self.write().trip(),
            reads,
            written: 4.0 * m * n,
            flight: 16.0,
        })
    }

    fn source(&self) -> Source {
        let mut text = self.write().finish();
        text.entry = format!("tiled_{}_{:x}", self.form.name(), hash(&text.body));
        Source::Made(text)
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
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, after, others) = finishing(tree, at)?;
    let Site::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = product
    else {
        return None;
    };
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
