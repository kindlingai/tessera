//! A matmul on the matrix unit, staged through group memory.

use codegen::{Gen, LANES, hash};
use graph::{DType, Order};
use lower::{Arg, Draft, Kernel, Launch, Load, Shape, Site, Source, Step, Stored, T, Tree};

/// How a block of the output is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// Rows, columns and depth of one group's block.
    pub block: (usize, usize, usize),
    /// How the group's warps split the block's rows and columns.
    pub warps: (usize, usize),
}

impl Schedule {
    /// Four warps on a 64 by 64 by 32 block.
    pub const DEFAULT: Schedule = Schedule {
        block: (64, 64, 32),
        warps: (2, 2),
    };

    fn threads(&self) -> usize {
        self.warps.0 * self.warps.1 * LANES
    }

    /// The rows and columns one warp owns.
    fn warp(&self) -> (usize, usize) {
        (self.block.0 / self.warps.0, self.block.1 / self.warps.1)
    }
}

/// `x[m, k]` times `w[n, k]` transposed, by tiles.
#[derive(Clone, Copy, Debug)]
pub struct Tiled {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: DType,
    pub schedule: Schedule,
}

/// Floats of padding after each staged row, against bank conflicts.
const PAD: usize = 4;

impl Tiled {
    fn groups(&self) -> (usize, usize) {
        let (bm, bn, _) = self.schedule.block;
        (self.m.div_ceil(bm), self.n.div_ceil(bn))
    }

    /// Stages a block of `x` and `w`, multiplies, then stores.
    fn write(&self) -> Gen {
        let (bm, bn, bk) = self.schedule.block;
        let (wm, wn) = self.schedule.warp();
        let (threads, stride) = (self.schedule.threads(), bk + PAD);
        let mut g = Gen::new("tiled");
        g.floats("x", false);
        g.buffer("w", self.form.ty(), false);
        g.floats("out", true);
        g.scalar("m", "u32");
        g.scalar("n", "u32");
        g.scalar("k", "u32");
        let lane = g.lane();
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
        let load_w = self.form.load("w", "(u64)(col0 + r) * k + k0 + c");
        g.repeat("k0", "0", "k", &bk.to_string(), 1, |g| {
            // Threads fill both slabs, zero past the edges.
            g.line(format!(
                "for (u32 i = lid; i < {bm} * {bk}; i += {threads}) {{"
            ));
            g.line(format!("    u32 r = i / {bk}, c = i % {bk};"));
            g.line("    bool in = row0 + r < m && k0 + c < k;");
            g.line(format!(
                "    As[r * {stride} + c] = in ? x[(u64)(row0 + r) * k + k0 + c] : 0.0f;"
            ));
            g.line("}");
            g.line(format!(
                "for (u32 i = lid; i < {bn} * {bk}; i += {threads}) {{"
            ));
            g.line(format!("    u32 r = i / {bk}, c = i % {bk};"));
            g.line("    bool in = col0 + r < n && k0 + c < k;");
            g.line(format!("    Bs[r * {stride} + c] = in ? {load_w} : 0.0f;"));
            g.line("}");
            g.line("f_barrier();");
            let a = format!("As + wm * {stride}");
            let b = format!("Bs + wn * {stride}");
            g.tile_block("acc", &a, &b, &stride.to_string(), [wm, wn, bk]);
            g.line("f_barrier();");
        });
        g.tile_each("acc", wm, wn, |g| {
            g.line("u32 gr = row0 + wm + r, gc = col0 + wn + c;");
            g.line("if (gr < m && gc < n) {");
            g.line("    out[(u64)gr * n + gc] = v;");
            g.line("}");
        });
        let _ = lane;
        g
    }
}

impl Kernel for Tiled {
    fn name(&self) -> &'static str {
        match self.form {
            DType::Bf16 => "tiled_bf16",
            _ => "tiled",
        }
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
        Some(Load {
            threads: gm * gn * self.schedule.threads(),
            group: self.schedule.threads(),
            trips: (self.k.div_ceil(bk)) as f64,
            trip: self.write().trip(),
            // Each operand is re-read once per group across it.
            reads: vec![
                4.0 * m * k * gn as f64,
                self.form.bytes() as f64 * n * k * gm as f64,
            ],
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
        let args = [step.read(0), step.read(1), step.out()];
        let sizes = [self.m, self.n, self.k].map(Arg::count);
        vec![Launch {
            args: args.into_iter().chain(sizes).collect(),
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
    let Site::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = tree.site(at)?
    else {
        return None;
    };
    let tiled = Tiled {
        m,
        n,
        k,
        form,
        schedule: Schedule::DEFAULT,
    };
    Some(vec![Draft::new(tiled, [x, w], m * n)])
}

lower::rule!("tiled", propose);
