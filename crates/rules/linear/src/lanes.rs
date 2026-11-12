//! A matmul for few rows, with threads sharing each sum.

use graph::Order;
use lower::kernels::{MapStage, operands, stage_lines};
use lower::{
    Arg, DType, Draft, Kernel, Launch, Load, Ops, Read, Shape, Site, Source, Step, Stored, T, Text,
    Tree, float, hash,
};
use std::cmp::Ordering;

/// Inputs one lane multiplies and adds by itself.
const CHUNK: usize = 16;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// The most lane sums one launch may write.
const ROOM: usize = 1 << 26;

/// Each lane's sum over its share of `k`.
#[derive(Clone, Copy, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lanes: usize,
    pub form: DType,
}

/// Elementwise stages a sum runs through before it is stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct After {
    pub stages: Vec<MapStage>,
    /// Which of the chain's reads the sum stands in for.
    pub sum: usize,
}

/// Each output's lane sums, added in lane order, then finished.
#[derive(Clone, Debug)]
pub struct Sums {
    pub count: usize,
    pub lanes: usize,
    pub after: After,
}

/// How many lanes share a sum over `k` inputs.
pub fn lanes(k: usize) -> usize {
    (k / CHUNK).clamp(1, 256)
}

// Lanes are whole runs of `k`, so rows sum alike.
const HEAD: &str = "    if (gid >= n * lanes) {
        return;
    }
    u32 lane = gid % lanes;
    u32 col = gid / lanes;
    u32 from = (u32)((u64)k * lane / lanes);
    u32 to = (u32)((u64)k * (lane + 1) / lanes);
    u64 at = (u64)col * k;
";

/// One row keeps its sum in a register.
fn one(load: &str) -> String {
    format!(
        "{HEAD}    float acc = 0.0f;
    for (u32 i = from; i < to; i++) {{
        acc += x[i] * {load};
    }}
    out[gid] = acc;
"
    )
}

/// Several rows share each weight as it is read.
fn several(load: &str) -> String {
    format!(
        "{HEAD}    float sums[{ROWS}];
    for (u32 row = 0; row < m; row++) {{
        sums[row] = 0.0f;
    }}
    for (u32 i = from; i < to; i++) {{
        float c = {load};
        for (u32 row = 0; row < m; row++) {{
            sums[row] += x[(u64)row * k + i] * c;
        }}
    }}
    for (u32 row = 0; row < m; row++) {{
        out[((u64)row * n + col) * lanes + lane] = sums[row];
    }}
"
    )
}

const SUMS: &str = "    if (gid >= count) {
        return;
    }
    u64 at = (u64)gid * lanes;
    float acc = 0.0f;
    for (u32 l = 0; l < lanes; l++) {
        acc += parts[at + l];
    }
";

impl Kernel for Lanes {
    fn name(&self) -> &'static str {
        match self.form {
            DType::Bf16 => "lanes_bf16",
            _ => "lanes",
        }
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// One row and several rows may round differently.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    /// Each thread runs along its own stretch of a row.
    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m as f64, self.n as f64, self.k as f64);
        let lanes = self.lanes as f64;
        let trip = Ops {
            float: m,
            int: 2.0,
            loads: 1.0 + m,
            fresh: 1.0 / 16.0,
            ..Ops::default()
        };
        Some(Load {
            threads: self.n * self.lanes,
            group: 256,
            trips: k / lanes,
            trip,
            reads: self.form.matmul_bytes(self.m, self.n, self.k),
            written: 4.0 * m * n * lanes,
            flight: 64.0,
        })
    }

    fn source(&self) -> Source {
        let load = self.form.load("w", "at + i");
        let w = ("w".to_string(), self.form.ty(), false);
        let (rows, body) = match self.m {
            1 => ("row", one(&load)),
            _ => ("rows", several(&load)),
        };
        Source::Made(Text {
            entry: format!("linear_{rows}_{}", self.form.name()),
            buffers: vec![float("x", false), w, float("out", true)],
            scalars: vec![("m", "u32"), ("n", "u32"), ("k", "u32"), ("lanes", "u32")],
            body,
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.m, self.n, self.k, self.lanes].map(Arg::count);
        let buffers = [step.read(0), step.read(1), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.n * self.lanes),
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

impl Kernel for Sums {
    fn name(&self) -> &'static str {
        "sums"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.count, self.lanes]
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (count, lanes) = (self.count as f64, self.lanes as f64);
        let trip = Ops {
            float: 1.0,
            int: 1.0,
            loads: 1.0,
            fresh: 1.0 / 16.0,
            ..Ops::default()
        };
        Some(Load {
            threads: self.count,
            group: 256,
            trips: lanes,
            trip,
            reads: vec![4.0 * count * lanes],
            written: 4.0 * count,
            flight: 64.0,
        })
    }

    fn source(&self) -> Source {
        let After { stages, sum } = &self.after;
        let others = if stages.is_empty() {
            0
        } else {
            operands(stages)
        };
        // The chain's reads in order, the sum among them.
        let read = |k: usize| match k.cmp(sum) {
            Ordering::Equal => "acc".to_string(),
            Ordering::Less => format!("a{k}[gid]"),
            Ordering::Greater => format!("a{}[gid]", k - 1),
        };
        let (entry, body) = match stages.is_empty() {
            true => ("linear_sums".into(), format!("{SUMS}    out[gid] = acc;\n")),
            false => {
                let lines = stage_lines(stages, &|k| read(k + 1));
                let first = read(0);
                let body = format!("{SUMS}    float v = {first};\n{lines}    out[gid] = v;\n");
                (format!("linear_sums_{:x}", hash(&body)), body)
            }
        };
        let mut buffers = vec![float("parts", false)];
        buffers.extend((0..others).map(|k| float(&format!("a{k}"), false)));
        buffers.push(float("out", true));
        Source::Made(Text {
            entry,
            buffers,
            scalars: vec![("count", "u32"), ("lanes", "u32")],
            body,
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.count, self.lanes].map(Arg::count);
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        vec![Launch {
            args: args.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.count),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.count * self.lanes) as f64
    }
}

/// A matmul, or stages finishing a matmul with one reader.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, after, others) = match tree.site(at)? {
        product @ Site::Matmul { .. } => (product, After::default(), Vec::new()),
        Site::Map { stages, reads, .. } => {
            let lone = |r: &T| match tree.site(*r) {
                Some(product @ Site::Matmul { .. }) if !tree.root(*r) => Some(product),
                _ => None,
            };
            let sum = reads.iter().position(|r| lone(r).is_some())?;
            let others = reads.iter().filter(|r| **r != reads[sum]);
            let others = others.map(|r| Read::Node(*r)).collect();
            (lone(&reads[sum])?, After { stages, sum }, others)
        }
        _ => return None,
    };
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
    let lanes = lanes(k);
    if m > ROWS || m * n * lanes > ROOM {
        return None;
    }
    let parts = Lanes {
        m,
        n,
        k,
        lanes,
        form,
    };
    let count = m * n;
    let sums = Sums {
        count,
        lanes,
        after,
    };
    let reads = [Read::Draft(0)].into_iter().chain(others);
    Some(vec![
        Draft::new(parts, [x, w], count * lanes),
        Draft::new(sums, reads, count),
    ])
}

lower::rule!("lanes", propose);
