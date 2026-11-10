//! A matmul for few rows, with threads sharing each sum.

use graph::Order;
use lower::{Arg, Draft, Kernel, Launch, Read, Shape, Site, Source, Step, Text, float};

/// Inputs one lane multiplies and adds by itself.
const CHUNK: usize = 16;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// The most lane sums one launch may write.
const ROOM: usize = 1 << 26;

/// How a weight element is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    F32,
}

impl Form {
    /// The buffer's element type, and how element `at + i` loads.
    fn load(self) -> (&'static str, &'static str) {
        match self {
            Form::F32 => ("float", "w[at + i]"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Form::F32 => "f32",
        }
    }
}

/// Each lane's sum over its share of `k`.
#[derive(Clone, Copy, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lanes: usize,
    pub form: Form,
}

/// Each output's lane sums, added in lane order.
#[derive(Clone, Copy, Debug)]
pub struct Sums {
    pub count: usize,
    pub lanes: usize,
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
    out[gid] = acc;
";

impl Kernel for Lanes {
    fn name(&self) -> &'static str {
        "lanes"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// One row and several rows may round differently.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn source(&self) -> Source {
        let (ty, load) = self.form.load();
        let w = ("w".to_string(), ty, false);
        let (rows, body) = match self.m {
            1 => ("row", one(load)),
            _ => ("rows", several(load)),
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
}

impl Kernel for Sums {
    fn name(&self) -> &'static str {
        "sums"
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.count, self.lanes]
    }

    fn source(&self) -> Source {
        Source::Made(Text {
            entry: "linear_sums".into(),
            buffers: vec![float("parts", false), float("out", true)],
            scalars: vec![("count", "u32"), ("lanes", "u32")],
            body: SUMS.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.count, self.lanes].map(Arg::count);
        let buffers = [step.read(0), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.count),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.count * self.lanes) as f64
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Matmul { x, w, m, n, k } = *site else {
        return None;
    };
    let (lanes, form) = (lanes(k), Form::F32);
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
    Some(vec![
        Draft::new(parts, [x, w], count * lanes),
        Draft::new(Sums { count, lanes }, [Read::Draft(0)], count),
    ])
}

lower::rule!("lanes", propose);
