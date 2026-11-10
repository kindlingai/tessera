//! A matmul with one thread per output column.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use graph::Order;

/// Its row limit: each thread keeps one sum per row.
pub const THIN_ROWS: usize = 32;

/// The same product as `Matmul`, for few rows.
#[derive(Clone, Copy, Debug)]
pub struct Thin {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

const BODY: &str = "    if (gid >= n) {
        return;
    }
    float sums[32];
    for (u32 row = 0; row < m; row++) {
        sums[row] = 0.0f;
    }
    for (u32 i = 0; i < k; i++) {
        float c = w[(u64)gid * k + i];
        for (u32 row = 0; row < m; row++) {
            sums[row] += x[row * k + i] * c;
        }
    }
    for (u32 row = 0; row < m; row++) {
        out[(u64)row * n + gid] = sums[row];
    }
";

impl Kernel for Thin {
    fn name(&self) -> &'static str {
        "thin"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn source(&self) -> Source {
        Source::Made(Text {
            entry: "plan_thin".into(),
            buffers: vec![float("x", false), float("w", false), float("out", true)],
            scalars: vec![("m", "u32"), ("n", "u32"), ("k", "u32")],
            body: BODY.into(),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.m, self.n, self.k].map(Arg::count);
        let buffers = [step.read(0), step.read(1), step.out()];
        vec![Launch {
            args: buffers.into_iter().chain(sizes).collect(),
            shape: Shape::Line(self.n),
        }]
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    fn work(&self, _: usize) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Matmul { x, w, m, n, k } = *site else {
        return None;
    };
    (m <= THIN_ROWS).then(|| vec![Draft::new(Thin { m, n, k }, [x, w], m * n)])
}

crate::rule!("thin", propose);
