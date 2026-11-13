//! A matmul with one thread per output column.

use codegen::prelude::*;
use graph::Order;
use lower::{Arg, DType, Draft, Kernel, Launch, Shape, Site, Source, Step, Stored, T, Tree};
use sim::{Load, Ops};

/// Its row limit: each thread keeps one sum per row.
pub const THIN_ROWS: usize = 32;

/// The same product as `Matmul`, for few rows.
#[derive(Clone, Copy, Debug)]
pub struct Thin {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: DType,
}

/// One thread per column with one sum per row.
#[kernel]
fn thin<F: Form<Elem = f32>>(x: Buf<F32>, w: Buf<F>, mut out: BufMut<F32>, m: u32, n: u32, k: u32) {
    if gid() >= n {
        return;
    }
    let mut sums = [0.0f32; THIN_ROWS];
    for row in 0..m {
        sums[row as usize] = 0.0;
    }
    for i in 0..k {
        let c = w[gid() as u64 * k as u64 + i as u64];
        for row in 0..m {
            sums[row as usize] += x[(row * k + i) as u64] * c;
        }
    }
    for row in 0..m {
        out[row as u64 * n as u64 + gid() as u64] = sums[row as usize];
    }
}

impl Kernel for Thin {
    fn name(&self) -> &'static str {
        match self.form {
            DType::Bf16 => "thin_bf16",
            _ => "thin",
        }
    }

    /// One thread per column loops over all of `k`.
    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m as f64, self.n as f64, self.k as f64);
        let trip = Ops {
            float: m,
            int: 2.0 + m,
            loads: 1.0 + m,
            fresh: 1.0 / 16.0,
            ..Ops::default()
        };
        Some(Load {
            threads: self.n,
            group: 256,
            trips: k,
            trip,
            reads: self.form.matmul_bytes(self.m, self.n, self.k),
            written: 4.0 * m * n,
            flight: 64.0,
        })
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn source(&self) -> Source {
        let body = match self.form {
            DType::Bf16 => thin_ir::<Bf16>(),
            _ => thin_ir::<F32>(),
        };
        let mut text = body.text();
        text.entry = format!("plan_thin_{}", self.form.name());
        Source::Made(text)
    }

    fn accepts(&self, input: usize, form: DType) -> bool {
        self.form.taken(input, 1, form)
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
    let thin = Thin { m, n, k, form };
    (m <= THIN_ROWS).then(|| vec![Draft::new(thin, [x, w], m * n)])
}

lower::rule!("thin", propose);
