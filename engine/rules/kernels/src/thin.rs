//! A matmul with one thread per output column.

use codegen::ir::Body;
use codegen::prelude::*;
use graph::Order;
use lower::sites;
use lower::{Arg, Draft, Form as Stored, Kernel, Launch, Shape, Source, Step, T, Tree};
use sim::Load;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// The kernel per row count and form, built once.
    static TREES: RefCell<HashMap<(usize, &'static str), Body>> = RefCell::new(HashMap::new());
}

/// Its row limit: each thread keeps one sum per row.
pub const THIN_ROWS: usize = 32;

/// The same product as `Matmul`, for few rows.
#[derive(Clone, Copy, Debug)]
pub struct Thin {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
}

/// One thread per column with one sum per row.
#[kernel]
fn thin(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= n {
        return;
    }
    let mut sums = [0.0f32; THIN_ROWS];
    for row in 0..m as u32 {
        sums[row as usize] = 0.0;
    }
    for i in 0..k {
        let c = w[gid() as u64 * k as u64 + i as u64];
        for row in 0..m as u32 {
            sums[row as usize] += x[(row * k + i) as u64] * c;
        }
    }
    for row in 0..m as u32 {
        out[row as u64 * n as u64 + gid() as u64] = sums[row as usize];
    }
}

impl Thin {
    fn tree(&self) -> Body {
        let build = || thin_ir(self.m, self.form);
        let key = (self.m, self.form.name());
        TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone())
    }
}

impl Kernel for Thin {
    fn name(&self) -> &'static str {
        self.form.suffixed("thin")
    }

    /// One thread per column loops over all of `k`.
    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m as f64, self.n as f64, self.k as f64);
        let trip = self.tree().trip();
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
        let mut body = self.tree();
        body.entry = format!("plan_thin_{}_{}", self.form.name(), self.m);
        Source::Made(body)
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let sizes = [self.n, self.k].map(Arg::count);
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
    let sites::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = tree.site(at)?;
    let thin = Thin { m, n, k, form };
    (m <= THIN_ROWS).then(|| vec![Draft::new(thin, [x, w], m * n)])
}

lower::rule!("thin", propose);
