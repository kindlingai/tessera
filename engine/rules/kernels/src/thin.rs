//! A matmul with one thread per output column.

use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Intrinsic, Stmt, Ty};
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{add_operands, finish};
use lower::sites;
use lower::sites::After;
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
#[derive(Clone, Debug)]
pub struct Thin {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
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
}

impl Thin {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let build = || thin_ir(self.m, self.form);
        let key = (self.m, self.form.name());
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        add_operands(&mut body, 2, self.after.others());
        let u64 = |e| Expr::cast(Ty::U64, e);
        let o = Expr::bin(BinOp::Mul, u64(Expr::var("row")), u64(Expr::var("n")));
        let o = Expr::bin(BinOp::Add, o, u64(Expr::Call(Intrinsic::Gid, Vec::new())));
        let sum = Expr::index(Expr::var("sums"), Expr::var("row"));
        body.stmts.push(Stmt::For {
            var: "row".into(),
            from: Expr::Int(0),
            to: Expr::Int(self.m as i64),
            step: Expr::Int(1),
            unroll: 0,
            body: finish(&self.after, sum, o),
        });
        body
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
        let body = self.tree();
        // The text holds the stages. The name hashes the text.
        let entry = format!("thin_{}_{:x}", self.form.name(), hash(&body.text().body));
        Source::named(body, entry)
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.n, self.k].map(Arg::count));
        vec![Launch {
            args,
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
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul { m, n, k, form, .. } = site.product;
    if m > THIN_ROWS {
        return None;
    }
    let thin = Thin {
        m,
        n,
        k,
        form,
        after: site.after.clone(),
    };
    Some(vec![Draft::new(thin, site.reads(), m * n)])
}

lower::rule!("thin", propose);
