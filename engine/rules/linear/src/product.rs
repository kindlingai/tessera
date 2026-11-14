//! What every matmul kernel shares.

use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use lower::kernels::{add_operands, finish};
use lower::sites;
use lower::sites::After;
use lower::{Arg, Form as Stored, Launch, Read, Shape, Source, Step, T, Tree};
use std::cell::RefCell;
use std::collections::HashMap;

/// Kernel, rows or schedule, form, wide loads.
pub(crate) type Variant = (&'static str, usize, &'static str, bool);

thread_local! {
    /// Each kernel per variant, built once.
    static TREES: RefCell<HashMap<Variant, Body>> = RefCell::new(HashMap::new());
}

/// `x[m, k]` times `w[n, k]` transposed, then `after`.
#[derive(Clone, Debug)]
pub struct Product {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
}

/// The index of the output at `row` and `col`.
pub(crate) fn place(row: Expr, col: Expr) -> Expr {
    let u64 = |e| Expr::cast(Ty::U64, e);
    let o = Expr::bin(BinOp::Mul, u64(row), u64(Expr::var("n")));
    Expr::bin(BinOp::Add, o, u64(col))
}

impl Product {
    /// The finished matmul at `at`, with its reads.
    pub(crate) fn find(tree: &Tree, at: T) -> Option<(Self, Vec<Read>)> {
        let site: sites::Finished = tree.site(at)?;
        let sites::Matmul { m, n, k, form, .. } = site.product;
        let after = site.after.clone();
        Some((
            Self {
                m,
                n,
                k,
                form,
                after,
            },
            site.reads(),
        ))
    }

    /// The kernel for `key`, with the chain's operands added.
    pub(crate) fn tree(&self, key: Variant, build: impl FnOnce() -> Body) -> Body {
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        add_operands(&mut body, 2, self.after.others());
        body
    }

    /// Each row's sum through the stages, into its place.
    pub(crate) fn rows(&self, row: &str, sums: &str, col: Expr) -> Stmt {
        let sum = Expr::index(Expr::var(sums), Expr::var(row));
        Stmt::For {
            var: row.into(),
            from: Expr::Int(0),
            to: Expr::Int(self.m as i64),
            step: Expr::Int(1),
            unroll: 0,
            body: finish(&self.after, sum, place(Expr::var(row), col)),
        }
    }

    /// The tree under an entry name that hashes its text.
    pub(crate) fn source(&self, mut body: Body, kind: &str) -> Source {
        let hashed = hash(&body.text().body);
        body.entry = format!("{kind}_{}_{hashed:x}", self.form.name());
        Source::Made(body)
    }

    /// One launch: the reads, the output, then `counts`.
    pub(crate) fn launch(&self, step: &Step, counts: &[usize], shape: Shape) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend(counts.iter().copied().map(Arg::count));
        vec![Launch { args, shape }]
    }

    pub(crate) fn outputs(&self) -> usize {
        self.m * self.n
    }

    pub(crate) fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    pub(crate) fn work(&self) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    pub(crate) fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }
}
