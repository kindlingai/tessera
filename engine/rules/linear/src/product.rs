//! Parts every matmul kernel shares.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use tessera_codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use tessera_codegen::{Dialect, hash};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::kernels::{add_operand_forms, finish, hash_stages};
use tessera_lower::sites;
use tessera_lower::sites::{After, Near, Picked};
use tessera_lower::{Launch, Read, Shape, Source, Step, T, Tree};

/// Kernel, schedule, operands, flags, and extra tag.
pub type Variant = (
    &'static str,
    usize,
    &'static str,
    &'static str,
    &'static str,
    u8,
    usize,
);

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
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub other_forms: Vec<&'static dyn Stored>,
    pub after: After,
    /// Set when each row reads its own matrix.
    pub picked: Option<Picked>,
    /// Set when `x` is each pixel's neighbours, read by id.
    pub near: Option<Near>,
}

/// The index of the output at `row` and `col`.
pub fn place(row: Expr, col: Expr) -> Expr {
    let u64 = |e| Expr::cast(Ty::U64, e);
    let o = Expr::bin(BinOp::Mul, u64(row), u64(Expr::var("n")));
    Expr::bin(BinOp::Add, o, u64(col))
}

impl Product {
    /// The finished matmul at `at`, with its reads.
    pub fn find(tree: &Tree, at: T) -> Option<(Self, Vec<Read>)> {
        let site: sites::Finished = tree.site(at)?;
        let sites::Matmul {
            x,
            m,
            n,
            k,
            form,
            picked,
            ..
        } = site.product;
        let x_form = plain(tree.node(x).form);
        let out_form = writable(tree.node(at).form);
        let mut other_forms = Vec::with_capacity(site.others.len());
        for read in &site.others {
            let form = match *read {
                Read::Node(t) => plain(tree.node(t).form),
                Read::Kept(_) | Read::Draft(_) => F32_FORM,
            };
            other_forms.push(form);
        }
        let after = site.after.clone();
        Some((
            Self {
                m,
                n,
                k,
                form,
                x_form,
                out_form,
                other_forms,
                after,
                picked,
                near: None,
            },
            site.reads(),
        ))
    }

    /// The kernel for `key`, with the chain's operands added.
    pub fn tree(&self, key: Variant, build: impl FnOnce() -> Body) -> Body {
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        let at = 2 + usize::from(self.picked.is_some()) + usize::from(self.near.is_some());
        add_operand_forms(&mut body, at, &self.other_forms);
        body
    }

    /// Each row's sum through the stages, into its place.
    pub fn rows(&self, row: &str, sums: &str, col: Expr) -> Stmt {
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
    pub fn source(&self, mut body: Body, kind: &str) -> Source {
        let hashed = hash(&body.text().render(Dialect::Cuda));
        body.entry = format!("{kind}_{}_{hashed:x}", self.form.name());
        Source::Made(body)
    }

    /// One launch: the reads, the output, then `counts`.
    pub fn launch(&self, step: &Step, counts: &[usize], shape: Shape) -> Vec<Launch> {
        vec![Launch::plain(step, counts, shape)]
    }

    pub fn outputs(&self) -> usize {
        self.m * self.n
    }

    pub fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    pub fn work(&self) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Hash of every product field affecting `load()`.
    pub fn load_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        (self.m, self.n, self.k).hash(&mut h);
        self.form.name().hash(&mut h);
        self.x_form.name().hash(&mut h);
        self.out_form.name().hash(&mut h);
        self.other_forms.len().hash(&mut h);
        for form in &self.other_forms {
            form.name().hash(&mut h);
        }
        self.after.sum.hash(&mut h);
        hash_stages(&self.after.stages, &mut h);
        self.picked
            .map(|p| (p.count, p.stride, p.offset, p.repeat, p.first))
            .hash(&mut h);
        self.near.map(|n| (n.pixels, n.cin, n.taps)).hash(&mut h);
        h.finish()
    }

    /// Combines the product load key with kernel-specific fields.
    pub fn key_with(&self, extra: impl Hash) -> u64 {
        let mut h = DefaultHasher::new();
        self.load_key().hash(&mut h);
        extra.hash(&mut h);
        h.finish()
    }

    pub fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        let extra = usize::from(self.picked.is_some()) + usize::from(self.near.is_some());
        match input {
            0 => form.name() == self.x_form.name(),
            1 => form.name() == self.form.name(),
            i if i < 2 + extra => !form.narrow(),
            i => self
                .other_forms
                .get(i - 2 - extra)
                .is_some_and(|f| form.name() == f.name()),
        }
    }

    pub fn emits(&self, form: &dyn Stored) -> bool {
        form.name() == self.out_form.name()
    }

    pub fn read_bytes(&self) -> Vec<f64> {
        let mut reads = vec![
            self.x_form.bytes(self.m * self.k) as f64,
            self.form.bytes(self.n * self.k) as f64,
        ];
        for other in &self.other_forms {
            reads.push(other.bytes(self.m * self.n) as f64);
        }
        reads
    }
}
