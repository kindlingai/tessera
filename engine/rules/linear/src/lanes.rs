//! A matmul for few rows, a SIMD group per column.

use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Stmt, Ty};
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{add_operands, finish};
use lower::sites;
use lower::sites::After;
use lower::{
    Arg, Draft, Form as Stored, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree,
};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// The kernel per row count and form, built once.
    static TREES: RefCell<HashMap<(usize, &'static str), Body>> = RefCell::new(HashMap::new());
}

/// Elements one wide load brings a lane.
const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: u32 = 4;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// Lanes sum parts of every row, then the group reduces.
#[kernel]
fn linear<const WIDE: u32, const AHEAD: u32>(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
    #[build] form: &'static dyn Stored,
) {
    let col = gid() / LANES as u32;
    if col >= n {
        return;
    }
    let at = col as u64 * k as u64;
    let mut s = [0.0f32; ROWS];
    for r in 0..m as u32 {
        s[r as usize] = 0.0;
    }
    for i in (lane() * WIDE..k).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let c = w.load4(at + i as u64);
        for r in 0..m as u32 {
            let v = x.load4(r as u64 * k as u64 + i as u64);
            s[r as usize] += v.x * c.x;
            s[r as usize] += v.y * c.y;
            s[r as usize] += v.z * c.z;
            s[r as usize] += v.w * c.w;
        }
    }
    for r in 0..m as u32 {
        s[r as usize] = sum32(s[r as usize]);
    }
    if lane() != 0 {
        return;
    }
}

/// `x[m, k]` times `w[n, k]` transposed, per SIMD group.
#[derive(Clone, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
}

impl Lanes {
    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let build = || linear_ir::<WIDE, AHEAD>(self.m, self.form);
        let key = (self.m, self.form.name());
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        // The operands sit between the weights and the output.
        add_operands(&mut body, 2, self.after.others());
        body.stmts.push(self.stores());
        body
    }

    /// Each row's sum through the stages, into its place.
    fn stores(&self) -> Stmt {
        let u64 = |e| Expr::cast(Ty::U64, e);
        let o = Expr::bin(BinOp::Mul, u64(Expr::var("r")), u64(Expr::var("n")));
        let o = Expr::bin(BinOp::Add, o, u64(Expr::var("col")));
        let sum = Expr::index(Expr::var("s"), Expr::var("r"));
        Stmt::For {
            var: "r".into(),
            from: Expr::Int(0),
            to: Expr::Int(self.m as i64),
            step: Expr::Int(1),
            unroll: 0,
            body: finish(&self.after, sum, o),
        }
    }
}

impl Kernel for Lanes {
    fn name(&self) -> &'static str {
        self.form.suffixed("lanes")
    }

    fn sizes(&self) -> Vec<usize> {
        vec![self.m, self.n, self.k]
    }

    /// The lanes sum in one fixed tree on every device.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m, self.n, self.k);
        let bytes = self.form.bytes(WIDE as usize);
        Some(Load {
            threads: n * LANES,
            group: 256,
            trips: (k / (LANES * WIDE as usize)) as f64,
            trip: self.tree().trip(),
            reads: self.form.matmul_bytes(m, n, k),
            written: (4 * m * n) as f64,
            flight: (bytes * AHEAD as usize) as f64,
        })
    }

    fn source(&self) -> Source {
        let mut body = self.tree();
        body.entry = format!("linear_{}_{:x}", self.form.name(), hash(&body.text().body));
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend([self.n, self.k].map(Arg::count));
        vec![Launch {
            args,
            shape: Shape::Line(self.n * LANES),
        }]
    }

    fn work(&self, _: usize) -> f64 {
        (self.m * self.n * self.k) as f64
    }

    /// Unproven, so it must earn its place by measurement.
    fn guess(&self) -> f64 {
        1e-9
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        self.form.taken(input, 1, form)
    }

    fn needs(&self) -> Needs {
        Needs::lanes()
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul { m, n, k, form, .. } = site.product;
    // Wide loads need whole groups of four along `k`.
    if m > ROWS || !k.is_multiple_of(WIDE as usize) {
        return None;
    }
    let lanes = Lanes {
        m,
        n,
        k,
        form,
        after: site.after.clone(),
    };
    Some(vec![Draft::new(lanes, site.reads(), m * n)])
}

lower::rule!("lanes", propose);
