//! A matmul for few rows, a SIMD group per column.

use codegen::hash;
use codegen::ir::{BinOp, Body, Expr, Param, Stmt, Ty};
use codegen::prelude::*;
use graph::{DType, Order};
use lower::kernels::{MapStage, operands, stage_stmts};
use lower::sites;
use lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, Stored, T, Tree};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;

thread_local! {
    /// The kernel per row count and form, built once.
    static TREES: RefCell<HashMap<(usize, bool), Body>> = RefCell::new(HashMap::new());
}

/// Elements one wide load brings a lane.
const WIDE: u32 = 4;

/// Trips unrolled, so a lane keeps several loads in flight.
const AHEAD: u32 = 4;

/// The most rows this schedule is offered for.
const ROWS: usize = 32;

/// Lanes sum parts of every row, then the group reduces.
#[kernel]
fn linear<const WIDE: u32, const AHEAD: u32, F: Form<Elem = f32>>(
    x: Buf<F32>,
    w: Buf<F>,
    out: BufMut<F32>,
    n: u32,
    k: u32,
    #[build] m: usize,
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

/// Elementwise stages a sum runs through before it is stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct After {
    pub stages: Vec<MapStage>,
    /// Which of the chain's reads the sum stands in for.
    pub sum: usize,
}

/// `x[m, k]` times `w[n, k]` transposed, per SIMD group.
#[derive(Clone, Debug)]
pub struct Lanes {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: DType,
    pub after: After,
}

impl Lanes {
    fn others(&self) -> usize {
        match self.after.stages.is_empty() {
            true => 0,
            false => operands(&self.after.stages),
        }
    }

    /// The kernel with the chain's operands and its stores added.
    fn tree(&self) -> Body {
        let build = || match self.form {
            DType::Bf16 => linear_ir::<WIDE, AHEAD, Bf16>(self.m),
            _ => linear_ir::<WIDE, AHEAD, F32>(self.m),
        };
        let key = (self.m, self.form == DType::Bf16);
        let mut body = TREES.with(|t| t.borrow_mut().entry(key).or_insert_with(build).clone());
        let operand = |a: usize| Param::Buffer {
            name: format!("a{a}"),
            dtype: DType::F32,
            writes: false,
        };
        // The operands sit between the weights and the output.
        body.params.splice(2..2, (0..self.others()).map(operand));
        body.stmts.push(self.stores());
        body
    }

    /// Each row's sum through the stages, into its place.
    fn stores(&self) -> Stmt {
        let After { stages, sum } = &self.after;
        let u64 = |e| Expr::cast(Ty::U64, e);
        let o = Expr::bin(BinOp::Mul, u64(Expr::var("r")), u64(Expr::var("n")));
        let o = Expr::bin(BinOp::Add, o, u64(Expr::var("col")));
        let v = Expr::index(Expr::var("s"), Expr::var("r"));
        // The chain's reads in order, the sum among them.
        let read = |j: usize| match j.cmp(sum) {
            Ordering::Equal => v.clone(),
            Ordering::Less => Expr::index(Expr::var(&format!("a{j}")), o.clone()),
            Ordering::Greater => Expr::index(Expr::var(&format!("a{}", j - 1)), o.clone()),
        };
        let mut body = vec![Stmt::Let {
            name: "v".into(),
            init: read(0),
        }];
        body.extend(stage_stmts(stages, &|j| read(j + 1)));
        body.push(Stmt::Assign {
            target: Expr::index(Expr::var("out"), o),
            op: None,
            value: Expr::var("v"),
        });
        Stmt::For {
            var: "r".into(),
            from: Expr::Int(0),
            to: Expr::Int(self.m as i64),
            step: Expr::Int(1),
            unroll: 0,
            body,
        }
    }
}

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

    /// The lanes sum in one fixed tree on every device.
    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn load(&self, _: usize) -> Option<Load> {
        let (m, n, k) = (self.m, self.n, self.k);
        let bytes = self.form.bytes() * WIDE as usize;
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

    fn accepts(&self, input: usize, form: DType) -> bool {
        self.form.taken(input, 1, form)
    }

    fn needs(&self) -> Needs {
        Needs {
            lanes: LANES,
            group_threads: 256,
            ..Needs::default()
        }
    }
}

/// The matmul at `at`, or the one its stages finish.
pub(crate) fn finishing(tree: &Tree, at: T) -> Option<(sites::Matmul, After, Vec<Read>)> {
    if let Some(product) = tree.site::<sites::Matmul>(at) {
        return Some((product, After::default(), Vec::new()));
    }
    let sites::Chain { stages, reads, .. } = tree.site(at)?;
    let lone = |r: &T| tree.site::<sites::Matmul>(*r).filter(|_| !tree.root(*r));
    let sum = reads.iter().position(|r| lone(r).is_some())?;
    let others = reads.iter().filter(|r| **r != reads[sum]);
    let others = others.map(|r| Read::Node(*r)).collect();
    Some((lone(&reads[sum])?, After { stages, sum }, others))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, after, others) = finishing(tree, at)?;
    let sites::Matmul {
        x,
        w,
        m,
        n,
        k,
        form,
    } = product;
    // Wide loads need whole groups of four along `k`.
    if m > ROWS || !k.is_multiple_of(WIDE as usize) {
        return None;
    }
    let lanes = Lanes {
        m,
        n,
        k,
        form,
        after,
    };
    let reads = [Read::Node(x), Read::Node(w)].into_iter().chain(others);
    Some(vec![Draft::new(lanes, reads, m * n)])
}

lower::rule!("lanes", propose);
