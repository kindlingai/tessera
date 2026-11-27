//! Generated elementwise kernels, one per chain of stages.

use crate::Kernel;
use crate::abi::{Arg, Launch, Shape, Source};
use crate::form::{F32_FORM, Load as Stored, plain, writable};
use crate::plan::Step;
use crate::rule::Draft;
use crate::sites;
use crate::sites::After;
use crate::tree::{T, Tree};
use std::cmp::Ordering;
use std::hash::{DefaultHasher, Hash, Hasher};
use tessera_codegen::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt};
use tessera_codegen::prelude::*;
use tessera_codegen::stage::Row;
use tessera_codegen::{Dialect, hash};
use tessera_graph::Act;
use tessera_work::Load;

/// Adds `count` f32 operands `a0..` to `body`, at `at`.
pub fn add_operands(body: &mut Body, at: usize, count: usize) {
    let forms = vec![F32_FORM; count];
    add_operand_forms(body, at, &forms);
}

/// Adds operands `a0..` with each form to `body`, at `at`.
pub fn add_operand_forms(body: &mut Body, at: usize, forms: &[&'static dyn Stored]) {
    let operand = |(k, form): (usize, &&'static dyn Stored)| Param::Buffer {
        name: format!("a{k}"),
        form: *form,
        writes: false,
    };
    body.params
        .splice(at..at, forms.iter().enumerate().map(operand));
}

/// One thread per element. The stages follow, then the store.
#[kernel]
fn map(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(out_form)] out: BufMut<Dyn>,
    len: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    let mut v = x[gid() as u64];
}

/// One elementwise step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapStage {
    /// Adds the next operand.
    Add,
    /// Multiplies by the next operand.
    Mul,
    Scale(f32),
    Shift(f32),
    Min(f32),
    Max(f32),
    Act(Act),
}

/// `out = stages(x, operands..)`. Empty `stages` makes a copy.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "map",
    work = len * (1 + self.0.len()),
    accepts = map_accepts,
    emits = map_emits,
    load = map_load,
    load_key = map_load_key,
    source = map_source,
    launches = map_launches,
)]
pub struct Map(
    pub Vec<MapStage>,
    pub Vec<Option<usize>>,
    pub Vec<&'static dyn Stored>,
    pub &'static dyn Stored,
);

impl MapStage {
    /// The host row call for this stage, and its scale.
    pub fn row(&self) -> (Row, f32) {
        match self {
            MapStage::Add => (Row::Add, 0.0),
            MapStage::Mul => (Row::Mul, 0.0),
            MapStage::Scale(by) => (Row::Scale, *by),
            MapStage::Shift(by) => (Row::Shift, *by),
            MapStage::Min(hi) => (Row::Min, *hi),
            MapStage::Max(lo) => (Row::Max, *lo),
            MapStage::Act(Act::GeluTanh) => (Row::GeluTanh, 0.0),
            MapStage::Act(Act::Silu) => (Row::Silu, 0.0),
            MapStage::Act(Act::Sigmoid) => (Row::Sigmoid, 0.0),
            MapStage::Act(Act::Relu) => (Row::Relu, 0.0),
            MapStage::Act(Act::Softplus) => (Row::Softplus, 0.0),
            MapStage::Act(Act::Exp) => (Row::Exp, 0.0),
            MapStage::Act(Act::Tanh) => (Row::Tanh, 0.0),
            MapStage::Act(Act::SignedSqrt) => (Row::SignedSqrt, 0.0),
        }
    }
}

impl Map {
    /// Stages whose operands are f32 and match input length.
    pub fn plain(stages: Vec<MapStage>) -> Self {
        let count = operands(&stages);
        Self(
            stages,
            vec![None; count],
            vec![F32_FORM; 1 + count],
            F32_FORM,
        )
    }

    /// A copy from `from` into `to`.
    pub fn copy(from: &'static dyn Stored, to: &'static dyn Stored) -> Self {
        Self(Vec::new(), Vec::new(), vec![plain(from)], writable(to))
    }

    /// Scales a `from` input by `by` into f32.
    pub fn scaled(by: f32, from: &'static dyn Stored) -> Self {
        Self(
            vec![MapStage::Scale(by)],
            Vec::new(),
            vec![plain(from)],
            F32_FORM,
        )
    }
}

/// `v` through an activation, as a tree.
fn act_expr(a: Act) -> Expr {
    let (v, lit) = (|| Expr::var("v"), Expr::Float);
    let call = |i, args| Expr::Call(i, args);
    let neg = |e| Expr::Neg(Box::new(e));
    match a {
        Act::Silu => {
            let den = Expr::bin(BinOp::Add, lit(1.0), call(Intrinsic::Exp, vec![neg(v())]));
            Expr::bin(BinOp::Div, v(), den)
        }
        Act::Sigmoid => {
            let den = Expr::bin(BinOp::Add, lit(1.0), call(Intrinsic::Exp, vec![neg(v())]));
            Expr::bin(BinOp::Div, lit(1.0), den)
        }
        Act::Relu => call(Intrinsic::Max, vec![v(), lit(0.0)]),
        Act::Exp => call(Intrinsic::Exp, vec![v()]),
        Act::Tanh => {
            let clamped = call(Intrinsic::Clamp, vec![v(), lit(-8.0), lit(8.0)]);
            call(Intrinsic::Tanh, vec![clamped])
        }
        Act::Softplus => {
            let most = call(Intrinsic::Max, vec![v(), neg(v())]);
            let tail = Expr::bin(BinOp::Add, lit(1.0), call(Intrinsic::Exp, vec![neg(most)]));
            let floor = call(Intrinsic::Max, vec![v(), lit(0.0)]);
            Expr::bin(BinOp::Add, floor, call(Intrinsic::Log, vec![tail]))
        }
        Act::GeluTanh => {
            let cube = Expr::bin(BinOp::Mul, Expr::bin(BinOp::Mul, v(), v()), v());
            let inner = Expr::bin(BinOp::Add, v(), Expr::bin(BinOp::Mul, lit(0.044715), cube));
            let scaled = Expr::bin(BinOp::Mul, lit(0.7978846), inner);
            let clamped = call(Intrinsic::Clamp, vec![scaled, lit(-8.0), lit(8.0)]);
            let t = Expr::bin(BinOp::Add, lit(1.0), call(Intrinsic::Tanh, vec![clamped]));
            Expr::bin(BinOp::Mul, Expr::bin(BinOp::Mul, lit(0.5), v()), t)
        }
        Act::SignedSqrt => {
            let mag = call(Intrinsic::Max, vec![v(), neg(v())]);
            let clamped = call(Intrinsic::Max, vec![mag, lit(1e-6)]);
            let root = Expr::bin(
                BinOp::Mul,
                clamped.clone(),
                call(Intrinsic::Rsqrt, vec![clamped]),
            );
            let neg_sign = Expr::select(Expr::bin(BinOp::Lt, v(), lit(0.0)), lit(-1.0), lit(0.0));
            let sign = Expr::select(Expr::bin(BinOp::Gt, v(), lit(0.0)), lit(1.0), neg_sign);
            Expr::bin(BinOp::Mul, sign, root)
        }
    }
}

/// Statements running `stages` on `v`. `operand` gives each read.
fn stage_stmts(stages: &[MapStage], operand: &dyn Fn(usize) -> Expr) -> Vec<Stmt> {
    let mut taken = 0;
    let assign = |value| Stmt::Assign {
        target: Expr::var("v"),
        op: None,
        value,
    };
    stages
        .iter()
        .map(|s| match s {
            MapStage::Add | MapStage::Mul => {
                let op = if *s == MapStage::Add {
                    BinOp::Add
                } else {
                    BinOp::Mul
                };
                taken += 1;
                assign(Expr::bin(op, Expr::var("v"), operand(taken - 1)))
            }
            MapStage::Scale(by) => assign(Expr::bin(
                BinOp::Mul,
                Expr::var("v"),
                Expr::Float(f64::from(*by)),
            )),
            MapStage::Shift(by) => assign(Expr::bin(
                BinOp::Add,
                Expr::var("v"),
                Expr::Float(f64::from(*by)),
            )),
            MapStage::Min(hi) => {
                let neg_v = Expr::Neg(Box::new(Expr::var("v")));
                let m = Expr::Call(Intrinsic::Max, vec![neg_v, Expr::Float(f64::from(-*hi))]);
                assign(Expr::Neg(Box::new(m)))
            }
            MapStage::Max(lo) => assign(Expr::Call(
                Intrinsic::Max,
                vec![Expr::var("v"), Expr::Float(f64::from(*lo))],
            )),
            MapStage::Act(a) => assign(act_expr(*a)),
        })
        .collect()
}

/// A sum through `after`'s stages, into `out[at]`.
pub fn finish(after: &After, sum: Expr, at: Expr) -> Vec<Stmt> {
    // Operands `a0..` sit at `at`. The sum takes its place.
    let read = |j: usize| match j.cmp(&after.sum) {
        Ordering::Equal => Expr::var("sum"),
        Ordering::Less => Expr::index(Expr::var(&format!("a{j}")), at.clone()),
        Ordering::Greater => Expr::index(Expr::var(&format!("a{}", j - 1)), at.clone()),
    };
    let mut stmts = vec![
        Stmt::Let {
            name: "sum".into(),
            init: sum,
        },
        Stmt::Let {
            name: "v".into(),
            init: read(0),
        },
    ];
    stmts.extend(stage_stmts(&after.stages, &|j| read(j + 1)));
    stmts.push(Stmt::Assign {
        target: Expr::index(Expr::var("out"), at),
        op: None,
        value: Expr::var("v"),
    });
    stmts
}

/// How many operands `stages` read, besides their input.
pub fn operands(stages: &[MapStage]) -> usize {
    let reads = |s: &&MapStage| matches!(s, MapStage::Add | MapStage::Mul);
    stages.iter().filter(reads).count()
}

fn map_accepts(k: &Map, input: usize, form: &dyn Stored) -> bool {
    k.2.get(input).is_some_and(|f| f.name() == form.name())
}

fn map_emits(k: &Map, form: &dyn Stored) -> bool {
    k.3.name() == form.name()
}

/// Hashes elementwise stages into `h` for unmeasured cost caching.
pub fn hash_stages(stages: &[MapStage], h: &mut impl Hasher) {
    stages.len().hash(h);
    for stage in stages {
        match *stage {
            MapStage::Add => 0u8.hash(h),
            MapStage::Mul => 1u8.hash(h),
            MapStage::Scale(by) => {
                2u8.hash(h);
                by.to_bits().hash(h);
            }
            MapStage::Shift(by) => {
                3u8.hash(h);
                by.to_bits().hash(h);
            }
            MapStage::Min(hi) => {
                4u8.hash(h);
                hi.to_bits().hash(h);
            }
            MapStage::Max(lo) => {
                5u8.hash(h);
                lo.to_bits().hash(h);
            }
            MapStage::Act(act) => {
                6u8.hash(h);
                let tag: u8 = match act {
                    Act::GeluTanh => 0,
                    Act::Silu => 1,
                    Act::Sigmoid => 2,
                    Act::Relu => 3,
                    Act::Softplus => 4,
                    Act::Exp => 5,
                    Act::Tanh => 6,
                    Act::SignedSqrt => 7,
                };
                tag.hash(h);
            }
        }
    }
}

fn map_load_key(k: &Map) -> u64 {
    let mut h = DefaultHasher::new();
    hash_stages(&k.0, &mut h);
    k.1.hash(&mut h);
    k.2.len().hash(&mut h);
    for form in &k.2 {
        form.name().hash(&mut h);
    }
    k.3.name().hash(&mut h);
    h.finish()
}

/// One thread per element, each on its own lines.
fn map_load(k: &Map, len: usize) -> Option<Load> {
    let x_bytes = k.2[0].bytes(len) as f64;
    let op_bytes =
        |(s, f): (&Option<usize>, &&'static dyn Stored)| f.bytes(s.unwrap_or(len)) as f64;
    let reads = [x_bytes]
        .into_iter()
        .chain(k.1.iter().zip(&k.2[1..]).map(op_bytes));
    let flight: f64 = k.2.iter().map(|f| f.element_bytes()).sum();
    Some(Load {
        threads: len,
        group: 256,
        trips: 1.0,
        trip: k.tree().trip(),
        reads: reads.collect(),
        written: k.3.bytes(len) as f64,
        flight,
        ..Load::default()
    })
}

fn map_source(k: &Map) -> Source {
    let mut body = k.tree();
    body.entry = format!("map_{:x}", hash(&body.text().render(Dialect::Cuda)));
    Source::Made(body)
}

fn map_launches(_: &Map, step: &Step) -> Vec<Launch> {
    let len = step.write.len;
    let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
    args.extend([step.out(), Arg::count(len)]);
    vec![Launch {
        args,
        shape: Shape::Line(len),
    }]
}

impl Map {
    /// The skeleton with the stages and the store appended.
    fn tree(&self) -> tessera_codegen::ir::Body {
        let mut body = map_ir(self.2[0], self.3);
        // The operands sit between the input and the output.
        add_operand_forms(&mut body, 1, &self.2[1..]);
        let gid = || Expr::Call(Intrinsic::Gid, Vec::new());
        let at = || Expr::cast(tessera_codegen::ir::Ty::U64, gid());
        // A repeating operand is read within its span.
        let read = |k: usize| {
            let at = match self.1[k] {
                Some(1) => Expr::cast(tessera_codegen::ir::Ty::U64, Expr::Int(0)),
                Some(span) => Expr::cast(
                    tessera_codegen::ir::Ty::U64,
                    Expr::bin(BinOp::Rem, gid(), Expr::Int(span as i64)),
                ),
                None => at(),
            };
            Expr::index(Expr::var(&format!("a{k}")), at)
        };
        body.stmts.extend(stage_stmts(&self.0, &read));
        body.stmts.push(Stmt::Assign {
            target: Expr::index(Expr::var("out"), at()),
            op: None,
            value: Expr::var("v"),
        });
        body
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Chain {
        stages,
        reads,
        len,
        spans,
    } = tree.site(at)?;
    let forms: Vec<&'static dyn Stored> = reads.iter().map(|r| plain(tree.node(*r).form)).collect();
    let out = writable(tree.node(at).form);
    Some(vec![Draft::new(Map(stages, spans, forms, out), reads, len)])
}

crate::rule!("map", propose);
