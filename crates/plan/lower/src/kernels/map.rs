//! Generated elementwise kernels, one per chain of stages.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use crate::tree::{T, Tree};
use codegen::hash;
use codegen::ir::{BinOp, Expr, Intrinsic, Param, Stmt};
use codegen::prelude::*;
use graph::{Act, DType};
use sim::Load;

/// One thread per element. The stages follow, then the store.
#[kernel]
fn map(x: Buf<F32>, out: BufMut<F32>, len: u32) {
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
    Act(Act),
}

/// `out = stages(x, operands..)`. No stages makes a copy.
#[derive(Clone, Debug)]
pub struct Map(pub Vec<MapStage>);

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
        Act::GeluTanh => {
            let cube = Expr::bin(BinOp::Mul, Expr::bin(BinOp::Mul, v(), v()), v());
            let inner = Expr::bin(BinOp::Add, v(), Expr::bin(BinOp::Mul, lit(0.044715), cube));
            let scaled = Expr::bin(BinOp::Mul, lit(0.7978846), inner);
            let clamped = call(Intrinsic::Clamp, vec![scaled, lit(-8.0), lit(8.0)]);
            let t = Expr::bin(BinOp::Add, lit(1.0), call(Intrinsic::Tanh, vec![clamped]));
            Expr::bin(BinOp::Mul, Expr::bin(BinOp::Mul, lit(0.5), v()), t)
        }
    }
}

/// Statements running `stages` on `v`. `operand` gives each read.
pub fn stage_stmts(stages: &[MapStage], operand: &dyn Fn(usize) -> Expr) -> Vec<Stmt> {
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
            MapStage::Act(a) => assign(act_expr(*a)),
        })
        .collect()
}

/// How many operands `stages` read, besides their input.
pub fn operands(stages: &[MapStage]) -> usize {
    let reads = |s: &&MapStage| matches!(s, MapStage::Add | MapStage::Mul);
    stages.iter().filter(reads).count()
}

impl Kernel for Map {
    fn name(&self) -> &'static str {
        "map"
    }

    /// One thread per element, each on its own lines.
    fn load(&self, len: usize) -> Option<Load> {
        let inputs = 1 + operands(&self.0);
        Some(Load {
            threads: len,
            group: 256,
            trips: 1.0,
            trip: self.tree().trip(),
            reads: vec![4.0 * len as f64; inputs],
            written: 4.0 * len as f64,
            flight: 4.0 * inputs as f64,
        })
    }

    fn source(&self) -> Source {
        let mut body = self.tree();
        body.entry = format!("map_{:x}", hash(&body.text().body));
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let len = step.write.len;
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.extend([step.out(), Arg::count(len)]);
        vec![Launch {
            args,
            shape: Shape::Line(len),
        }]
    }

    fn work(&self, len: usize) -> f64 {
        (len * (1 + self.0.len())) as f64
    }
}

impl Map {
    /// The skeleton with the stages and the store appended.
    fn tree(&self) -> codegen::ir::Body {
        let mut body = map_ir();
        let operand = |k: usize| Param::Buffer {
            name: format!("a{k}"),
            dtype: DType::F32,
            writes: false,
        };
        // The operands sit between the input and the output.
        body.params
            .splice(1..1, (0..operands(&self.0)).map(operand));
        let at = || Expr::cast(codegen::ir::Ty::U64, Expr::Call(Intrinsic::Gid, Vec::new()));
        let read = |k: usize| Expr::index(Expr::var(&format!("a{k}")), at());
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
    let Site::Map { stages, reads, len } = tree.site(at)? else {
        return None;
    };
    Some(vec![Draft::new(Map(stages), reads, len)])
}

crate::rule!("map", propose);
