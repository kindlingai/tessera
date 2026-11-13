//! Generated elementwise kernels, one per chain of stages.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use crate::tree::{T, Tree};
use codegen::ir::{BinOp, Expr, Intrinsic, Stmt};
use codegen::{Text, float, hash};
use graph::Act;
use sim::{Load, Ops};
use std::fmt::Write;

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

fn act(a: Act) -> &'static str {
    match a {
        Act::Silu => "v = v / (1.0f + f_exp(-v));",
        Act::Sigmoid => "v = 1.0f / (1.0f + f_exp(-v));",
        Act::Relu => "v = f_max(v, 0.0f);",
        Act::GeluTanh => {
            "v = 0.5f * v * (1.0f + f_tanh(f_clamp(\
             0.7978846f * (v + 0.044715f * v * v * v), -8.0f, 8.0f)));"
        }
    }
}

/// Source lines running `stages` on `v`. `operand` names each read.
pub fn stage_lines(stages: &[MapStage], operand: &dyn Fn(usize) -> String) -> String {
    let (mut lines, mut taken) = (String::new(), 0);
    for s in stages {
        let line = match s {
            MapStage::Add | MapStage::Mul => {
                let op = if *s == MapStage::Add { '+' } else { '*' };
                taken += 1;
                format!("v = v {op} {};", operand(taken - 1))
            }
            // Bits, so the constant survives the round trip.
            MapStage::Scale(by) => format!("v = v * f_bits(0x{:08x}u);", by.to_bits()),
            MapStage::Act(a) => act(*a).to_string(),
        };
        let _ = writeln!(lines, "    {line}");
    }
    lines
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
        let special = |s: &&MapStage| matches!(s, MapStage::Act(_));
        let acts = self.0.iter().filter(special).count() as f64;
        let trip = Ops {
            float: self.0.len() as f64 - acts,
            int: 1.0,
            special: acts,
            loads: inputs as f64,
            stores: 1.0,
            fresh: 1.0,
            width: 0.0,
            tile_flops: 0.0,
        };
        Some(Load {
            threads: len,
            group: 256,
            trips: 1.0,
            trip,
            reads: vec![4.0 * len as f64; inputs],
            written: 4.0 * len as f64,
            flight: 4.0 * inputs as f64,
        })
    }

    fn source(&self) -> Source {
        let mut buffers = vec![float("x", false)];
        buffers.extend((0..operands(&self.0)).map(|k| float(&format!("a{k}"), false)));
        let start = "    if (gid >= len) {\n        return;\n    }\n    float v = x[gid];\n";
        let lines = stage_lines(&self.0, &|k| format!("a{k}[gid]"));
        let body = format!("{start}{lines}    out[gid] = v;\n");
        buffers.push(float("out", true));
        Source::Made(Text {
            entry: format!("map_{:x}", hash(&body)),
            buffers,
            scalars: vec![("len", "u32")],
            body,
        })
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

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Site::Map { stages, reads, len } = tree.site(at)? else {
        return None;
    };
    Some(vec![Draft::new(Map(stages), reads, len)])
}

crate::rule!("map", propose);
