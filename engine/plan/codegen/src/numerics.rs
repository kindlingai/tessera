//! Tile rounding per backend, and error bounds from the tree.

use crate::ir::{BinOp, Body, Expr, Intrinsic, Stmt};
use std::collections::HashMap;

/// How the tile keeps a float operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    F32,
    /// Two tf32 pieces: 21 bits, f32's range.
    Tf32x2,
    /// Two f16 pieces: 22 bits, overflow past 65504.
    F16x2,
}

/// What a backend's tile multiply rounds away.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Precision {
    pub operand: Operand,
    /// Products the unit sums before adding to its accumulator.
    pub depth: usize,
    /// Accumulator bits between tile sums. Assumed for tensor cores.
    pub acc_bits: u32,
    /// True where the accumulator truncates.
    pub truncates: bool,
}

impl Precision {
    /// Plain f32 arithmetic, as the host computes.
    pub const EXACT: Self = Self {
        operand: Operand::F32,
        depth: 1,
        acc_bits: 24,
        truncates: false,
    };
    pub const METAL: Self = Self {
        operand: Operand::F32,
        depth: 8,
        acc_bits: 24,
        truncates: false,
    };
    pub const CUDA: Self = Self {
        operand: Operand::Tf32x2,
        depth: 8,
        acc_bits: 22,
        truncates: true,
    };
    pub const HIP: Self = Self {
        operand: Operand::F16x2,
        depth: 16,
        acc_bits: 22,
        truncates: true,
    };

    /// Significant bits an operand keeps.
    pub fn bits(&self) -> u32 {
        match self.operand {
            Operand::F32 => 24,
            Operand::Tf32x2 => 21,
            Operand::F16x2 => 22,
        }
    }

    /// The least error an operand can carry: f16's subnormal step.
    pub fn floor(&self) -> f32 {
        match self.operand {
            Operand::F16x2 => 2f32.powi(-25),
            _ => 0.0,
        }
    }

    /// The error `x` picks up as an operand.
    pub fn operand_error(&self, x: f32) -> f32 {
        (x.abs() * 2f32.powi(-(self.bits() as i32))).max(self.floor())
    }

    /// Relative error one accumulator add may cost.
    pub fn acc_error(&self) -> f32 {
        let unit = 2f32.powi(-(self.acc_bits as i32));
        if self.truncates { 2.0 * unit } else { unit }
    }

    /// `x` as the tile holds it: `(hi, lo)` pieces.
    pub fn split(&self, x: f32) -> (f32, f32) {
        match self.operand {
            Operand::F32 => (x, 0.0),
            Operand::Tf32x2 => {
                let hi = tf32(x);
                (hi, tf32(x - hi))
            }
            Operand::F16x2 => {
                let hi = f16(x);
                (hi, f16(x - hi))
            }
        }
    }

    /// `acc + sum`, as the accumulator keeps it.
    pub fn accumulate(&self, acc: f32, sum: f32) -> f32 {
        let total = acc + sum;
        if !self.truncates || total == 0.0 || !total.is_finite() {
            return total;
        }
        let ulp = 2f32.powi(total.abs().log2().floor() as i32 - self.acc_bits as i32 + 1);
        (total / ulp).trunc() * ulp
    }
}

/// `x` rounded to `bits` significant bits, ties away from zero.
fn round_to(x: f32, bits: i32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let ulp = 2f32.powi(x.abs().log2().floor() as i32 - bits + 1);
    (x / ulp).round() * ulp
}

/// `x` as tf32: 11 significant bits, f32's exponent.
fn tf32(x: f32) -> f32 {
    round_to(x, 11)
}

/// `x` as f16, subnormals kept, overflow to infinity.
fn f16(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let exp = x.abs().log2().floor() as i32;
    // Below 2^-14 the step stays at 2^-24.
    let ulp = 2f32.powi((exp - 10).max(-24));
    let rounded = (x / ulp).round_ties_even() * ulp;
    if rounded.abs() > 65504.0 {
        return f32::INFINITY.copysign(x);
    }
    rounded
}

/// How many rounded sums reach one output of a tree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Terms {
    /// Matrix-unit products summed.
    pub tile: f64,
    /// Adds in f32 lanes.
    pub lane: f64,
}

impl Terms {
    /// The error allowed in `x` times `w` transposed, `k` wide.
    pub fn matmul(&self, p: &Precision, x: &[f32], w: &[f32], k: usize) -> f32 {
        let op = |v: &[f32]| -> Vec<f32> {
            match self.tile > 0.0 {
                true => v.iter().map(|x| p.operand_error(*x)).collect(),
                false => v.iter().map(|x| x.abs() * 2f32.powi(-24)).collect(),
            }
        };
        let (ex, ew) = (op(x), op(w));
        let adds = self.tile as f32 * p.acc_error() + self.lane as f32 * 2f32.powi(-24);
        let mut most = 0.0f32;
        for (xi, exi) in x.chunks(k).zip(ex.chunks(k)) {
            for (wj, ewj) in w.chunks(k).zip(ew.chunks(k)) {
                let mut bound = 0.0f32;
                for i in 0..k {
                    let (a, b) = (xi[i].abs(), wj[i].abs());
                    bound += a * ewj[i] + b * exi[i] + a * b * adds;
                }
                most = most.max(bound);
            }
        }
        most
    }
}

/// Sums reaching one output of `body`. `None`: an unbound loop.
pub fn terms(body: &Body, params: &HashMap<String, f64>) -> Option<Terms> {
    let mut walk = Walk {
        params,
        times: 1.0,
        terms: Terms::default(),
    };
    walk.stmts(&body.stmts)?;
    Some(walk.terms)
}

struct Walk<'a> {
    params: &'a HashMap<String, f64>,
    /// The product of enclosing loop trip counts.
    times: f64,
    terms: Terms,
}

impl Walk<'_> {
    /// `e` as a number. Lane ids read as zero.
    fn eval(&self, e: &Expr) -> Option<f64> {
        Some(match e {
            Expr::Int(v) => *v as f64,
            Expr::Float(v) => *v,
            Expr::Var(name) => *self.params.get(name)?,
            Expr::Cast(_, inner) => self.eval(inner)?,
            Expr::Call(Intrinsic::Lane | Intrinsic::Lid | Intrinsic::Gid, _) => 0.0,
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.eval(a)?, self.eval(b)?);
                match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => (a / b).trunc(),
                    BinOp::Rem => a % b,
                    BinOp::Shl => a * 2f64.powi(b as i32),
                    BinOp::Shr => (a / 2f64.powi(b as i32)).trunc(),
                    _ => return None,
                }
            }
            _ => return None,
        })
    }

    fn stmts(&mut self, stmts: &[Stmt]) -> Option<()> {
        stmts.iter().try_for_each(|s| self.stmt(s))
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            // A lane sum: a tree of five roundings.
            Expr::Call(Intrinsic::Sum32, args) => {
                self.terms.lane += 5.0 * self.times;
                args.iter().for_each(|a| self.expr(a));
            }
            Expr::Call(_, args) | Expr::Fn(_, _, args) => args.iter().for_each(|a| self.expr(a)),
            Expr::Bin(_, a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Expr::Neg(a) | Expr::Not(a) | Expr::Cast(_, a) | Expr::Field(a, _) => self.expr(a),
            Expr::Index(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Expr::Select(c, a, b) => {
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => {}
        }
    }

    fn stmt(&mut self, s: &Stmt) -> Option<()> {
        match s {
            Stmt::Let { init, .. } | Stmt::Expr(init) => self.expr(init),
            Stmt::Assign { op, value, .. } => {
                // An add into a variable is one more rounding.
                if *op == Some(BinOp::Add) {
                    self.terms.lane += self.times;
                }
                self.expr(value);
            }
            Stmt::For {
                from,
                to,
                step,
                body,
                ..
            } => {
                let (from, to, step) = (self.eval(from)?, self.eval(to)?, self.eval(step)?);
                let trips = ((to - from).max(0.0) / step).ceil();
                let was = self.times;
                self.times *= trips;
                self.stmts(body)?;
                self.times = was;
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.stmts(then)?;
                self.stmts(els)?;
            }
            // Each block's products, then its flush into f32.
            Stmt::TileBlock { depth, .. } => {
                self.terms.tile += *depth as f64 * self.times;
                self.terms.lane += self.times;
            }
            Stmt::TileEach { body, .. } => self.stmts(body)?,
            Stmt::TileScale { by, .. } => self.expr(by),
            Stmt::Array { .. }
            | Stmt::Shared { .. }
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::TileAcc { .. }
            | Stmt::TileZero { .. } => {}
        }
        Some(())
    }
}
