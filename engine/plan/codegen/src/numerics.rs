//! Tile rounding per backend, and error bounds from the tree.

use crate::ir::{BinOp, Body, Expr, Intrinsic, Stmt};
use std::collections::HashMap;
use tessera_engine_settings::Backend;

/// How the tile keeps a float operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    F32,
    /// Two tf32 pieces: 21 bits, f32's range.
    Tf32x2,
    /// Two f16 pieces: 22 bits, overflow past 65504.
    F16x2,
}

/// The error a backend's tile multiply rounds away.
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

    /// How `backend`'s tiles round.
    pub fn of(backend: Backend) -> Self {
        match backend {
            Backend::Metal => Self::METAL,
            Backend::Cuda => Self::CUDA,
            Backend::Rocm => Self::HIP,
        }
    }

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

    /// The unit on bf16 parts: exact products, 16 a sum.
    pub fn parts(self) -> Self {
        Self {
            operand: Operand::F32,
            depth: 16,
            ..self
        }
    }

    /// `x` in the tile's form: `(hi, lo)` pieces.
    #[inline]
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

    /// `acc + sum`, in the accumulator's precision.
    #[inline]
    pub fn accumulate(&self, acc: f32, sum: f32) -> f32 {
        let total = acc + sum;
        if !self.truncates || self.acc_bits >= 24 {
            return total;
        }
        let bits = total.to_bits();
        let exp = (bits >> 23) & 0xff;
        if exp == 0 || exp == 0xff {
            return if total == 0.0 || exp == 0xff {
                total
            } else {
                truncate_subnormal(total, self.acc_bits)
            };
        }
        let shift = 24 - self.acc_bits;
        f32::from_bits(bits & !((1u32 << shift) - 1))
    }
}

#[cold]
fn truncate_subnormal(total: f32, acc_bits: u32) -> f32 {
    let x = f64::from(total);
    let ulp = 2f64.powi(x.abs().log2().floor() as i32 - acc_bits as i32 + 1);
    ((x / ulp).trunc() * ulp) as f32
}

#[cold]
fn round_subnormal(x: f32, bits: i32) -> f32 {
    let v = f64::from(x);
    let ulp = 2f64.powi(v.abs().log2().floor() as i32 - bits + 1);
    ((v / ulp).round() * ulp) as f32
}

/// `x` as tf32: 11 significant bits, ties away from zero.
fn tf32(x: f32) -> f32 {
    let raw = x.to_bits();
    let exp = (raw >> 23) & 0xff;
    if exp == 0 || exp == 0xff {
        return if x == 0.0 || exp == 0xff {
            x
        } else {
            round_subnormal(x, 11)
        };
    }
    f32::from_bits((raw + 0x1000) & !0x1fff)
}

/// `x` as f16, subnormals kept, overflow to infinity.
fn f16(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mag = bits & 0x7fff_ffff;
    if mag >= 0x7f80_0000 || mag == 0 {
        return x;
    }
    if mag >= 0x3880_0000 {
        let half = 0x0fff + ((mag >> 13) & 1);
        let r = mag.wrapping_add(half) & !0x1fff;
        if r > 0x477f_e000 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | r);
    }
    const STEP: f32 = f32::from_bits(0x3380_0000);
    const INV_STEP: f32 = f32::from_bits(0x4b80_0000);
    (x * INV_STEP).round_ties_even() * STEP
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
        let tiled = self.tile > 0.0;
        let eps = if tiled {
            2f32.powi(-(p.bits() as i32))
        } else {
            2f32.powi(-24)
        };
        let floor = if tiled { p.floor() } else { 0.0 };
        let adds = self.tile as f32 * p.acc_error() + self.lane as f32 * 2f32.powi(-24);
        let mut most = 0.0f32;
        if floor == 0.0 {
            let scale = 2.0 * eps + adds;
            for xi in x.chunks(k) {
                for wj in w.chunks(k) {
                    let mut acc = [0.0f32; 4];
                    let (xc, xr) = xi.as_chunks::<4>();
                    let (wc, wr) = wj.as_chunks::<4>();
                    for (a, b) in xc.iter().zip(wc) {
                        for lane in 0..4 {
                            acc[lane] += a[lane].abs() * b[lane].abs();
                        }
                    }
                    let mut dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                    for (&a, &b) in xr.iter().zip(wr) {
                        dot += a.abs() * b.abs();
                    }
                    most = most.max(dot * scale);
                }
            }
            return most;
        }
        for xi in x.chunks(k) {
            for wj in w.chunks(k) {
                let mut bound = 0.0f32;
                for (&a, &b) in xi.iter().zip(wj) {
                    let (a, b) = (a.abs(), b.abs());
                    let ex = (a * eps).max(floor);
                    let ew = (b * eps).max(floor);
                    bound += a * ew + b * ex + a * b * adds;
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
            | Stmt::AtomicShared { .. }
            | Stmt::AtomicAdd { .. }
            | Stmt::Return
            | Stmt::Barrier
            | Stmt::TileAcc { .. }
            | Stmt::TileHeld { .. }
            | Stmt::TileHold { .. }
            | Stmt::TileZero { .. } => {}
        }
        Some(())
    }
}
