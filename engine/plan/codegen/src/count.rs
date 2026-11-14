//! A trip's operations, counted from the tree.

use crate::LANES;
use crate::form::Form;
use crate::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt, Ty};
use sim::Ops;
use std::collections::HashMap;

/// Bytes in a cache line.
const LINE: f64 = 64.0;

/// How an address moves with the loop variable, per trip.
#[derive(Clone, Copy, PartialEq)]
enum Advance {
    /// The same address every trip.
    Still,
    /// This many index units a trip.
    By(f64),
    /// Any other way: a new line every trip.
    Jumps,
}

impl Advance {
    fn add(self, other: Advance) -> Advance {
        match (self, other) {
            (Advance::Jumps, _) | (_, Advance::Jumps) => Advance::Jumps,
            (Advance::By(a), Advance::By(b)) => Advance::By(a + b),
            (Advance::By(a), Advance::Still) | (Advance::Still, Advance::By(a)) => Advance::By(a),
            (Advance::Still, Advance::Still) => Advance::Still,
        }
    }

    fn times(self, by: f64) -> Advance {
        match self {
            Advance::By(a) => Advance::By(a * by),
            other => other,
        }
    }
}

/// A runtime loop being counted: its variable and step.
struct Loop {
    var: String,
    step: Option<f64>,
    /// Names bound from the variable inside the loop.
    derived: Vec<String>,
}

/// Counts a trip's operations, inside the runtime loops.
#[derive(Default)]
struct Counter {
    trip: Ops,
    /// The runtime loops entered, innermost last.
    loops: Vec<Loop>,
    /// A body without runtime loops counts as one trip.
    straight: bool,
    /// The product of enclosing constant loop counts.
    times: f64,
    /// Freshness and width summed over loads, for their means.
    fresh: f64,
    width: f64,
    buffers: HashMap<String, &'static dyn Form>,
    shared: Vec<String>,
    tiles: HashMap<String, (usize, usize)>,
    /// Global loads this statement has made, which repeats share.
    seen: Vec<Expr>,
}

impl Counter {
    fn add(&mut self, ops: Ops) {
        if self.loops.is_empty() && !self.straight {
            return;
        }
        let (t, n) = (&mut self.trip, self.times);
        t.float += ops.float * n;
        t.int += ops.int * n;
        t.special += ops.special * n;
        t.loads += ops.loads * n;
        t.stores += ops.stores * n;
        t.tile_flops += ops.tile_flops * n;
        self.fresh += ops.fresh * ops.loads * n;
        self.width += ops.width * ops.loads * n;
    }

    /// A load at `at`: how often it opens a line.
    fn load(&mut self, at: &Expr, bytes: f64) {
        let fresh = match self.loops.last() {
            None => 1.0,
            Some(l) => match self.advance(at, l).times(l.step.unwrap_or(f64::NAN)) {
                Advance::Still => 0.0,
                Advance::By(units) if units.is_finite() => (units * bytes / LINE).min(1.0),
                _ => 1.0,
            },
        };
        self.add(Ops {
            loads: 1.0,
            fresh,
            width: bytes,
            ..Ops::default()
        });
    }

    /// How `e` moves per trip of `l`, in index units.
    fn advance(&self, e: &Expr, l: &Loop) -> Advance {
        match e {
            Expr::Var(name) if *name == l.var => Advance::By(1.0),
            Expr::Var(name) if l.derived.contains(name) => Advance::Jumps,
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => Advance::Still,
            Expr::Neg(a) => self.advance(a, l).times(-1.0),
            Expr::Cast(_, a) => self.advance(a, l),
            Expr::Bin(BinOp::Add, a, b) => self.advance(a, l).add(self.advance(b, l)),
            Expr::Bin(BinOp::Sub, a, b) => self.advance(a, l).add(self.advance(b, l).times(-1.0)),
            Expr::Bin(BinOp::Mul, a, b) => match (self.advance(a, l), self.advance(b, l)) {
                (Advance::Still, Advance::Still) => Advance::Still,
                (Advance::By(n), Advance::Still) => match &**b {
                    Expr::Int(c) => Advance::By(n * *c as f64),
                    _ => Advance::Jumps,
                },
                (Advance::Still, Advance::By(n)) => match &**a {
                    Expr::Int(c) => Advance::By(n * *c as f64),
                    _ => Advance::Jumps,
                },
                _ => Advance::Jumps,
            },
            Expr::Bin(..) | Expr::Call(..) | Expr::Fn(..) | Expr::Select(..) | Expr::Index(..) => {
                match self.mentions(e, l) {
                    true => Advance::Jumps,
                    false => Advance::Still,
                }
            }
            Expr::Not(a) | Expr::Field(a, _) => self.advance(a, l),
        }
    }

    /// Whether `e` reads the loop variable or a derivative.
    fn mentions(&self, e: &Expr, l: &Loop) -> bool {
        match e {
            Expr::Var(name) => *name == l.var || l.derived.contains(name),
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => false,
            Expr::Neg(a) | Expr::Not(a) | Expr::Cast(_, a) | Expr::Field(a, _) => {
                self.mentions(a, l)
            }
            Expr::Bin(_, a, b) | Expr::Index(a, b) => self.mentions(a, l) || self.mentions(b, l),
            Expr::Select(c, a, b) => [c, a, b].iter().any(|e| self.mentions(e, l)),
            Expr::Call(_, args) | Expr::Fn(_, _, args) => args.iter().any(|a| self.mentions(a, l)),
        }
    }

    fn global(&self, e: &Expr) -> Option<&'static dyn Form> {
        match e {
            Expr::Var(name) => self.buffers.get(name).copied(),
            _ => None,
        }
    }

    fn expr(&mut self, e: &Expr) -> bool {
        let float = |ops: f64| Ops {
            float: ops,
            ..Ops::default()
        };
        match e {
            Expr::Int(_) | Expr::Bool(_) | Expr::Var(_) => false,
            Expr::Float(_) => true,
            Expr::Neg(a) | Expr::Not(a) | Expr::Cast(_, a) => self.expr(a),
            Expr::Bin(op, a, b) => {
                let (fa, fb) = (self.expr(a), self.expr(b));
                let is_float = fa || fb || matches!(op, BinOp::Div) && (fa || fb);
                match (op.compares(), is_float) {
                    (true, _) => self.add(Ops {
                        int: 1.0,
                        ..Ops::default()
                    }),
                    (false, true) => self.add(float(1.0)),
                    (false, false) => self.add(Ops {
                        int: 1.0,
                        ..Ops::default()
                    }),
                }
                is_float
            }
            Expr::Index(base, at) => {
                self.expr(at);
                if let Some(form) = self.global(base) {
                    // The compiler loads a repeated element once.
                    if !self.seen.contains(e) {
                        self.seen.push(e.clone());
                        self.load(at, form.element_bytes());
                    }
                } else if matches!(&**base, Expr::Var(n) if self.shared.contains(n)) {
                    self.add(Ops {
                        loads: 1.0,
                        width: 4.0,
                        ..Ops::default()
                    });
                }
                true
            }
            Expr::Field(base, _) => {
                self.expr(base);
                true
            }
            Expr::Select(c, a, b) => {
                self.expr(c);
                let (fa, fb) = (self.expr(a), self.expr(b));
                self.add(Ops {
                    int: 1.0,
                    ..Ops::default()
                });
                fa || fb
            }
            Expr::Fn(_, ty, args) => {
                for a in args {
                    self.expr(a);
                }
                *ty == Ty::F32
            }
            Expr::Call(i, args) => {
                for a in args {
                    self.expr(a);
                }
                match i {
                    Intrinsic::Rsqrt
                    | Intrinsic::Exp
                    | Intrinsic::Tanh
                    | Intrinsic::Pow
                    | Intrinsic::Cos
                    | Intrinsic::Sin => {
                        self.add(Ops {
                            special: 1.0,
                            ..Ops::default()
                        });
                        true
                    }
                    Intrinsic::ShflXor | Intrinsic::Max => {
                        self.add(float(1.0));
                        true
                    }
                    Intrinsic::KeptRow => {
                        self.add(Ops {
                            int: 2.0,
                            ..Ops::default()
                        });
                        false
                    }
                    Intrinsic::Clamp => {
                        self.add(float(2.0));
                        true
                    }
                    Intrinsic::Sum32 => {
                        self.add(Ops {
                            float: 5.0,
                            int: 5.0,
                            ..Ops::default()
                        });
                        true
                    }
                    Intrinsic::Load4 => {
                        let bytes = self.global(&args[0]).map_or(4.0, |f| f.element_bytes());
                        let (at, mut l) = (&args[1], self.loops.pop());
                        // Four elements move the address by one of them.
                        let fresh = match &mut l {
                            None => 1.0,
                            Some(l) => {
                                match self.advance(at, l).times(l.step.unwrap_or(f64::NAN)) {
                                    Advance::Still => 0.0,
                                    Advance::By(u) if u.is_finite() => (u * bytes / LINE).min(1.0),
                                    _ => 1.0,
                                }
                            }
                        };
                        self.loops.extend(l);
                        self.add(Ops {
                            loads: 1.0,
                            fresh,
                            width: 4.0 * bytes,
                            ..Ops::default()
                        });
                        true
                    }
                    Intrinsic::Store4 => {
                        self.add(Ops {
                            stores: 1.0,
                            ..Ops::default()
                        });
                        false
                    }
                    Intrinsic::FromBits | Intrinsic::Make4 => true,
                    _ => false,
                }
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.seen.clear();
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, init } => {
                self.expr(init);
                // Names set from the loop variable move with it.
                if let Some(l) = self.loops.last()
                    && self.mentions(init, l)
                {
                    let name = name.clone();
                    self.loops.last_mut().expect("a loop").derived.push(name);
                }
            }
            Stmt::Array { .. } => {}
            Stmt::Shared { name, .. } => self.shared.push(name.clone()),
            Stmt::TileAcc { name, rows, cols } => {
                self.tiles.insert(name.clone(), (*rows, *cols));
            }
            Stmt::Assign { target, op, value } => {
                let is_float = self.expr(value);
                // `acc += a * b` is one multiply-add.
                let fused = matches!((op, value), (Some(BinOp::Add), Expr::Bin(BinOp::Mul, ..)));
                if op.is_some() && !fused {
                    self.add(match is_float {
                        true => Ops {
                            float: 1.0,
                            ..Ops::default()
                        },
                        false => Ops {
                            int: 1.0,
                            ..Ops::default()
                        },
                    });
                }
                match target {
                    Expr::Index(base, at) if self.global(base).is_some() => {
                        self.expr(at);
                        self.add(Ops {
                            stores: 1.0,
                            ..Ops::default()
                        });
                    }
                    Expr::Index(_, at) => {
                        self.expr(at);
                    }
                    _ => {}
                }
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                body,
                ..
            } => {
                let trips = match (from, to, step) {
                    (Expr::Int(a), Expr::Int(b), Expr::Int(s)) if *s > 0 => {
                        Some(((b - a).max(0) as f64 / *s as f64).ceil())
                    }
                    _ => None,
                };
                match trips {
                    Some(n) => {
                        let was = self.times;
                        self.times *= n;
                        self.stmts(body);
                        self.times = was;
                    }
                    None => {
                        let step = match step {
                            Expr::Int(s) => Some(*s as f64),
                            _ => None,
                        };
                        self.loops.push(Loop {
                            var: var.clone(),
                            step,
                            derived: Vec::new(),
                        });
                        self.add(Ops {
                            int: 2.0,
                            ..Ops::default()
                        });
                        self.stmts(body);
                        self.loops.pop();
                    }
                }
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.stmts(then);
                self.stmts(els);
            }
            Stmt::Return | Stmt::Barrier => {}
            Stmt::Expr(e) => {
                self.expr(e);
            }
            Stmt::TileBlock { acc, depth, .. } => {
                let Some((rows, cols)) = self.tiles.get(acc).copied() else {
                    panic!("`{acc}` is not a tile block");
                };
                // A lane's share of the block's multiply-adds, as tile flops.
                self.add(Ops {
                    tile_flops: (2 * rows * cols * depth) as f64 / LANES as f64,
                    float: (rows * cols) as f64 / LANES as f64,
                    loads: ((rows + cols) * depth) as f64 / LANES as f64,
                    width: 4.0,
                    ..Ops::default()
                });
            }
            Stmt::TileEach { body, .. } => self.stmts(body),
            Stmt::TileZero { .. } => {}
            Stmt::TileScale { acc, by } => {
                self.expr(by);
                if let Some((rows, cols)) = self.tiles.get(acc).copied() {
                    self.add(Ops {
                        float: (rows * cols) as f64 / LANES as f64,
                        ..Ops::default()
                    });
                }
            }
        }
    }
}

/// Whether any loop's bounds are read at launch.
fn runtime_loops(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::For {
            from,
            to,
            step,
            body,
            ..
        } => {
            let fixed = matches!((from, to, step), (Expr::Int(_), Expr::Int(_), Expr::Int(_)));
            !fixed || runtime_loops(body)
        }
        Stmt::If { then, els, .. } => runtime_loops(then) || runtime_loops(els),
        Stmt::TileEach { body, .. } => runtime_loops(body),
        _ => false,
    })
}

impl Body {
    /// One trip's operations: a runtime loop's, or the whole body's.
    pub fn trip(&self) -> Ops {
        let mut c = Counter {
            times: 1.0,
            straight: !runtime_loops(&self.stmts),
            ..Counter::default()
        };
        for param in &self.params {
            if let Param::Buffer { name, form, .. } = param {
                c.buffers.insert(name.clone(), *form);
            }
        }
        c.stmts(&self.stmts);
        let mut trip = c.trip;
        if trip.loads > 0.0 {
            trip.fresh = c.fresh / trip.loads;
            trip.width = c.width / trip.loads;
        }
        trip
    }
}
