//! A trip's operations, counted from the tree.

use crate::LANES;
use crate::ir::{BinOp, Body, Expr, Intrinsic, Param, Stmt};
use graph::DType;
use sim::Ops;
use std::collections::HashMap;

/// Counts a trip's operations, inside the runtime loops.
#[derive(Default)]
struct Counter {
    trip: Ops,
    /// Runtime loops entered.
    looping: usize,
    /// The product of enclosing constant loop counts.
    times: f64,
    buffers: HashMap<String, DType>,
    shared: Vec<String>,
    tiles: HashMap<String, (usize, usize)>,
}

impl Counter {
    fn add(&mut self, ops: Ops) {
        if self.looping == 0 {
            return;
        }
        let (t, n) = (&mut self.trip, self.times);
        t.float += ops.float * n;
        t.int += ops.int * n;
        t.special += ops.special * n;
        t.loads += ops.loads * n;
        t.stores += ops.stores * n;
        t.tile_flops += ops.tile_flops * n;
        // Loads of one trip share one width and freshness.
        if ops.loads > 0.0 {
            t.width = ops.width;
            t.fresh = ops.fresh;
        }
    }

    fn global(&self, e: &Expr) -> Option<DType> {
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
                if let Some(dtype) = self.global(base) {
                    self.add(Ops {
                        loads: 1.0,
                        width: dtype.bytes() as f64,
                        ..Ops::default()
                    });
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
                        let width = 4.0 * self.global(&args[0]).map_or(4, DType::bytes) as f64;
                        self.add(Ops {
                            loads: 1.0,
                            fresh: 1.0,
                            width,
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
                    _ => false,
                }
            }
        }
    }

    fn stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { init, .. } => {
                self.expr(init);
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
                        self.looping += 1;
                        self.add(Ops {
                            int: 2.0,
                            ..Ops::default()
                        });
                        self.stmts(body);
                        self.looping -= 1;
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

impl Body {
    /// One trip's operations, from the loops that read a size.
    pub fn trip(&self) -> Ops {
        let mut c = Counter {
            times: 1.0,
            ..Counter::default()
        };
        for param in &self.params {
            if let Param::Buffer { name, dtype, .. } = param {
                c.buffers.insert(name.clone(), *dtype);
            }
        }
        c.stmts(&self.stmts);
        c.trip
    }
}
