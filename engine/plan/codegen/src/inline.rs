//! One kernel's tree inlined into another's, at build time.

use crate::ir::{Body, Expr, Helper, Param, Stmt};
use std::collections::HashMap;

/// A called kernel's argument: a build value or a tree.
pub trait Pass<T> {
    fn pass(self) -> T;
}

impl<T> Pass<T> for T {
    fn pass(self) -> T {
        self
    }
}

macro_rules! int_args {
    ($($t:ty),*) => {$(
        impl Pass<Expr> for $t {
            fn pass(self) -> Expr {
                Expr::Int(self as i64)
            }
        }
    )*};
}

int_args!(i32, i64, u8, u16, u32, u64, usize);

impl Pass<Expr> for f32 {
    fn pass(self) -> Expr {
        Expr::Float(f64::from(self))
    }
}

impl Pass<Expr> for f64 {
    fn pass(self) -> Expr {
        Expr::Float(self)
    }
}

impl Pass<Expr> for bool {
    fn pass(self) -> Expr {
        Expr::Bool(self)
    }
}

/// A call's target: a kernel to inline, or a helper.
pub enum Callee {
    Kernel(Body),
    Helper(Helper),
}

/// A call: the callee and the caller's arguments.
pub struct Inline {
    pub callee: Callee,
    pub args: Vec<Expr>,
}

/// Numbers a body's calls, keeping inlined names apart.
#[derive(Default)]
pub struct Inliner {
    calls: usize,
    /// The helpers called so far, each once.
    pub helpers: Vec<Helper>,
}

impl Inliner {
    /// A call as a statement. Its statements go into `pre`.
    pub fn stmt(&mut self, pre: &mut Vec<Stmt>, call: Inline) {
        self.inline(pre, call);
    }

    /// A call as a value, after the statements in `pre`.
    pub fn expr(&mut self, pre: &mut Vec<Stmt>, call: Inline) -> Expr {
        let entry = match &call.callee {
            Callee::Kernel(body) => body.entry.clone(),
            Callee::Helper(h) => h.name.clone(),
        };
        match self.inline(pre, call) {
            Some(value) => value,
            None => panic!("`{entry}` has no value"),
        }
    }

    fn inline(&mut self, pre: &mut Vec<Stmt>, call: Inline) -> Option<Expr> {
        self.calls += 1;
        let Inline { callee, args } = call;
        let body = match callee {
            Callee::Kernel(body) => body,
            Callee::Helper(h) => {
                let took = h.body.params.len();
                assert_eq!(args.len(), took, "`{}` takes {took} arguments", h.name);
                let value = Expr::Fn(h.name.clone(), h.ret, args);
                self.helper(h);
                return Some(value);
            }
        };
        assert_eq!(
            args.len(),
            body.params.len(),
            "`{}` takes {} arguments",
            body.entry,
            body.params.len()
        );
        assert!(
            !returns(&body.stmts),
            "`{}` returns; a called kernel runs to its end",
            body.entry
        );
        let mut names = HashMap::new();
        for (param, arg) in body.params.iter().zip(args) {
            match param {
                // Buffers, group memory, tiles and arrays pass by name.
                Param::Buffer { name, .. }
                | Param::Shared { name, .. }
                | Param::Tiles { name }
                | Param::Array { name, .. } => {
                    let Expr::Var(named) = arg else {
                        panic!("`{}` takes `{name}` by name", body.entry);
                    };
                    names.insert(name.clone(), named);
                }
                // A name the callee never assigns reads as itself.
                Param::Scalar { name, .. }
                    if matches!(arg, Expr::Var(_)) && !assigns(&body.stmts, name) =>
                {
                    let Expr::Var(named) = arg else {
                        unreachable!()
                    };
                    names.insert(name.clone(), named);
                }
                Param::Scalar { name, ty } => {
                    let fresh = self.fresh(name);
                    pre.push(Stmt::Let {
                        name: fresh.clone(),
                        init: Expr::cast(*ty, arg),
                    });
                    names.insert(name.clone(), fresh);
                }
            }
        }
        let mut declared = Vec::new();
        declares(&body.stmts, &mut declared);
        for name in declared {
            let fresh = self.fresh(&name);
            names.insert(name, fresh);
        }
        let renamer = Rename { names };
        pre.extend(body.stmts.iter().map(|s| renamer.stmt(s)));
        for h in body.helpers {
            self.helper(h);
        }
        body.ret.as_ref().map(|e| renamer.expr(e))
    }

    fn helper(&mut self, h: Helper) {
        if !self.helpers.iter().any(|k| k.name == h.name) {
            self.helpers.push(h);
        }
    }

    fn fresh(&self, name: &str) -> String {
        format!("{name}_{}", self.calls)
    }
}

/// Whether any statement returns from the kernel.
fn returns(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Return => true,
        Stmt::For { body, .. } | Stmt::TileEach { body, .. } => returns(body),
        Stmt::If { then, els, .. } => returns(then) || returns(els),
        _ => false,
    })
}

/// Whether any statement assigns to `name`.
fn assigns(stmts: &[Stmt], name: &str) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Assign { target, .. } => matches!(target, Expr::Var(n) if n == name),
        Stmt::For { body, .. } | Stmt::TileEach { body, .. } => assigns(body, name),
        Stmt::If { then, els, .. } => assigns(then, name) || assigns(els, name),
        _ => false,
    })
}

/// Every name the statements bind, nested ones included.
fn declares(stmts: &[Stmt], out: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Let { name, .. }
            | Stmt::Array { name, .. }
            | Stmt::Shared { name, .. }
            | Stmt::TileAcc { name, .. } => out.push(name.clone()),
            Stmt::For { var, body, .. } => {
                out.push(var.clone());
                declares(body, out);
            }
            Stmt::If { then, els, .. } => {
                declares(then, out);
                declares(els, out);
            }
            Stmt::TileEach { body, .. } => declares(body, out),
            _ => {}
        }
    }
}

/// Rewrites a tree with the callee's names replaced.
struct Rename {
    names: HashMap<String, String>,
}

impl Rename {
    fn name(&self, name: &str) -> String {
        self.names.get(name).cloned().unwrap_or_else(|| name.into())
    }

    fn block(&self, stmts: &[Stmt]) -> Vec<Stmt> {
        stmts.iter().map(|s| self.stmt(s)).collect()
    }

    fn expr(&self, e: &Expr) -> Expr {
        let sub = |e: &Expr| Box::new(self.expr(e));
        match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => e.clone(),
            Expr::Var(name) => Expr::Var(self.name(name)),
            Expr::Neg(a) => Expr::Neg(sub(a)),
            Expr::Not(a) => Expr::Not(sub(a)),
            Expr::Bin(op, a, b) => Expr::Bin(*op, sub(a), sub(b)),
            Expr::Cast(ty, a) => Expr::Cast(*ty, sub(a)),
            Expr::Index(base, at) => Expr::Index(sub(base), sub(at)),
            Expr::Field(base, h) => Expr::Field(sub(base), *h),
            Expr::Call(i, args) => Expr::Call(*i, args.iter().map(|a| self.expr(a)).collect()),
            Expr::Select(c, a, b) => Expr::Select(sub(c), sub(a), sub(b)),
            Expr::Fn(name, ty, args) => Expr::Fn(
                name.clone(),
                *ty,
                args.iter().map(|a| self.expr(a)).collect(),
            ),
        }
    }

    fn stmt(&self, s: &Stmt) -> Stmt {
        match s {
            Stmt::Let { name, init } => Stmt::Let {
                name: self.name(name),
                init: self.expr(init),
            },
            Stmt::Array { name, ty, len } => Stmt::Array {
                name: self.name(name),
                ty: *ty,
                len: *len,
            },
            Stmt::Shared { name, ty, len } => Stmt::Shared {
                name: self.name(name),
                ty: *ty,
                len: *len,
            },
            Stmt::Assign { target, op, value } => Stmt::Assign {
                target: self.expr(target),
                op: *op,
                value: self.expr(value),
            },
            Stmt::For {
                var,
                from,
                to,
                step,
                unroll,
                body,
            } => Stmt::For {
                var: self.name(var),
                from: self.expr(from),
                to: self.expr(to),
                step: self.expr(step),
                unroll: *unroll,
                body: self.block(body),
            },
            Stmt::If { cond, then, els } => Stmt::If {
                cond: self.expr(cond),
                then: self.block(then),
                els: self.block(els),
            },
            Stmt::Return => Stmt::Return,
            Stmt::Barrier => Stmt::Barrier,
            Stmt::Expr(e) => Stmt::Expr(self.expr(e)),
            Stmt::TileAcc { name, rows, cols } => Stmt::TileAcc {
                name: self.name(name),
                rows: *rows,
                cols: *cols,
            },
            Stmt::TileBlock {
                acc,
                a,
                a_at,
                b,
                b_at,
                stride,
                depth,
            } => Stmt::TileBlock {
                acc: self.name(acc),
                a: self.expr(a),
                a_at: self.expr(a_at),
                b: self.expr(b),
                b_at: self.expr(b_at),
                stride: self.expr(stride),
                depth: *depth,
            },
            Stmt::TileEach { acc, body } => Stmt::TileEach {
                acc: self.name(acc),
                body: self.block(body),
            },
            Stmt::TileZero { acc } => Stmt::TileZero {
                acc: self.name(acc),
            },
            Stmt::TileScale { acc, by } => Stmt::TileScale {
                acc: self.name(acc),
                by: self.expr(by),
            },
        }
    }
}
