//! Per-thread lowering for independent thread bodies.

use crate::expr::{Val, clif, elem_size};
use crate::lower::{Kind, Lowerer, Res};
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{Block, InstBuilder, Value, types};
use std::collections::HashMap;
use tessera_codegen::ir::{Expr, Intrinsic, Stmt, Ty};

/// Whether `e` stays within a single thread.
fn expr_independent(e: &Expr) -> bool {
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => true,
        Expr::Neg(a) | Expr::Not(a) | Expr::Cast(_, a) | Expr::Field(a, _) => expr_independent(a),
        Expr::Bin(_, a, b) | Expr::Index(a, b) => expr_independent(a) && expr_independent(b),
        Expr::Select(c, a, b) => expr_independent(c) && expr_independent(a) && expr_independent(b),
        Expr::Fn(_, _, args) => args.iter().all(expr_independent),
        Expr::Call(Intrinsic::ShflXor | Intrinsic::Sum32, _) => false,
        Expr::Call(_, args) => args.iter().all(expr_independent),
    }
}

/// Whether every thread in a group can run `stmts` independently.
pub(crate) fn is_independent(stmts: &[Stmt]) -> bool {
    stmts.iter().all(|s| match s {
        Stmt::Let { init, .. } | Stmt::Expr(init) => expr_independent(init),
        Stmt::Array { .. } | Stmt::Return => true,
        Stmt::Assign { target, value, .. } => expr_independent(target) && expr_independent(value),
        Stmt::For {
            from,
            to,
            step,
            body,
            ..
        } => {
            expr_independent(from)
                && expr_independent(to)
                && expr_independent(step)
                && is_independent(body)
        }
        Stmt::If { cond, then, els } => {
            expr_independent(cond) && is_independent(then) && is_independent(els)
        }
        Stmt::Shared { .. }
        | Stmt::AtomicShared { .. }
        | Stmt::AtomicAdd { .. }
        | Stmt::Barrier
        | Stmt::TileAcc { .. }
        | Stmt::TileHeld { .. }
        | Stmt::TileZero { .. }
        | Stmt::TileScale { .. }
        | Stmt::TileHold { .. }
        | Stmt::TileBlock { .. }
        | Stmt::TileEach { .. } => false,
    })
}

impl Lowerer<'_, '_> {
    /// Runs `stmts` for each thread `0..n` independently in SSA registers.
    pub(crate) fn run_independent(&mut self, stmts: &[Stmt]) -> Res {
        let (header, body, next, exit) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        let t = self.b.declare_var(self.ptr);
        let zero = self.iconst(self.ptr, 0);
        self.b.def_var(t, zero);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let tv = self.b.use_var(t);
        let more = self.b.ins().icmp(IntCC::UnsignedLessThan, tv, self.n);
        self.b.ins().brif(more, body, &[], exit, &[]);
        self.b.switch_to_block(body);
        if !self.scalar_block(stmts, tv, next)? {
            self.b.ins().jump(next, &[]);
        }
        self.b.switch_to_block(next);
        let one = self.iconst(self.ptr, 1);
        let step = self.b.ins().iadd(tv, one);
        self.b.def_var(t, step);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(exit);
        Ok(())
    }

    fn scalar_block(&mut self, stmts: &[Stmt], t: Value, next_thread: Block) -> Res<bool> {
        self.scope.push(HashMap::new());
        let done = self.scalar_stmts(stmts, t, next_thread);
        self.scope.pop();
        done
    }

    fn scalar_stmts(&mut self, stmts: &[Stmt], t: Value, next_thread: Block) -> Res<bool> {
        for s in stmts {
            if self.scalar_stmt(s, t, next_thread)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn scalar_stmt(&mut self, s: &Stmt, t: Value, next_thread: Block) -> Res<bool> {
        match s {
            Stmt::Let { name, init } => {
                let ty = self.ty(init)?;
                let val = self.expr_as(init, ty, t)?;
                match (ty, val) {
                    (Ty::F32x4, Val::V(parts)) => {
                        let vars = parts.map(|p| {
                            let var = self.b.declare_var(types::F32);
                            self.b.def_var(var, p);
                            var
                        });
                        self.bind(name, Kind::Reg4(vars));
                    }
                    (_, Val::S(v)) => {
                        let var = self.b.declare_var(clif(ty));
                        self.b.def_var(var, v);
                        self.bind(name, Kind::Reg(ty, var));
                    }
                    (ty, _) => return Err(format!("bad init for {ty:?}")),
                }
                Ok(false)
            }
            Stmt::Array { name, ty, len } => {
                let slot = self.slot(elem_size(*ty) * len);
                self.bind(name, Kind::LocalArray(*ty, slot));
                Ok(false)
            }
            Stmt::Assign { target, op, value } => {
                self.store(target, *op, value, t)?;
                Ok(false)
            }
            Stmt::Expr(e) => {
                self.expr(e, t)?;
                Ok(false)
            }
            Stmt::Return => {
                self.b.ins().jump(next_thread, &[]);
                Ok(true)
            }
            Stmt::If {
                cond: Expr::Bool(taken),
                then,
                els,
            } => {
                let branch = if *taken { then } else { els };
                self.scalar_block(branch, t, next_thread)
            }
            Stmt::If { cond, then, els } => {
                let Val::S(c) = self.expr_as(cond, Ty::Bool, t)? else {
                    return Err("a four-wide condition".into());
                };
                if els.is_empty() {
                    let (yes, after) = (self.b.create_block(), self.b.create_block());
                    self.b.ins().brif(c, yes, &[], after, &[]);
                    self.b.switch_to_block(yes);
                    if !self.scalar_block(then, t, next_thread)? {
                        self.b.ins().jump(after, &[]);
                    }
                    self.b.switch_to_block(after);
                    return Ok(false);
                }
                let (yes, no, after) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().brif(c, yes, &[], no, &[]);
                self.b.switch_to_block(yes);
                let r1 = self.scalar_block(then, t, next_thread)?;
                if !r1 {
                    self.b.ins().jump(after, &[]);
                }
                self.b.switch_to_block(no);
                let r2 = self.scalar_block(els, t, next_thread)?;
                if !r2 {
                    self.b.ins().jump(after, &[]);
                }
                if r1 && r2 {
                    return Ok(true);
                }
                self.b.switch_to_block(after);
                Ok(false)
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                body,
                ..
            } => {
                let Val::S(start) = self.expr_as(from, Ty::U32, t)? else {
                    return Err("a four-wide loop start".into());
                };
                let Val::S(end) = self.expr_as(to, Ty::U32, t)? else {
                    return Err("a four-wide loop end".into());
                };
                let Val::S(by) = self.expr_as(step, Ty::U32, t)? else {
                    return Err("a four-wide loop step".into());
                };
                let counter = self.b.declare_var(types::I32);
                self.b.def_var(counter, start);
                let (header, run, exit) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().jump(header, &[]);
                self.b.switch_to_block(header);
                let i = self.b.use_var(counter);
                let more = self.b.ins().icmp(IntCC::UnsignedLessThan, i, end);
                self.b.ins().brif(more, run, &[], exit, &[]);
                self.b.switch_to_block(run);
                self.scope.push(HashMap::new());
                self.bind(var, Kind::Reg(Ty::U32, counter));
                let ret = self.scalar_stmts(body, t, next_thread);
                self.scope.pop();
                if !ret? {
                    let i = self.b.use_var(counter);
                    let next = self.b.ins().iadd(i, by);
                    self.b.def_var(counter, next);
                    self.b.ins().jump(header, &[]);
                }
                self.b.switch_to_block(exit);
                Ok(false)
            }
            other => Err(format!("{other:?} requires lockstep lowering")),
        }
    }
}
