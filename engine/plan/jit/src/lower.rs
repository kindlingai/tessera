//! Statements: a loop over live lanes, or a new mask.

use crate::expr::{Val, elem_size};
use crate::runtime::{Arg, Buf, Ctx};
use crate::signature;
use cranelift_codegen::ir::{
    FuncRef, Function, InstBuilder, MemFlagsData, StackSlot, StackSlotData, StackSlotKind, Type,
    Value, condcodes::IntCC, types,
};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use tessera_codegen::ir::{Body, Expr, Intrinsic, Param, Stmt, Ty};
use tessera_host::lockstep::THREADS;

pub(crate) type Res<T = ()> = Result<T, String>;

/// Flags for the function's own stack and the launch's buffers.
pub(crate) fn trusted() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// The kind of a name in scope.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// One value per thread, in a lane array.
    Lanes(Ty, StackSlot),
    /// One value, the same for every thread.
    One(Ty, Value),
    Buffer {
        form: &'static dyn tessera_codegen::Load,
        ptr: Value,
        len: Value,
        skip: Value,
    },
    /// A local array per thread, `len` long.
    Array(Ty, StackSlot, usize),
    /// Group memory, `len` long.
    Shared(Ty, StackSlot, usize),
    Tiles(Value),
}

pub(crate) struct Lowerer<'a, 'b> {
    pub(crate) b: FunctionBuilder<'b>,
    module: &'a mut JITModule,
    frontend: TargetFrontendConfig,
    pub(crate) ptr: Type,
    helpers: HashMap<&'static str, FuncRef>,
    scope: Vec<HashMap<String, Kind>>,
    /// Masks of the enclosing blocks, outermost first.
    masks: Vec<StackSlot>,
    pub(crate) n: Value,
    pub(crate) lanes: Value,
    pub(crate) first: Variable,
    precision: Value,
    /// Tile sums to free when the group ends.
    tiles: Vec<Value>,
}

impl Lowerer<'_, '_> {
    pub(crate) fn kind(&self, name: &str) -> Res<Kind> {
        for scope in self.scope.iter().rev() {
            if let Some(k) = scope.get(name) {
                return Ok(*k);
            }
        }
        Err(format!("`{name}` is not in scope"))
    }

    pub(crate) fn bind(&mut self, name: &str, kind: Kind) {
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), kind);
    }

    pub(crate) fn mask(&self) -> StackSlot {
        *self.masks.last().expect("a mask")
    }

    /// A call to helper `name`, declared on first use.
    pub(crate) fn call(&mut self, name: &'static str, args: &[Value]) -> Option<Value> {
        let fref = match self.helpers.get(name) {
            Some(f) => *f,
            None => {
                let sig = signature(name, self.ptr, self.frontend.default_call_conv);
                let id = self
                    .module
                    .declare_function(name, Linkage::Import, &sig)
                    .expect("a helper declares");
                let f = self.module.declare_func_in_func(id, self.b.func);
                self.helpers.insert(name, f);
                f
            }
        };
        let inst = self.b.ins().call(fref, args);
        self.b.inst_results(inst).first().copied()
    }

    pub(crate) fn iconst(&mut self, ty: Type, v: i64) -> Value {
        self.b.ins().iconst(ty, v)
    }

    /// A stack slot of `bytes`, zeroed.
    pub(crate) fn slot(&mut self, bytes: usize) -> StackSlot {
        let data = StackSlotData::new(StackSlotKind::ExplicitSlot, bytes as u32, 4);
        let slot = self.b.create_sized_stack_slot(data);
        let addr = self.b.ins().stack_addr(self.ptr, slot, 0);
        let flags = MemFlagsData::trusted();
        self.b
            .emit_small_memset(self.frontend, addr, 0, bytes as u64, 16, flags);
        slot
    }

    /// A lane array of `ty`.
    pub(crate) fn lane_slot(&mut self, ty: Ty) -> StackSlot {
        self.slot(elem_size(ty) * THREADS)
    }

    /// The address of element `i` of `slot`, `size` bytes each.
    pub(crate) fn at(&mut self, slot: StackSlot, i: Value, size: usize) -> Value {
        let base = self.b.ins().stack_addr(self.ptr, slot, 0);
        let size = self.iconst(self.ptr, size as i64);
        let off = self.b.ins().imul(i, size);
        self.b.ins().iadd(base, off)
    }

    pub(crate) fn slot_addr(&mut self, slot: StackSlot) -> Value {
        self.b.ins().stack_addr(self.ptr, slot, 0)
    }

    /// Whether lane `t` of `mask` is live.
    fn live(&mut self, mask: StackSlot, t: Value) -> Value {
        let addr = self.at(mask, t, 1);
        let m = self.b.ins().load(types::I8, trusted(), addr, 0);
        self.b.ins().icmp_imm_u(IntCC::NotEqual, m, 0)
    }

    fn set_mask(&mut self, mask: StackSlot, t: Value, on: Value) {
        let addr = self.at(mask, t, 1);
        self.b.ins().store(trusted(), on, addr, 0);
    }

    /// `f` per lane under `mask`, or per lane when none.
    pub(crate) fn lanes<F>(&mut self, mask: Option<StackSlot>, f: F) -> Res
    where
        F: FnOnce(&mut Self, Value) -> Res,
    {
        let (header, test, body, next, exit) = (
            self.b.create_block(),
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
        self.b.ins().brif(more, test, &[], exit, &[]);
        self.b.switch_to_block(test);
        match mask {
            Some(mask) => {
                let on = self.live(mask, tv);
                self.b.ins().brif(on, body, &[], next, &[]);
            }
            None => {
                self.b.ins().jump(body, &[]);
            }
        }
        self.b.switch_to_block(body);
        f(self, tv)?;
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
        let one = self.iconst(self.ptr, 1);
        let step = self.b.ins().iadd(tv, one);
        self.b.def_var(t, step);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(exit);
        Ok(())
    }

    /// `e` under the mask, in a fresh array of `ty`.
    fn lane_array(&mut self, e: &Expr, ty: Ty) -> Res<StackSlot> {
        let e = self.hoist(e)?;
        let slot = self.lane_slot(ty);
        let mask = self.mask();
        self.lanes(Some(mask), |s, t| {
            let v = s.expr_as(&e, ty, t)?;
            s.store_lane(slot, ty, t, v);
            Ok(())
        })?;
        Ok(slot)
    }

    /// Whether element `at` fits in `len` bytes, and its address.
    pub(crate) fn guarded(&mut self, [ptr, len, at]: [Value; 3], bytes: i64) -> [Value; 2] {
        let size = self.iconst(self.ptr, bytes);
        let off = self.b.ins().imul(at, size);
        let end = self.b.ins().iadd(off, size);
        let fits = self.b.ins().icmp(IntCC::UnsignedLessThanOrEqual, end, len);
        [fits, self.b.ins().iadd(ptr, off)]
    }

    /// `fast` where the element fits, else the checking helper.
    pub(crate) fn either(
        &mut self,
        fits: Value,
        fast: impl FnOnce(&mut Self) -> Option<Value>,
        slow: impl FnOnce(&mut Self) -> Option<Value>,
        ty: Option<Type>,
    ) -> Option<Value> {
        let (yes, no, merge) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        if let Some(ty) = ty {
            self.b.append_block_param(merge, ty);
        }
        self.b.ins().brif(fits, yes, &[], no, &[]);
        self.b.switch_to_block(yes);
        let v = fast(self);
        let args: Vec<_> = v.into_iter().map(Into::into).collect();
        self.b.ins().jump(merge, &args);
        self.b.switch_to_block(no);
        let v = slow(self);
        let args: Vec<_> = v.into_iter().map(Into::into).collect();
        self.b.ins().jump(merge, &args);
        self.b.switch_to_block(merge);
        self.b.block_params(merge).first().copied()
    }

    /// A lane array's element `t` of `ty`.
    pub(crate) fn load_lane(&mut self, slot: StackSlot, ty: Ty, t: Value) -> Val {
        let addr = self.at(slot, t, elem_size(ty));
        self.load_at(addr, ty)
    }

    pub(crate) fn store_lane(&mut self, slot: StackSlot, ty: Ty, t: Value, v: Val) {
        let addr = self.at(slot, t, elem_size(ty));
        self.store_at(addr, ty, v);
    }

    pub(crate) fn load_at(&mut self, addr: Value, ty: Ty) -> Val {
        let flags = trusted();
        match ty {
            Ty::F32x4 => {
                let part = |s: &mut Self, i: i32| s.b.ins().load(types::F32, flags, addr, 4 * i);
                Val::V([part(self, 0), part(self, 1), part(self, 2), part(self, 3)])
            }
            _ => Val::S(self.b.ins().load(crate::expr::clif(ty), flags, addr, 0)),
        }
    }

    pub(crate) fn store_at(&mut self, addr: Value, ty: Ty, v: Val) {
        let flags = trusted();
        match (ty, v) {
            (Ty::F32x4, Val::V(parts)) => {
                for (i, p) in parts.into_iter().enumerate() {
                    self.b.ins().store(flags, p, addr, 4 * i as i32);
                }
            }
            (_, Val::S(v)) => {
                self.b.ins().store(flags, v, addr, 0);
            }
            (ty, _) => panic!("a four-wide value stored as {ty:?}"),
        }
    }

    /// `stmts` under `mask`, in their own scope.
    fn block(&mut self, stmts: &[Stmt], mask: StackSlot) -> Res {
        self.scope.push(HashMap::new());
        self.masks.push(mask);
        for s in stmts {
            self.stmt(s)?;
        }
        self.masks.pop();
        self.scope.pop();
        Ok(())
    }

    /// `target = value` for lane `t`, or `target op= value`.
    fn store(
        &mut self,
        target: &Expr,
        op: Option<tessera_codegen::ir::BinOp>,
        value: &Expr,
        t: Value,
    ) -> Res {
        let combined = match op {
            Some(op) => Expr::Bin(op, Box::new(target.clone()), Box::new(value.clone())),
            None => value.clone(),
        };
        match target {
            Expr::Var(name) => {
                let Kind::Lanes(ty, slot) = self.kind(name)? else {
                    return Err(format!("`{name}` is not a lane variable"));
                };
                let v = self.expr_as(&combined, ty, t)?;
                self.store_lane(slot, ty, t, v);
                Ok(())
            }
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    return Err("an index names what it writes".into());
                };
                match self.kind(name)? {
                    Kind::Buffer {
                        form,
                        ptr,
                        len,
                        skip,
                    } => {
                        let idx = self.index(at, t)?;
                        let at = self.b.ins().iadd(skip, idx);
                        match form.read() {
                            tessera_codegen::form::Read::F32 | tessera_codegen::form::Read::U32 => {
                                let (st, ty) = match form.read() {
                                    tessera_codegen::form::Read::F32 => ("st_f32", Ty::F32),
                                    _ => ("st_u32", Ty::U32),
                                };
                                let Val::S(v) = self.expr_as(&combined, ty, t)? else {
                                    return Err("a four-wide value stored by index".into());
                                };
                                // Stores in range skip the helper and its checks.
                                let [fits, addr] = self.guarded([ptr, len, at], 4);
                                let store = |s: &mut Self| {
                                    s.b.ins().store(MemFlagsData::new(), v, addr, 0);
                                    None
                                };
                                let check = |s: &mut Self| s.call(st, &[ptr, len, at, v]);
                                self.either(fits, store, check, None);
                            }
                            tessera_codegen::form::Read::Bf16
                            | tessera_codegen::form::Read::Fp8 => {
                                let st = match form.read() {
                                    tessera_codegen::form::Read::Bf16 => "st_bf16",
                                    _ => "st_fp8",
                                };
                                let Val::S(v) = self.expr_as(&combined, Ty::F32, t)? else {
                                    return Err("a four-wide value stored by index".into());
                                };
                                self.call(st, &[ptr, len, at, v]);
                            }
                            tessera_codegen::form::Read::U8 => {
                                let Val::S(v) = self.expr_as(&combined, Ty::U32, t)? else {
                                    return Err("a four-wide value stored by index".into());
                                };
                                self.call("st_u8", &[ptr, len, at, v]);
                            }
                            other => return Err(format!("cannot write form {other:?}")),
                        }
                        Ok(())
                    }
                    Kind::Shared(ty, slot, _) => {
                        let idx = self.index(at, t)?;
                        let addr = self.at(slot, idx, elem_size(ty));
                        let v = self.expr_as(&combined, ty, t)?;
                        self.store_at(addr, ty, v);
                        Ok(())
                    }
                    Kind::Array(ty, slot, len) => {
                        let addr = self.array_at(slot, ty, len, t, at)?;
                        let v = self.expr_as(&combined, ty, t)?;
                        self.store_at(addr, ty, v);
                        Ok(())
                    }
                    _ => Err(format!("indexed writes to `{name}` are unsupported")),
                }
            }
            Expr::Field(base, h) => {
                let Expr::Var(name) = &**base else {
                    return Err("a field names its value".into());
                };
                let Kind::Lanes(Ty::F32x4, slot) = self.kind(name)? else {
                    return Err(format!("`{name}` is not four wide"));
                };
                let i = ['x', 'y', 'z', 'w']
                    .iter()
                    .position(|c| c == h)
                    .unwrap_or(3);
                let addr = self.at(slot, t, 16);
                let Val::S(v) = self.expr_as(&combined, Ty::F32, t)? else {
                    return Err("a four-wide value in a field".into());
                };
                self.b.ins().store(trusted(), v, addr, 4 * i as i32);
                Ok(())
            }
            other => Err(format!("cannot assign to {other:?}")),
        }
    }

    /// Element `at` of lane `t`'s local array.
    pub(crate) fn array_at(
        &mut self,
        slot: StackSlot,
        ty: Ty,
        len: usize,
        t: Value,
        at: &Expr,
    ) -> Res<Value> {
        let idx = self.index(at, t)?;
        let len = self.iconst(self.ptr, len as i64);
        let row = self.b.ins().imul(t, len);
        let i = self.b.ins().iadd(row, idx);
        Ok(self.at(slot, i, elem_size(ty)))
    }

    /// Clears every mask where the current one is live.
    fn leave(&mut self) -> Res {
        let (mask, outer) = (self.mask(), self.masks.clone());
        self.lanes(Some(mask), |s, t| {
            let off = s.iconst(types::I8, 0);
            for m in outer {
                s.set_mask(m, t, off);
            }
            Ok(())
        })
    }

    /// `cond` under the mask: lanes where it holds, and not.
    fn split(&mut self, cond: &Expr) -> Res<(StackSlot, StackSlot, Value, Value)> {
        let cond = self.hoist(cond)?;
        let (yes, no) = (self.lane_slot(Ty::Bool), self.lane_slot(Ty::Bool));
        let (any_yes, any_no) = (self.b.declare_var(types::I8), self.b.declare_var(types::I8));
        let off = self.iconst(types::I8, 0);
        self.b.def_var(any_yes, off);
        self.b.def_var(any_no, off);
        let mask = self.mask();
        self.lanes(Some(mask), |s, t| {
            let Val::S(c) = s.expr_as(&cond, Ty::Bool, t)? else {
                return Err("a four-wide condition".into());
            };
            let not = s.b.ins().bxor_imm_u(c, 1);
            s.set_mask(yes, t, c);
            s.set_mask(no, t, not);
            let (y, n) = (s.b.use_var(any_yes), s.b.use_var(any_no));
            let (y, n) = (s.b.ins().bor(y, c), s.b.ins().bor(n, not));
            s.b.def_var(any_yes, y);
            s.b.def_var(any_no, n);
            Ok(())
        })?;
        let (y, n) = (self.b.use_var(any_yes), self.b.use_var(any_no));
        Ok((yes, no, y, n))
    }

    /// `stmts` under `mask`, skipped when `any` is zero.
    fn branch(&mut self, any: Value, stmts: &[Stmt], mask: StackSlot) -> Res {
        let (taken, after) = (self.b.create_block(), self.b.create_block());
        self.b.ins().brif(any, taken, &[], after, &[]);
        self.b.switch_to_block(taken);
        self.block(stmts, mask)?;
        self.b.ins().jump(after, &[]);
        self.b.switch_to_block(after);
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Res {
        let mask = self.mask();
        match s {
            Stmt::Let { name, init } => {
                let ty = self.ty(init)?;
                let slot = self.lane_array(init, ty)?;
                self.bind(name, Kind::Lanes(ty, slot));
                Ok(())
            }
            Stmt::Array { name, ty, len } => {
                let slot = self.slot(elem_size(*ty) * len * THREADS);
                self.bind(name, Kind::Array(*ty, slot, *len));
                Ok(())
            }
            Stmt::Shared { name, ty, len } => {
                let slot = self.slot(elem_size(*ty) * len);
                self.bind(name, Kind::Shared(*ty, slot, *len));
                Ok(())
            }
            Stmt::AtomicShared { name, len } => {
                let slot = self.slot(elem_size(Ty::U32) * len);
                self.bind(name, Kind::Shared(Ty::U32, slot, *len));
                Ok(())
            }
            Stmt::AtomicAdd { name, at, value } => {
                let (at, value) = (self.hoist(at)?, self.hoist(value)?);
                let target = Expr::Index(Box::new(Expr::Var(name.clone())), Box::new(at));
                self.lanes(Some(mask), |s, t| {
                    s.store(&target, Some(tessera_codegen::ir::BinOp::Add), &value, t)
                })
            }
            Stmt::Assign {
                target,
                op,
                value: Expr::Call(Intrinsic::Sum32, args),
            } => {
                if op.is_some() || args[0] != *target {
                    return Err("`v = sum32(v)`".into());
                }
                // The target sums over lanes, then goes back.
                let summed = self.lane_array(target, Ty::F32)?;
                let (v, m) = (self.slot_addr(summed), self.slot_addr(mask));
                self.call("sum32", &[v, m, self.n, self.lanes]);
                let name = format!("sum32__{}", summed.as_u32());
                self.bind(&name, Kind::Lanes(Ty::F32, summed));
                let value = Expr::Var(name);
                self.lanes(Some(mask), |s, t| s.store(target, None, &value, t))
            }
            Stmt::Assign { target, op, value } => {
                let (value, target) = (self.hoist(value)?, self.hoist(target)?);
                self.lanes(Some(mask), |s, t| s.store(&target, *op, &value, t))
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                body,
                ..
            } => {
                let counter = self.lane_array(from, Ty::U32)?;
                let to = self.lane_array(to, Ty::U32)?;
                let step = self.lane_array(step, Ty::U32)?;
                let live = self.lane_slot(Ty::Bool);
                let (header, run, exit) = (
                    self.b.create_block(),
                    self.b.create_block(),
                    self.b.create_block(),
                );
                self.b.ins().jump(header, &[]);
                self.b.switch_to_block(header);
                let any = self.b.declare_var(types::I8);
                let off = self.iconst(types::I8, 0);
                self.b.def_var(any, off);
                self.lanes(None, |s, t| {
                    let on = s.live(mask, t);
                    let Val::S(i) = s.load_lane(counter, Ty::U32, t) else {
                        unreachable!()
                    };
                    let Val::S(end) = s.load_lane(to, Ty::U32, t) else {
                        unreachable!()
                    };
                    let more = s.b.ins().icmp(IntCC::UnsignedLessThan, i, end);
                    let on = s.b.ins().band(on, more);
                    s.set_mask(live, t, on);
                    let a = s.b.use_var(any);
                    let a = s.b.ins().bor(a, on);
                    s.b.def_var(any, a);
                    Ok(())
                })?;
                let a = self.b.use_var(any);
                self.b.ins().brif(a, run, &[], exit, &[]);
                self.b.switch_to_block(run);
                self.scope.push(HashMap::new());
                self.bind(var, Kind::Lanes(Ty::U32, counter));
                self.block(body, live)?;
                self.scope.pop();
                self.lanes(Some(live), |s, t| {
                    let (Val::S(i), Val::S(by)) = (
                        s.load_lane(counter, Ty::U32, t),
                        s.load_lane(step, Ty::U32, t),
                    ) else {
                        unreachable!()
                    };
                    let next = s.b.ins().iadd(i, by);
                    s.store_lane(counter, Ty::U32, t, Val::S(next));
                    Ok(())
                })?;
                self.b.ins().jump(header, &[]);
                self.b.switch_to_block(exit);
                Ok(())
            }
            Stmt::If {
                cond: Expr::Bool(taken),
                then,
                els,
            } => {
                let branch = if *taken { then } else { els };
                self.block(branch, mask)
            }
            Stmt::If { cond, then, els } => {
                let (yes, no, any_yes, any_no) = self.split(cond)?;
                self.branch(any_yes, then, yes)?;
                if !els.is_empty() {
                    self.branch(any_no, els, no)?;
                }
                Ok(())
            }
            Stmt::Return => self.leave(),
            // Lockstep leaves every thread at the barrier already.
            Stmt::Barrier => Ok(()),
            Stmt::Expr(e) => {
                let e = self.hoist(e)?;
                self.lanes(Some(mask), |s, t| s.expr(&e, t).map(|_| ()))
            }
            Stmt::TileAcc { name, rows, cols }
            | Stmt::TileHeld {
                name,
                rows,
                depth: cols,
            } => {
                let (rows, cols) = (
                    self.iconst(self.ptr, *rows as i64),
                    self.iconst(self.ptr, *cols as i64),
                );
                let tiles = self
                    .call("tiles_new", &[rows, cols, self.n, self.lanes])
                    .expect("a tile pointer");
                self.tiles.push(tiles);
                self.bind(name, Kind::Tiles(tiles));
                Ok(())
            }
            Stmt::TileZero { acc } => {
                let tiles = self.tiles_of(acc)?;
                self.call("tiles_zero", &[tiles]);
                Ok(())
            }
            Stmt::TileScale { acc, by } => {
                let tiles = self.tiles_of(acc)?;
                let by = self.hoist(by)?;
                self.each_sum(tiles, |s, e| {
                    s.lanes(Some(mask), |s, t| {
                        let r = s.call("tiles_row", &[tiles, t, e]).expect("a row");
                        s.scope.push(HashMap::new());
                        s.bind("r", Kind::One(Ty::U32, r));
                        let Val::S(by) = s.expr_as(&by, Ty::F32, t)? else {
                            return Err("a four-wide scale".into());
                        };
                        s.scope.pop();
                        s.call("tiles_scale", &[tiles, t, e, by]);
                        Ok(())
                    })
                })
            }
            Stmt::TileHold {
                held,
                a,
                a_at,
                stride,
            } => {
                let held = self.tiles_of(held)?;
                let (a, a_len) = self.shared_of(a)?;
                let a_at = self.lane_array(a_at, Ty::U32)?;
                let stride = self.lane_array(stride, Ty::U32)?;
                let args = [
                    held,
                    a,
                    a_len,
                    self.slot_addr(a_at),
                    self.slot_addr(stride),
                    self.slot_addr(mask),
                    self.n,
                ];
                self.call("tiles_hold", &args);
                Ok(())
            }
            Stmt::TileBlock {
                acc,
                a,
                a_at,
                b,
                b_at,
                stride,
                depth,
                parts,
                apart,
            } => {
                let tiles = self.tiles_of(acc)?;
                let (b, b_len) = self.shared_of(b)?;
                let b_at = self.lane_array(b_at, Ty::U32)?;
                let stride = self.lane_array(stride, Ty::U32)?;
                let depth = self.iconst(self.ptr, *depth as i64);
                let (b_at, stride, mask) = (
                    self.slot_addr(b_at),
                    self.slot_addr(stride),
                    self.slot_addr(mask),
                );
                // A held slab is a tile object.
                if let Expr::Var(name) = a
                    && let Ok(Kind::Tiles(held)) = self.kind(name)
                {
                    let args = [
                        tiles,
                        held,
                        b,
                        b_len,
                        b_at,
                        stride,
                        depth,
                        mask,
                        self.n,
                        self.precision,
                    ];
                    self.call("tiles_block_held", &args);
                    return Ok(());
                }
                let (a, a_len) = self.shared_of(a)?;
                let a_at = self.lane_array(a_at, Ty::U32)?;
                let apart = self.lane_array(apart, Ty::U32)?;
                let parts = self.iconst(self.ptr, *parts as i64);
                let args = [
                    tiles,
                    a,
                    a_len,
                    self.slot_addr(a_at),
                    b,
                    b_len,
                    b_at,
                    stride,
                    depth,
                    parts,
                    self.slot_addr(apart),
                    mask,
                    self.n,
                    self.precision,
                ];
                self.call("tiles_block", &args);
                Ok(())
            }
            Stmt::TileEach { acc, body } => {
                let tiles = self.tiles_of(acc)?;
                let live = self.lane_slot(Ty::Bool);
                let (r, c, v) = (
                    self.lane_slot(Ty::U32),
                    self.lane_slot(Ty::U32),
                    self.lane_slot(Ty::F32),
                );
                self.each_sum(tiles, |s, e| {
                    s.lanes(None, |s, t| {
                        let on = s.live(mask, t);
                        let (ra, ca, va) = (s.at(r, t, 4), s.at(c, t, 4), s.at(v, t, 4));
                        let on = s
                            .call("tiles_each", &[tiles, t, e, on, ra, ca, va])
                            .expect("whether live");
                        s.set_mask(live, t, on);
                        Ok(())
                    })?;
                    s.scope.push(HashMap::new());
                    s.bind("r", Kind::Lanes(Ty::U32, r));
                    s.bind("c", Kind::Lanes(Ty::U32, c));
                    s.bind("v", Kind::Lanes(Ty::F32, v));
                    s.block(body, live)?;
                    s.scope.pop();
                    Ok(())
                })
            }
        }
    }

    fn tiles_of(&mut self, name: &str) -> Res<Value> {
        match self.kind(name)? {
            Kind::Tiles(t) => Ok(t),
            _ => Err(format!("`{name}` is not a tile block")),
        }
    }

    /// Group memory `e` names: its address and length.
    fn shared_of(&mut self, e: &Expr) -> Res<(Value, Value)> {
        let Expr::Var(name) = e else {
            return Err("tiles multiply group memory by name".into());
        };
        let Kind::Shared(_, slot, len) = self.kind(name)? else {
            return Err(format!("`{name}` is not group memory"));
        };
        let addr = self.slot_addr(slot);
        Ok((addr, self.iconst(self.ptr, len as i64)))
    }

    /// `f` once per sum index `e` of a lane.
    fn each_sum<F>(&mut self, tiles: Value, f: F) -> Res
    where
        F: FnOnce(&mut Self, Value) -> Res,
    {
        let per = self.call("tiles_per", &[tiles]).expect("sums per lane");
        let (header, body, exit) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        let e = self.b.declare_var(self.ptr);
        let zero = self.iconst(self.ptr, 0);
        self.b.def_var(e, zero);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let ev = self.b.use_var(e);
        let more = self.b.ins().icmp(IntCC::UnsignedLessThan, ev, per);
        self.b.ins().brif(more, body, &[], exit, &[]);
        self.b.switch_to_block(body);
        f(self, ev)?;
        let one = self.iconst(self.ptr, 1);
        let next = self.b.ins().iadd(ev, one);
        self.b.def_var(e, next);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(exit);
        Ok(())
    }
}

/// Fills `func` with `body` over groups `lo..hi` of a context.
pub(crate) fn function(
    module: &mut JITModule,
    func: &mut Function,
    fctx: &mut FunctionBuilderContext,
    body: &Body,
) -> Res {
    let frontend = module.target_config();
    let ptr = frontend.pointer_type();
    let mut b = FunctionBuilder::new(func, fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    let (ctx, lo, hi) = {
        let p = b.block_params(entry);
        (p[0], p[1], p[2])
    };
    let field = |b: &mut FunctionBuilder, off: usize| b.ins().load(ptr, trusted(), ctx, off as i32);
    let bufs = field(&mut b, std::mem::offset_of!(Ctx, bufs));
    let args = field(&mut b, std::mem::offset_of!(Ctx, args));
    let n = field(&mut b, std::mem::offset_of!(Ctx, threads));
    let lanes = field(&mut b, std::mem::offset_of!(Ctx, lanes));
    let precision = field(&mut b, std::mem::offset_of!(Ctx, precision));
    let first = b.declare_var(ptr);
    let zero = b.ins().iconst(ptr, 0);
    b.def_var(first, zero);
    let mut l = Lowerer {
        b,
        module,
        frontend,
        ptr,
        helpers: HashMap::new(),
        scope: vec![HashMap::new()],
        masks: Vec::new(),
        n,
        lanes,
        first,
        precision,
        tiles: Vec::new(),
    };
    for (i, param) in body.params.iter().enumerate() {
        let off = (i * std::mem::size_of::<Arg>()) as i32;
        match param {
            Param::Buffer { name, form, .. } => {
                let index = l.b.ins().load(ptr, trusted(), args, off);
                let skip = l.b.ins().load(
                    ptr,
                    trusted(),
                    args,
                    off + std::mem::offset_of!(Arg, skip) as i32,
                );
                let size = l.iconst(ptr, std::mem::size_of::<Buf>() as i64);
                let at = l.b.ins().imul(index, size);
                let buf = l.b.ins().iadd(bufs, at);
                let bptr = l.b.ins().load(ptr, trusted(), buf, 0);
                let len =
                    l.b.ins()
                        .load(ptr, trusted(), buf, std::mem::offset_of!(Buf, len) as i32);
                l.bind(
                    name,
                    Kind::Buffer {
                        form: *form,
                        ptr: bptr,
                        len,
                        skip,
                    },
                );
            }
            Param::Scalar { name, ty } => {
                let bits_off = off + std::mem::offset_of!(Arg, bits) as i32;
                let bits = l.b.ins().load(types::I32, trusted(), args, bits_off);
                let v = match ty {
                    Ty::U32 => bits,
                    Ty::F32 => l.b.ins().bitcast(types::F32, MemFlagsData::new(), bits),
                    other => return Err(format!("a {other:?} scalar")),
                };
                l.bind(name, Kind::One(*ty, v));
            }
            other => return Err(format!("{other:?} as an entry parameter")),
        }
    }
    // Groups `lo..hi`, each a lockstep run of the body.
    let (header, group, exit) = (l.b.create_block(), l.b.create_block(), l.b.create_block());
    let g = l.b.declare_var(ptr);
    l.b.def_var(g, lo);
    l.b.ins().jump(header, &[]);
    l.b.switch_to_block(header);
    let gv = l.b.use_var(g);
    let more = l.b.ins().icmp(IntCC::UnsignedLessThan, gv, hi);
    l.b.ins().brif(more, group, &[], exit, &[]);
    l.b.switch_to_block(group);
    let start = l.b.ins().imul(gv, n);
    l.b.def_var(first, start);
    let m0 = l.lane_slot(Ty::Bool);
    l.lanes(None, |s, t| {
        let on = s.iconst(types::I8, 1);
        s.set_mask(m0, t, on);
        Ok(())
    })?;
    l.block(&body.stmts, m0)?;
    for tiles in std::mem::take(&mut l.tiles) {
        l.call("tiles_free", &[tiles]);
    }
    let one = l.iconst(ptr, 1);
    let next = l.b.ins().iadd(gv, one);
    l.b.def_var(g, next);
    l.b.ins().jump(header, &[]);
    l.b.switch_to_block(exit);
    l.b.ins().return_(&[]);
    l.b.seal_all_blocks();
    l.b.finalize(frontend);
    Ok(())
}
