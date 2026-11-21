//! Runs a body on the host, each group in lockstep.

use crate::lockstep::{Slabs, bf16_pair, leaders, round_e4m3, round_f16};
use std::collections::HashMap;
use std::sync::LazyLock;
use tessera_codegen::form::Load;
use tessera_codegen::ir::{BinOp, Body, Expr, Helper, Intrinsic, Param, Stmt, Ty, gather};
use tessera_codegen::numerics::Precision;
use tessera_graph::{E2M1, E4M3, F16};

/// One argument of a launch, on the host side.
#[derive(Clone, Copy, Debug)]
pub enum ArgVal {
    /// Buffer `at` in the launch's list, `skip` elements in.
    Buf {
        at: usize,
        skip: usize,
        form: &'static dyn Load,
    },
    U32(u32),
    F32(f32),
}

impl ArgVal {
    /// A buffer is a pointer to its first element.
    fn val(self) -> Val {
        match self {
            ArgVal::Buf { at, skip, form } => Val::P(at, form.offset(skip)),
            ArgVal::U32(v) => Val::U(v),
            ArgVal::F32(v) => Val::F(v),
        }
    }
}

/// No arguments, for a helper's own scope.
static NONE: LazyLock<HashMap<String, ArgVal>> = LazyLock::new(HashMap::new);

/// The launch's layout of thread groups.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub groups: usize,
    pub threads: usize,
}

/// A value in a thread's register.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Val {
    F(f32),
    U(u32),
    L(u64),
    B(bool),
    V([f32; 4]),
    /// A byte offset into a buffer.
    P(usize, usize),
}

impl Val {
    fn f(self) -> f32 {
        match self {
            Val::F(x) => x,
            Val::U(x) => x as f32,
            Val::L(x) => x as f32,
            Val::B(b) => b as u32 as f32,
            Val::V(_) | Val::P(..) => panic!("{self:?} where a scalar was due"),
        }
    }

    /// A float's bits, or a word as stored.
    fn bits(self) -> u32 {
        match self {
            Val::F(x) => x.to_bits(),
            other => other.u(),
        }
    }

    fn u(self) -> u32 {
        match self {
            Val::U(x) => x,
            Val::L(x) => x as u32,
            Val::F(x) => x as u32,
            Val::B(b) => b as u32,
            Val::V(_) | Val::P(..) => panic!("{self:?} where an index was due"),
        }
    }

    fn l(self) -> u64 {
        match self {
            Val::L(x) => x,
            Val::U(x) => u64::from(x),
            Val::F(x) => x as u64,
            Val::B(b) => b as u64,
            Val::V(_) | Val::P(..) => panic!("{self:?} where an index was due"),
        }
    }

    fn b(self) -> bool {
        match self {
            Val::B(b) => b,
            Val::U(x) => x != 0,
            Val::L(x) => x != 0,
            Val::F(x) => x != 0.0,
            Val::V(_) | Val::P(..) => panic!("{self:?} where a condition was due"),
        }
    }

    fn v(self) -> [f32; 4] {
        match self {
            Val::V(v) => v,
            other => panic!("{other:?} where a float4 was due"),
        }
    }
}

/// A buffer argument: bytes, with an element offset.
struct Arg {
    buffer: usize,
    skip: usize,
    form: &'static dyn Load,
}

/// One group's state while it runs.
struct Group<'a, 'b> {
    threads: usize,
    lanes: usize,
    precision: Precision,
    first: usize,
    /// Names in scope, innermost last, each a value per thread.
    vars: Vec<(String, Vec<Val>)>,
    /// Local arrays, by name, then thread, then element.
    arrays: HashMap<String, Vec<Vec<Val>>>,
    shared: HashMap<String, Vec<Val>>,
    /// Tile sums by name: rows, columns, each SIMD group's block.
    tiles: HashMap<String, (usize, usize, Vec<Vec<f32>>)>,
    /// Threads that returned.
    done: Vec<bool>,
    args: &'b HashMap<String, ArgVal>,
    buffers: &'b mut [&'a mut [u8]],
    /// The helpers a call may name.
    helpers: &'b [Helper],
}

impl Group<'_, '_> {
    fn lookup(&self, name: &str) -> &[Val] {
        match self.vars.iter().rev().find(|(n, _)| n == name) {
            Some((_, vals)) => vals,
            None => panic!("`{name}` is not in scope"),
        }
    }

    fn lookup_mut(&mut self, name: &str) -> &mut Vec<Val> {
        match self.vars.iter_mut().rev().find(|(n, _)| n == name) {
            Some((_, vals)) => vals,
            None => panic!("`{name}` is not in scope"),
        }
    }

    fn arg(&self, name: &str) -> Arg {
        match self.args.get(name) {
            Some(ArgVal::Buf { at, skip, form }) => Arg {
                buffer: *at,
                skip: *skip,
                form: *form,
            },
            _ => panic!("`{name}` is not a buffer"),
        }
    }

    /// Element `index` of a buffer, widened to the register type.
    fn load(&self, arg: &Arg, index: u64) -> Val {
        let bytes = &self.buffers[arg.buffer];
        let at = arg.skip + index as usize;
        if arg.form.bytes(at + 1) > bytes.len() {
            panic!("a read {index} elements past buffer {}", arg.buffer);
        }
        match arg.form.dtype().is_float() {
            true => Val::F(arg.form.unpack(bytes, at)),
            false => Val::U(arg.form.unpack(bytes, at) as u32),
        }
    }

    /// Writes store plain f32 or ids, never packed.
    fn store(&mut self, arg: &Arg, index: u64, value: Val) {
        let at = (arg.skip + index as usize) * 4;
        let bytes = &mut self.buffers[arg.buffer];
        let Some(raw) = bytes.get_mut(at..at + 4) else {
            panic!("a write {index} elements past buffer {}", arg.buffer);
        };
        match arg.form.dtype().is_float() {
            true if arg.form.narrow() => panic!("kernels write only f32"),
            true => raw.copy_from_slice(&value.f().to_le_bytes()),
            false => raw.copy_from_slice(&value.u().to_le_bytes()),
        }
    }

    /// Four consecutive elements as floats.
    fn load4(&self, arg: &Arg, index: u64) -> Val {
        let mut out = [0.0f32; 4];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.load(arg, index + i as u64).f();
        }
        Val::V(out)
    }

    /// `e` for every thread. `mask` says which values matter.
    fn eval(&mut self, e: &Expr, mask: &[bool]) -> Vec<Val> {
        let n = self.threads;
        match e {
            Expr::Int(x) => vec![Val::U(*x as u32); n],
            Expr::Float(x) => vec![Val::F(*x as f32); n],
            Expr::Bool(b) => vec![Val::B(*b); n],
            Expr::Var(name) => match self.args.get(name) {
                Some(arg) => vec![arg.val(); n],
                None => self.lookup(name).to_vec(),
            },
            Expr::Neg(a) => {
                let mut v = self.eval(a, mask);
                for x in &mut v {
                    *x = match *x {
                        Val::F(f) => Val::F(-f),
                        Val::U(u) => Val::U(u.wrapping_neg()),
                        Val::L(l) => Val::L(l.wrapping_neg()),
                        other => other,
                    };
                }
                v
            }
            Expr::Not(a) => {
                let mut v = self.eval(a, mask);
                for x in &mut v {
                    *x = match *x {
                        Val::U(u) => Val::U(!u),
                        Val::L(l) => Val::L(!l),
                        other => Val::B(!other.b()),
                    };
                }
                v
            }
            Expr::Cast(ty, a) => {
                let mut v = self.eval(a, mask);
                for x in &mut v {
                    *x = match ty {
                        Ty::F32 => Val::F(x.f()),
                        Ty::U32 => Val::U(x.u()),
                        Ty::U64 => Val::L(x.l()),
                        Ty::Bool => Val::B(x.b()),
                        other => panic!("no cast to {other:?}"),
                    };
                }
                v
            }
            // The right side runs only where it decides the result.
            Expr::Bin(BinOp::And, a, b) => self.branch(a, b, &Expr::Bool(false), mask),
            Expr::Bin(BinOp::Or, a, b) => self.branch(a, &Expr::Bool(true), b, mask),
            Expr::Bin(op, a, b) => {
                let (va, vb) = (self.eval(a, mask), self.eval(b, mask));
                va.into_iter()
                    .zip(vb)
                    .zip(mask)
                    .map(|((x, y), on)| if *on { binary(*op, x, y) } else { x })
                    .collect()
            }
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    panic!("an index names what it reads");
                };
                let at = self.eval(at, mask);
                if self.args.contains_key(name) {
                    let arg = self.arg(name);
                    (0..n)
                        .map(|t| {
                            if mask[t] {
                                self.load(&arg, at[t].l())
                            } else {
                                Val::F(0.0)
                            }
                        })
                        .collect()
                } else if let Some(shared) = self.shared.get(name) {
                    (0..n)
                        .map(|t| {
                            if mask[t] {
                                shared[at[t].u() as usize]
                            } else {
                                Val::F(0.0)
                            }
                        })
                        .collect()
                } else if let Some(arrays) = self.arrays.get(name) {
                    (0..n)
                        .map(|t| {
                            if mask[t] {
                                arrays[t][at[t].u() as usize]
                            } else {
                                Val::F(0.0)
                            }
                        })
                        .collect()
                } else {
                    let base = self.lookup(name).to_vec();
                    self.bytes(&base, &at, mask, 1)
                }
            }
            Expr::Field(base, h) => {
                let v = self.eval(base, mask);
                let i = match h {
                    'x' => 0,
                    'y' => 1,
                    'z' => 2,
                    _ => 3,
                };
                v.into_iter().map(|x| Val::F(x.v()[i])).collect()
            }
            Expr::Select(c, a, b) => self.branch(c, a, b, mask),
            Expr::Call(i, args) => self.call(*i, args, mask),
            Expr::Fn(name, _, args) => {
                let helpers = self.helpers;
                let Some(h) = helpers.iter().find(|h| h.name == *name) else {
                    panic!("`{name}` is not a helper of this body");
                };
                let args = args.iter().map(|a| self.eval(a, mask)).collect();
                self.enter(h, args, mask)
            }
        }
    }

    /// `if c { a } else { b }`, each side only where taken.
    fn branch(&mut self, c: &Expr, a: &Expr, b: &Expr, mask: &[bool]) -> Vec<Val> {
        let c = self.eval(c, mask);
        let n = mask.len();
        let yes: Vec<bool> = (0..n).map(|t| mask[t] && c[t].b()).collect();
        let no: Vec<bool> = (0..n).map(|t| mask[t] && !c[t].b()).collect();
        let (va, vb) = (self.eval(a, &yes), self.eval(b, &no));
        c.into_iter()
            .zip(va.into_iter().zip(vb))
            .map(|(c, (x, y))| if c.b() { x } else { y })
            .collect()
    }

    fn call(&mut self, i: Intrinsic, args: &[Expr], mask: &[bool]) -> Vec<Val> {
        host(i)(self, args, mask)
    }

    fn unary(&mut self, args: &[Expr], mask: &[bool], f: fn(f32) -> f32) -> Vec<Val> {
        let v = self.eval(&args[0], mask);
        v.into_iter().map(|x| Val::F(f(x.f()))).collect()
    }

    fn pair(&mut self, args: &[Expr], mask: &[bool], f: fn(f32, f32) -> f32) -> Vec<Val> {
        let (a, b) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
        a.into_iter()
            .zip(b)
            .map(|(x, y)| Val::F(f(x.f(), y.f())))
            .collect()
    }

    /// The `bits` wide field at `args`, widened with `f`.
    fn narrow(&mut self, args: &[Expr], mask: &[bool], bits: u32, f: fn(u32) -> f32) -> Vec<Val> {
        let (w, l) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
        (0..self.threads)
            .map(|t| Val::F(f(field(w[t].u(), l[t].u(), bits))))
            .collect()
    }

    /// `f` of each bit field and its width.
    fn fields(&mut self, args: &[Expr], mask: &[bool], f: fn(u32, u32) -> Val) -> Vec<Val> {
        let (w, l, n) = (
            self.eval(&args[0], mask),
            self.eval(&args[1], mask),
            self.eval(&args[2], mask),
        );
        (0..n.len())
            .map(|t| f(field(w[t].u(), l[t].u(), n[t].u()), n[t].u()))
            .collect()
    }

    /// The `width` bytes at a byte offset, as a u32.
    fn read(&mut self, args: &[Expr], mask: &[bool], width: usize) -> Vec<Val> {
        let (base, at) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
        self.bytes(&base, &at, mask, width)
    }

    /// The `width` bytes `at` past each pointer, as a u32.
    fn bytes(&self, base: &[Val], at: &[Val], mask: &[bool], width: usize) -> Vec<Val> {
        let mut out = Vec::with_capacity(base.len());
        for t in 0..base.len() {
            if !mask[t] {
                out.push(Val::U(0));
                continue;
            }
            let Val::P(buffer, from) = base[t] else {
                panic!("{:?} where bytes were due", base[t]);
            };
            let from = from + at[t].l() as usize;
            let Some(raw) = self.buffers[buffer].get(from..from + width) else {
                panic!("a {width}-byte read at byte {from}, past buffer {buffer}");
            };
            let mut word = [0u8; 4];
            word[..width].copy_from_slice(raw);
            out.push(Val::U(u32::from_le_bytes(word)));
        }
        out
    }

    /// Helper `h` with `args` for its parameters, under `mask`.
    fn enter(&mut self, h: &Helper, args: Vec<Vec<Val>>, mask: &[bool]) -> Vec<Val> {
        // A helper sees its own names only.
        let named = std::mem::replace(&mut self.args, &NONE);
        let arrays = std::mem::take(&mut self.arrays);
        let shared = std::mem::take(&mut self.shared);
        let depth = self.vars.len();
        for (param, vals) in h.body.params.iter().zip(args) {
            self.vars.push((param.name().to_string(), vals));
        }
        // Its locals stay in scope for the value.
        for s in &h.body.stmts {
            self.stmt(s, mask);
        }
        let ret = h.body.ret.as_ref().expect("a helper has a value");
        let value = self.eval(ret, mask);
        self.vars.truncate(depth);
        (self.args, self.arrays, self.shared) = (named, arrays, shared);
        value
    }

    /// Every thread's statements, under `mask`.
    fn block(&mut self, stmts: &[Stmt], mask: &[bool]) {
        let depth = self.vars.len();
        for s in stmts {
            let live: Vec<bool> = mask
                .iter()
                .zip(&self.done)
                .map(|(m, d)| *m && !*d)
                .collect();
            if !live.iter().any(|m| *m) {
                break;
            }
            self.stmt(s, &live);
        }
        self.vars.truncate(depth);
    }

    fn assign(&mut self, target: &Expr, value: Vec<Val>, mask: &[bool]) {
        match target {
            Expr::Var(name) => {
                let vals = self.lookup_mut(name);
                for (t, v) in value.into_iter().enumerate() {
                    if mask[t] {
                        vals[t] = v;
                    }
                }
            }
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    panic!("an index names what it writes");
                };
                let at = self.eval(at, mask);
                let n = self.threads;
                if self.args.contains_key(name) {
                    let arg = self.arg(name);
                    for t in (0..n).filter(|t| mask[*t]) {
                        self.store(&arg, at[t].l(), value[t]);
                    }
                } else if self.shared.contains_key(name) {
                    let shared = self.shared.get_mut(name).expect("checked");
                    for t in (0..n).filter(|t| mask[*t]) {
                        shared[at[t].u() as usize] = value[t];
                    }
                } else {
                    let arrays = self.arrays.get_mut(name).expect("a local array");
                    for t in (0..n).filter(|t| mask[*t]) {
                        arrays[t][at[t].u() as usize] = value[t];
                    }
                }
            }
            Expr::Field(base, h) => {
                let Expr::Var(name) = &**base else {
                    panic!("a field names its value");
                };
                let i = match h {
                    'x' => 0,
                    'y' => 1,
                    'z' => 2,
                    _ => 3,
                };
                let vals = self.lookup_mut(name);
                for (t, v) in value.into_iter().enumerate() {
                    if mask[t] {
                        let mut four = vals[t].v();
                        four[i] = v.f();
                        vals[t] = Val::V(four);
                    }
                }
            }
            other => panic!("cannot assign to {other:?}"),
        }
    }

    /// A tile block's rows, columns and this lane's share.
    fn tile_shape(&self, acc: &str) -> (usize, usize, usize) {
        let (rows, cols, _) = &self.tiles[acc];
        (*rows, *cols, (rows * cols).div_ceil(self.lanes))
    }

    fn stmt(&mut self, s: &Stmt, mask: &[bool]) {
        let n = self.threads;
        match s {
            Stmt::Let { name, init } => {
                let v = self.eval(init, mask);
                self.vars.push((name.clone(), v));
            }
            Stmt::Array { name, ty, len } => {
                let zero = if *ty == Ty::F32 {
                    Val::F(0.0)
                } else {
                    Val::U(0)
                };
                self.arrays.insert(name.clone(), vec![vec![zero; *len]; n]);
            }
            Stmt::Shared { name, ty, len } => {
                let zero = if *ty == Ty::F32 {
                    Val::F(0.0)
                } else {
                    Val::U(0)
                };
                self.shared.insert(name.clone(), vec![zero; *len]);
            }
            Stmt::Assign {
                target,
                op,
                value: Expr::Call(Intrinsic::Sum32, args),
            } => {
                assert!(op.is_none(), "`v = sum32(v)`");
                let v = self.call(Intrinsic::Sum32, args, mask);
                self.assign(target, v, mask);
            }
            Stmt::Assign { target, op, value } => {
                let mut v = self.eval(value, mask);
                if let Some(op) = op {
                    let old = self.eval(target, mask);
                    for t in 0..n {
                        if mask[t] {
                            v[t] = binary(*op, old[t], v[t]);
                        }
                    }
                }
                self.assign(target, v, mask);
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                body,
                ..
            } => {
                let (from, to, step) = (
                    self.eval(from, mask),
                    self.eval(to, mask),
                    self.eval(step, mask),
                );
                self.vars.push((var.clone(), from));
                loop {
                    let at = self.lookup(var).to_vec();
                    let live: Vec<bool> = (0..n)
                        .map(|t| mask[t] && !self.done[t] && at[t].u() < to[t].u())
                        .collect();
                    if !live.iter().any(|m| *m) {
                        break;
                    }
                    self.block(body, &live);
                    let vals = self.lookup_mut(var);
                    for t in (0..n).filter(|t| live[*t]) {
                        vals[t] = Val::U(vals[t].u().wrapping_add(step[t].u()));
                    }
                }
                self.vars.pop();
            }
            Stmt::If { cond, then, els } => {
                let c = self.eval(cond, mask);
                let yes: Vec<bool> = (0..n).map(|t| mask[t] && c[t].b()).collect();
                let no: Vec<bool> = (0..n).map(|t| mask[t] && !c[t].b()).collect();
                self.block(then, &yes);
                self.block(els, &no);
            }
            Stmt::Return => {
                for t in (0..n).filter(|t| mask[*t]) {
                    self.done[t] = true;
                }
            }
            // Lockstep leaves every thread at the barrier already.
            Stmt::Barrier => {}
            Stmt::Expr(e) => {
                self.eval(e, mask);
            }
            Stmt::TileAcc { name, rows, cols } => {
                let simds = n.div_ceil(self.lanes);
                let blocks = vec![vec![0.0f32; rows * cols]; simds];
                self.tiles.insert(name.clone(), (*rows, *cols, blocks));
            }
            Stmt::TileHeld { name, rows, depth } => {
                let simds = n.div_ceil(self.lanes);
                let slabs = vec![vec![0.0f32; rows * depth]; simds];
                self.tiles.insert(name.clone(), (*rows, *depth, slabs));
            }
            Stmt::TileHold {
                held,
                a,
                a_at,
                stride,
            } => {
                let Expr::Var(a) = a else {
                    panic!("a slab holds group memory by name");
                };
                let (a_at, stride) = (self.eval(a_at, mask), self.eval(stride, mask));
                let (rows, depth, _) = self.tiles[held];
                for (s, t) in leaders(self.lanes, n, mask) {
                    let (at, stride) = (a_at[t].u() as usize, stride[t].u() as usize);
                    let from: Vec<f32> = (0..rows * depth)
                        .map(|i| self.shared[a][at + i / depth * stride + i % depth].f())
                        .collect();
                    self.tiles.get_mut(held).expect("held slab").2[s] = from;
                }
            }
            Stmt::TileZero { acc } => {
                for block in &mut self.tiles.get_mut(acc).expect("tile sums").2 {
                    block.fill(0.0);
                }
            }
            Stmt::TileScale { acc, by } => {
                let (rows, cols, per) = self.tile_shape(acc);
                for e in 0..per {
                    let r: Vec<Val> = (0..n)
                        .map(|t| Val::U(((t % self.lanes + e * self.lanes) / cols) as u32))
                        .collect();
                    self.vars.push(("r".into(), r));
                    let by = self.eval(by, mask);
                    self.vars.pop();
                    for t in (0..n).filter(|t| mask[*t]) {
                        let at = t % self.lanes + e * self.lanes;
                        if at < rows * cols {
                            let block =
                                &mut self.tiles.get_mut(acc).expect("tile sums").2[t / self.lanes];
                            block[at] *= by[t].f();
                        }
                    }
                }
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
                let (rows, cols, _) = self.tile_shape(acc);
                let (Expr::Var(a), Expr::Var(b)) = (a, b) else {
                    panic!("tiles multiply group memory by name");
                };
                let (a_at, b_at, stride, apart) = (
                    self.eval(a_at, mask),
                    self.eval(b_at, mask),
                    self.eval(stride, mask),
                    self.eval(apart, mask),
                );
                for (s, t) in leaders(self.lanes, n, mask) {
                    let at = Slabs {
                        a: a_at[t].u() as usize,
                        b: b_at[t].u() as usize,
                        stride: stride[t].u() as usize,
                        parts: *parts,
                        apart: apart[t].u() as usize,
                    };
                    let mut block =
                        std::mem::take(&mut self.tiles.get_mut(acc).expect("tile sums").2[s]);
                    let sb = &self.shared[b];
                    let rb = |i: usize| sb[i].bits();
                    match self.tiles.get(a) {
                        // A held slab: row `i / stride`, column `i % stride`.
                        Some((_, held, slabs)) => {
                            let (slab, at) = (&slabs[s], Slabs { a: 0, ..at });
                            let ra =
                                |i: usize| slab[i / at.stride * held + i % at.stride].to_bits();
                            at.products(self.precision, rows, cols, *depth, ra, rb, &mut block);
                        }
                        None => {
                            let sa = &self.shared[a];
                            let ra = |i: usize| sa[i].bits();
                            at.products(self.precision, rows, cols, *depth, ra, rb, &mut block);
                        }
                    }
                    self.tiles.get_mut(acc).expect("tile sums").2[s] = block;
                }
            }
            Stmt::TileEach { acc, body } => {
                let (rows, cols, per) = self.tile_shape(acc);
                for e in 0..per {
                    let mut r = Vec::with_capacity(n);
                    let mut c = Vec::with_capacity(n);
                    let mut v = Vec::with_capacity(n);
                    let mut live = Vec::with_capacity(n);
                    for (t, m) in mask.iter().enumerate() {
                        let at = t % self.lanes + e * self.lanes;
                        let on = *m && at < rows * cols;
                        live.push(on);
                        r.push(Val::U((at / cols) as u32));
                        c.push(Val::U((at % cols) as u32));
                        let value = match on {
                            true => self.tiles[acc].2[t / self.lanes][at],
                            false => 0.0,
                        };
                        v.push(Val::F(value));
                    }
                    self.vars.push(("r".into(), r));
                    self.vars.push(("c".into(), c));
                    self.vars.push(("v".into(), v));
                    self.block(body, &live);
                    self.vars.truncate(self.vars.len() - 3);
                }
            }
        }
    }
}

type Host = for<'a, 'b> fn(&mut Group<'a, 'b>, &[Expr], &[bool]) -> Vec<Val>;

/// Runs each intrinsic per thread, from the table.
macro_rules! hosts {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:tt)*) $(-> $ret:ident)?
        { device: $dev:expr, text: $text:expr, wgsl: $wgsl:expr, host: $host:expr, count: $count:expr $(,)? })*) => {
        fn host(i: Intrinsic) -> Host {
            match i {
                $(Intrinsic::$var => $host,)*
            }
        }
    };
}

crate::intrinsic::table!(hosts);

/// An integer `op`. Sums, differences and products wrap.
macro_rules! integer {
    ($op:expr, $x:expr, $y:expr) => {{
        let (x, y) = ($x, $y);
        match $op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => x / y,
            BinOp::Rem => x % y,
            BinOp::Shl => x << y,
            BinOp::Shr => x >> y,
            BinOp::BitAnd => x & y,
            BinOp::BitOr => x | y,
            BinOp::BitXor => x ^ y,
            other => panic!("{other:?} on integers"),
        }
    }};
}

/// Integers wrap. Anything with a float is a float.
fn binary(op: BinOp, a: Val, b: Val) -> Val {
    if let Val::P(buffer, at) = a {
        assert_eq!(op, BinOp::Add, "a pointer only moves forward");
        return Val::P(buffer, at + b.l() as usize);
    }
    let float = matches!(a, Val::F(_)) || matches!(b, Val::F(_));
    let long = matches!(a, Val::L(_)) || matches!(b, Val::L(_));
    match op {
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
            let ordering = match (float, long) {
                (true, _) => a.f().partial_cmp(&b.f()),
                (false, true) => Some(a.l().cmp(&b.l())),
                (false, false) => Some(a.u().cmp(&b.u())),
            };
            let Some(ordering) = ordering else {
                return Val::B(op == BinOp::Ne);
            };
            Val::B(match op {
                BinOp::Lt => ordering.is_lt(),
                BinOp::Le => ordering.is_le(),
                BinOp::Gt => ordering.is_gt(),
                BinOp::Ge => ordering.is_ge(),
                BinOp::Eq => ordering.is_eq(),
                _ => ordering.is_ne(),
            })
        }
        BinOp::And => Val::B(a.b() && b.b()),
        BinOp::Or => Val::B(a.b() || b.b()),
        _ if float => {
            let (x, y) = (a.f(), b.f());
            Val::F(match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                other => panic!("{other:?} on floats"),
            })
        }
        _ if long => Val::L(integer!(op, a.l(), b.l())),
        _ => Val::U(integer!(op, a.u(), b.u())),
    }
}

/// Runs `body` over `grid`. `args` follow the parameters.
pub fn run(
    body: &Body,
    args: &[ArgVal],
    buffers: &mut [&mut [u8]],
    grid: Grid,
    lanes: usize,
    precision: Precision,
) {
    assert_eq!(args.len(), body.params.len(), "one argument per parameter");
    let mut named = HashMap::new();
    let mut helpers = Vec::new();
    for (param, arg) in body.params.iter().zip(args) {
        if let Param::Buffer { form, .. } = param {
            gather(&form.header(), &mut helpers);
        }
        named.insert(param.name().to_string(), *arg);
    }
    gather(&body.helpers, &mut helpers);
    for g in 0..grid.groups {
        let mut group = Group::new(grid.threads, lanes, precision, &named, buffers, &helpers);
        group.first = g * grid.threads;
        let mask = vec![true; grid.threads];
        group.block(&body.stmts, &mask);
    }
}

/// Helper `h` on one thread, `args` for its parameters.
pub fn call(h: &Helper, args: &[ArgVal], buffers: &mut [&mut [u8]]) -> Vec<f32> {
    assert_eq!(
        args.len(),
        h.body.params.len(),
        "one argument per parameter"
    );
    let mut helpers = Vec::new();
    gather(std::slice::from_ref(h), &mut helpers);
    let mut group = Group::new(1, 1, Precision::EXACT, &NONE, buffers, &helpers);
    let vals = args.iter().map(|a| vec![a.val()]).collect();
    match group.enter(h, vals, &[true])[0] {
        Val::V(four) => four.to_vec(),
        one => vec![one.f()],
    }
}

impl<'a, 'b> Group<'a, 'b> {
    fn new(
        threads: usize,
        lanes: usize,
        precision: Precision,
        args: &'b HashMap<String, ArgVal>,
        buffers: &'b mut [&'a mut [u8]],
        helpers: &'b [Helper],
    ) -> Self {
        Group {
            threads,
            lanes,
            precision,
            first: 0,
            vars: Vec::new(),
            arrays: HashMap::new(),
            shared: HashMap::new(),
            tiles: HashMap::new(),
            done: vec![false; threads],
            args,
            buffers,
            helpers,
        }
    }
}

/// The `n` bits of `w` from bit `low`.
fn field(w: u32, low: u32, n: u32) -> u32 {
    (w >> low) & ((1u64 << n) - 1) as u32
}

/// An `n`-bit field as a signed value.
fn sign_extend(field: u32, n: u32) -> i32 {
    let shift = 32 - n;
    ((field << shift) as i32) >> shift
}
