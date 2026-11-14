//! Runs a body on the host, each group in lockstep.

use crate::form::Form;
use crate::ir::{BinOp, Body, Expr, Intrinsic, Mini, Param, Stmt, Ty};
use crate::kept::Layout;
use crate::numerics::Precision;
use graph::{E2M1, E4M3, F16};
use std::collections::HashMap;

/// One argument of a launch, as the host sees it.
#[derive(Clone, Copy, Debug)]
pub enum ArgVal {
    /// Buffer `at` in the launch's list, `skip` elements in.
    Buf {
        at: usize,
        skip: usize,
        form: &'static dyn Form,
    },
    U32(u32),
    F32(f32),
}

/// Groups of threads, as the launch laid them out.
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
}

impl Val {
    fn f(self) -> f32 {
        match self {
            Val::F(x) => x,
            Val::U(x) => x as f32,
            Val::L(x) => x as f32,
            Val::B(b) => b as u32 as f32,
            Val::V(_) => panic!("a float4 where a scalar was due"),
        }
    }

    fn u(self) -> u32 {
        match self {
            Val::U(x) => x,
            Val::L(x) => x as u32,
            Val::F(x) => x as u32,
            Val::B(b) => b as u32,
            Val::V(_) => panic!("a float4 where an index was due"),
        }
    }

    fn l(self) -> u64 {
        match self {
            Val::L(x) => x,
            Val::U(x) => u64::from(x),
            Val::F(x) => x as u64,
            Val::B(b) => b as u64,
            Val::V(_) => panic!("a float4 where an index was due"),
        }
    }

    fn b(self) -> bool {
        match self {
            Val::B(b) => b,
            Val::U(x) => x != 0,
            Val::L(x) => x != 0,
            Val::F(x) => x != 0.0,
            Val::V(_) => panic!("a float4 where a condition was due"),
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
    form: &'static dyn Form,
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
    layout: Option<&'static dyn Layout>,
    args: &'b HashMap<String, ArgVal>,
    buffers: &'b mut [&'a mut [u8]],
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

    /// Writes store f32 or ids. No form packs them.
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
                Some(ArgVal::U32(v)) => vec![Val::U(*v); n],
                Some(ArgVal::F32(v)) => vec![Val::F(*v); n],
                _ => self.lookup(name).to_vec(),
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
                } else {
                    let arrays = &self.arrays[name];
                    (0..n)
                        .map(|t| {
                            if mask[t] {
                                arrays[t][at[t].u() as usize]
                            } else {
                                Val::F(0.0)
                            }
                        })
                        .collect()
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
            Expr::Select(c, a, b) => {
                // A branch runs where taken: its loads may be unsafe.
                let c = self.eval(c, mask);
                let yes: Vec<bool> = (0..n).map(|t| mask[t] && c[t].b()).collect();
                let no: Vec<bool> = (0..n).map(|t| mask[t] && !c[t].b()).collect();
                let (va, vb) = (self.eval(a, &yes), self.eval(b, &no));
                c.into_iter()
                    .zip(va.into_iter().zip(vb))
                    .map(|(c, (x, y))| if c.b() { x } else { y })
                    .collect()
            }
            Expr::Call(i, args) => self.call(*i, args, mask),
            Expr::Fn(name, ..) => panic!("`{name}` runs on the device: forms unpack on the host"),
        }
    }

    fn call(&mut self, i: Intrinsic, args: &[Expr], mask: &[bool]) -> Vec<Val> {
        let n = self.threads;
        let unary = |g: &mut Self, f: fn(f32) -> f32| {
            let v = g.eval(&args[0], mask);
            v.into_iter().map(|x| Val::F(f(x.f()))).collect()
        };
        match i {
            Intrinsic::Gid => (0..n).map(|t| Val::U((self.first + t) as u32)).collect(),
            Intrinsic::Lid => (0..n).map(|t| Val::U(t as u32)).collect(),
            Intrinsic::Lane => (0..n).map(|t| Val::U((t % self.lanes) as u32)).collect(),
            Intrinsic::ShflXor => {
                let (v, m) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
                (0..n)
                    .map(|t| {
                        let other = t ^ (m[t].u() as usize);
                        // Lanes shuffle within their own SIMD group.
                        if other / self.lanes == t / self.lanes && other < n {
                            v[other]
                        } else {
                            v[t]
                        }
                    })
                    .collect()
            }
            Intrinsic::Sum32 => {
                let v = self.eval(&args[0], mask);
                let lanes = self.lanes;
                (0..n)
                    .map(|t| {
                        let group = t / lanes * lanes;
                        let end = (group + lanes).min(n);
                        Val::F((group..end).filter(|s| mask[*s]).map(|s| v[s].f()).sum())
                    })
                    .collect()
            }
            Intrinsic::Rsqrt => unary(self, |x| 1.0 / x.sqrt()),
            Intrinsic::Exp => unary(self, f32::exp),
            Intrinsic::Tanh => unary(self, f32::tanh),
            Intrinsic::Cos => unary(self, f32::cos),
            Intrinsic::Sin => unary(self, f32::sin),
            Intrinsic::Max => {
                let (a, b) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
                a.into_iter()
                    .zip(b)
                    .map(|(x, y)| Val::F(x.f().max(y.f())))
                    .collect()
            }
            Intrinsic::Pow => {
                let (a, b) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
                a.into_iter()
                    .zip(b)
                    .map(|(x, y)| Val::F(x.f().powf(y.f())))
                    .collect()
            }
            Intrinsic::Clamp => {
                let v = self.eval(&args[0], mask);
                let (lo, hi) = (self.eval(&args[1], mask), self.eval(&args[2], mask));
                (0..n)
                    .map(|t| Val::F(v[t].f().clamp(lo[t].f(), hi[t].f())))
                    .collect()
            }
            Intrinsic::Load4 => {
                let Expr::Var(name) = &args[0] else {
                    panic!("a load names its buffer");
                };
                let (arg, at) = (self.arg(name), self.eval(&args[1], mask));
                (0..n)
                    .map(|t| {
                        if mask[t] {
                            self.load4(&arg, at[t].l())
                        } else {
                            Val::V([0.0; 4])
                        }
                    })
                    .collect()
            }
            Intrinsic::Store4 => {
                let Expr::Var(name) = &args[0] else {
                    panic!("a store names its buffer");
                };
                let arg = self.arg(name);
                let (at, v) = (self.eval(&args[1], mask), self.eval(&args[2], mask));
                for t in (0..n).filter(|t| mask[*t]) {
                    for (i, x) in v[t].v().into_iter().enumerate() {
                        self.store(&arg, at[t].l() + i as u64, Val::F(x));
                    }
                }
                vec![Val::B(false); n]
            }
            Intrinsic::Zero4 => vec![Val::V([0.0; 4]); n],
            Intrinsic::Make4 => {
                let parts: Vec<Vec<Val>> = args.iter().map(|a| self.eval(a, mask)).collect();
                (0..n)
                    .map(|t| Val::V([0, 1, 2, 3].map(|i| parts[i][t].f())))
                    .collect()
            }
            Intrinsic::ToBits => {
                let v = self.eval(&args[0], mask);
                v.into_iter().map(|x| Val::U(x.f().to_bits())).collect()
            }
            Intrinsic::Field(m) => {
                let (w, l) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
                (0..n)
                    .map(|t| {
                        let x = field(w[t].u(), l[t].u(), m.bits());
                        Val::F(match m {
                            Mini::F16 => f32::from(F16(x as u16)),
                            Mini::E4M3 => f32::from(E4M3(x as u8)),
                            Mini::E2M1 => f32::from(E2M1(x as u8)),
                        })
                    })
                    .collect()
            }
            Intrinsic::Bits | Intrinsic::Signed => {
                let (w, l, n) = (
                    self.eval(&args[0], mask),
                    self.eval(&args[1], mask),
                    self.eval(&args[2], mask),
                );
                (0..n.len())
                    .map(|t| {
                        let field = field(w[t].u(), l[t].u(), n[t].u());
                        match i {
                            Intrinsic::Bits => Val::U(field),
                            _ => Val::F(sign_extend(field, n[t].u()) as f32),
                        }
                    })
                    .collect()
            }
            Intrinsic::FromBits => {
                let v = self.eval(&args[0], mask);
                v.into_iter()
                    .map(|x| Val::F(f32::from_bits(x.u())))
                    .collect()
            }
            Intrinsic::Word | Intrinsic::Half => {
                let Expr::Var(name) = &args[0] else {
                    panic!("a word read names its buffer");
                };
                let width = if i == Intrinsic::Word { 4 } else { 2 };
                let (arg, at) = (self.arg(name), self.eval(&args[1], mask));
                let bytes = &self.buffers[arg.buffer];
                (0..n)
                    .map(|t| {
                        let at = arg.skip + at[t].u() as usize;
                        let mut raw = [0u8; 4];
                        raw[..width].copy_from_slice(&bytes[at..at + width]);
                        Val::U(u32::from_le_bytes(raw))
                    })
                    .collect()
            }
            Intrinsic::KeptRow => {
                let layout = self
                    .layout
                    .expect("a kernel reading kept rows names its layout");
                let values: Vec<u32> = (layout.params().iter())
                    .map(|p| match self.args.get(*p) {
                        Some(ArgVal::U32(v)) => *v,
                        other => panic!("layout scalar `{p}` is {other:?}"),
                    })
                    .collect();
                let (s, p) = (self.eval(&args[0], mask), self.eval(&args[1], mask));
                (0..n)
                    .map(|t| Val::U(layout.at(s[t].u(), p[t].u(), &values)))
                    .collect()
            }
        }
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
            } => {
                let (rows, cols, _) = self.tile_shape(acc);
                let (Expr::Var(a), Expr::Var(b)) = (a, b) else {
                    panic!("tiles multiply group memory by name");
                };
                let (a_at, b_at, stride) = (
                    self.eval(a_at, mask),
                    self.eval(b_at, mask),
                    self.eval(stride, mask),
                );
                let simds = n.div_ceil(self.lanes);
                for s in 0..simds {
                    // Any live lane of the SIMD group names its slabs.
                    let Some(t) =
                        (s * self.lanes..((s + 1) * self.lanes).min(n)).find(|t| mask[*t])
                    else {
                        continue;
                    };
                    let (a0, b0, ld) = (
                        a_at[t].u() as usize,
                        b_at[t].u() as usize,
                        stride[t].u() as usize,
                    );
                    let (sa, sb) = (&self.shared[a], &self.shared[b]);
                    let mut block =
                        std::mem::take(&mut self.tiles.get_mut(acc).expect("tile sums").2[s]);
                    let p = self.precision;
                    for i in 0..rows {
                        for j in 0..cols {
                            let sum = &mut block[i * cols + j];
                            // The unit sums `p.depth` products, then adds them in.
                            for chunk in (0..*depth).step_by(p.depth.max(1)) {
                                let mut part = 0.0f32;
                                for k in chunk..(chunk + p.depth.max(1)).min(*depth) {
                                    let (ah, al) = p.split(sa[a0 + i * ld + k].f());
                                    let (bh, bl) = p.split(sb[b0 + j * ld + k].f());
                                    // The split's products: hi·hi, hi·lo, lo·hi.
                                    part = ah.mul_add(bh, part);
                                    part = ah.mul_add(bl, part);
                                    part = al.mul_add(bh, part);
                                }
                                *sum = p.accumulate(*sum, part);
                            }
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

/// Integers wrap. Anything with a float is a float.
fn binary(op: BinOp, a: Val, b: Val) -> Val {
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
        _ if long => {
            let (x, y) = (a.l(), b.l());
            Val::L(match op {
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
            })
        }
        _ => {
            let (x, y) = (a.u(), b.u());
            Val::U(match op {
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
            })
        }
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
    for (param, arg) in body.params.iter().zip(args) {
        let name = match param {
            Param::Buffer { name, .. }
            | Param::Scalar { name, .. }
            | Param::Shared { name, .. }
            | Param::Tiles { name }
            | Param::Array { name, .. } => name.clone(),
        };
        named.insert(name, *arg);
    }
    for g in 0..grid.groups {
        let mut group = Group {
            threads: grid.threads,
            lanes,
            precision,
            first: g * grid.threads,
            vars: Vec::new(),
            arrays: HashMap::new(),
            shared: HashMap::new(),
            tiles: HashMap::new(),
            done: vec![false; grid.threads],
            layout: body.layout,
            args: &named,
            buffers: &mut *buffers,
        };
        let mask = vec![true; grid.threads];
        group.block(&body.stmts, &mask);
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
