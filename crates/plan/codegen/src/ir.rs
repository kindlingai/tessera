//! A kernel body as a tree, printed and costed.

use crate::builder::LANES;
use crate::dialect::Text;
use graph::DType;
use sim::Ops;
use std::collections::HashMap;

/// A value's type, as the printer needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F32,
    U32,
    U64,
    Bool,
    /// Four floats loaded at once.
    F32x4,
    /// Four bf16 loaded at once, as two words.
    Bf16x4,
}

impl Ty {
    fn c(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::Bool => "bool",
            Ty::F32x4 => "float4",
            Ty::Bf16x4 => "uint2",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinOp {
    fn c(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    fn compares(self) -> bool {
        use BinOp::*;
        matches!(self, Lt | Le | Gt | Ge | Eq | Ne | And | Or)
    }
}

/// What a kernel may call besides arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    /// The thread's index in the grid, and in its group.
    Gid,
    Lid,
    /// The thread's lane among the 32 that shuffle together.
    Lane,
    /// `shfl_xor(value, mask)`.
    ShflXor,
    /// `sum32(value)`: the sum over the 32 lanes.
    Sum32,
    Rsqrt,
    Exp,
    Tanh,
    Max,
    /// `clamp(value, low, high)`.
    Clamp,
    /// `load4(buffer, index)`: four elements as floats.
    Load4,
    /// `store4(buffer, index, value)`.
    Store4,
    /// A zeroed `float4`.
    Zero4,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    Var(String),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Cast(Ty, Box<Expr>),
    /// A buffer, shared or local array element.
    Index(Box<Expr>, Box<Expr>),
    /// `.x`, `.y`, `.z` or `.w` of a four-wide value.
    Field(Box<Expr>, char),
    Call(Intrinsic, Vec<Expr>),
}

impl Expr {
    pub fn var(name: &str) -> Expr {
        Expr::Var(name.into())
    }

    pub fn lit(n: impl Into<i64>) -> Expr {
        Expr::Int(n.into())
    }

    pub fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::Bin(op, Box::new(a), Box::new(b))
    }

    pub fn index(base: Expr, at: Expr) -> Expr {
        Expr::Index(Box::new(base), Box::new(at))
    }

    pub fn field(base: Expr, name: char) -> Expr {
        Expr::Field(Box::new(base), name)
    }

    pub fn cast(ty: Ty, e: Expr) -> Expr {
        Expr::Cast(ty, Box::new(e))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        init: Expr,
    },
    /// An uninitialized local array of `len` elements.
    Array {
        name: String,
        ty: Ty,
        len: usize,
    },
    /// Group memory of `len` elements.
    Shared {
        name: String,
        ty: Ty,
        len: usize,
    },
    /// `target = value`, or `target op= value` with `op`.
    Assign {
        target: Expr,
        op: Option<BinOp>,
        value: Expr,
    },
    For {
        var: String,
        from: Expr,
        to: Expr,
        step: Expr,
        /// Trips unrolled. Zero leaves it to the compiler.
        unroll: usize,
        body: Vec<Stmt>,
    },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Vec<Stmt>,
    },
    Return,
    Barrier,
    Expr(Expr),
}

/// A kernel parameter: a buffer of `dtype`, or a scalar.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Buffer {
        name: String,
        dtype: DType,
        writes: bool,
    },
    Scalar {
        name: String,
        ty: Ty,
    },
}

/// A whole kernel, before any dialect.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub entry: String,
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
}

/// Every name in scope and what it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Value(Ty),
    /// A buffer parameter of this element type.
    Buffer(DType),
    /// Group memory, or a local array, of this element type.
    Array(Ty),
}

/// Prints a body as the dialect-neutral text.
struct Printer<'a> {
    scope: Vec<HashMap<String, Kind>>,
    lines: Vec<String>,
    indent: usize,
    lane: bool,
    params: &'a [Param],
}

impl Printer<'_> {
    fn kind(&self, name: &str) -> Kind {
        for scope in self.scope.iter().rev() {
            if let Some(k) = scope.get(name) {
                return *k;
            }
        }
        panic!("`{name}` is not in scope")
    }

    fn bind(&mut self, name: &str, kind: Kind) {
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), kind);
    }

    fn line(&mut self, text: String) {
        self.lines
            .push(format!("{}{text}", "    ".repeat(self.indent)));
    }

    /// The type an expression has, from the names it uses.
    fn ty(&self, e: &Expr) -> Ty {
        match e {
            Expr::Int(_) => Ty::U32,
            Expr::Float(_) => Ty::F32,
            Expr::Bool(_) => Ty::Bool,
            Expr::Var(name) => match self.kind(name) {
                Kind::Value(ty) => ty,
                other => panic!("`{name}` is {other:?}, not a value"),
            },
            Expr::Neg(a) => self.ty(a),
            Expr::Not(_) => Ty::Bool,
            Expr::Bin(op, a, b) => match op.compares() {
                true => Ty::Bool,
                false => wider(self.ty(a), self.ty(b)),
            },
            Expr::Cast(ty, _) => *ty,
            Expr::Index(base, _) => match &**base {
                Expr::Var(name) => match self.kind(name) {
                    Kind::Array(ty) => ty,
                    _ => Ty::F32,
                },
                _ => Ty::F32,
            },
            Expr::Field(..) => Ty::F32,
            Expr::Call(i, args) => match i {
                Intrinsic::Gid | Intrinsic::Lid | Intrinsic::Lane => Ty::U32,
                Intrinsic::Load4 => match self.buffer(&args[0]) {
                    DType::Bf16 => Ty::Bf16x4,
                    _ => Ty::F32x4,
                },
                Intrinsic::Zero4 => Ty::F32x4,
                _ => Ty::F32,
            },
        }
    }

    /// The element type of a buffer parameter.
    fn buffer(&self, e: &Expr) -> DType {
        match e {
            Expr::Var(name) => match self.kind(name) {
                Kind::Buffer(dtype) => dtype,
                other => panic!("`{name}` is {other:?}, not a buffer"),
            },
            _ => panic!("a buffer is named directly"),
        }
    }

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::Int(n) => n.to_string(),
            Expr::Float(x) => float(*x),
            Expr::Bool(b) => b.to_string(),
            Expr::Var(name) => name.clone(),
            Expr::Neg(a) => format!("(-{})", self.expr(a)),
            Expr::Not(a) => format!("(!{})", self.expr(a)),
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.expr(a), self.expr(b));
                format!("({a} {} {b})", op.c())
            }
            Expr::Cast(ty, a) => format!("({}){}", ty.c(), self.expr(a)),
            Expr::Index(base, at) => {
                let at = self.expr(at);
                let Expr::Var(name) = &**base else {
                    panic!("an index names what it reads");
                };
                match self.kind(name) {
                    Kind::Buffer(DType::Bf16) => format!("f_bits((u32){name}[{at}] << 16)"),
                    _ => format!("{name}[{at}]"),
                }
            }
            Expr::Field(base, h) => {
                let ty = self.ty(base);
                let base = self.expr(base);
                match ty {
                    // Two bf16 share each word: low half first.
                    Ty::Bf16x4 => match h {
                        'x' => format!("f_bits({base}.x << 16)"),
                        'y' => format!("f_bits({base}.x & 0xffff0000u)"),
                        'z' => format!("f_bits({base}.y << 16)"),
                        _ => format!("f_bits({base}.y & 0xffff0000u)"),
                    },
                    _ => format!("{base}.{h}"),
                }
            }
            Expr::Call(i, args) => self.call(*i, args),
        }
    }

    fn call(&mut self, i: Intrinsic, args: &[Expr]) -> String {
        let arg = |p: &mut Self, n: usize| p.expr(&args[n]);
        match i {
            Intrinsic::Gid => "gid".into(),
            Intrinsic::Lid => "lid".into(),
            Intrinsic::Lane => {
                self.lane = true;
                "lane".into()
            }
            Intrinsic::ShflXor => format!("f_shfl_xor({}, {})", arg(self, 0), arg(self, 1)),
            Intrinsic::Sum32 => panic!("sum32 is a statement: `v = sum32(v)`"),
            Intrinsic::Rsqrt => format!("f_rsqrt({})", arg(self, 0)),
            Intrinsic::Exp => format!("f_exp({})", arg(self, 0)),
            Intrinsic::Tanh => format!("f_tanh({})", arg(self, 0)),
            Intrinsic::Max => format!("f_max({}, {})", arg(self, 0), arg(self, 1)),
            Intrinsic::Clamp => {
                let (v, lo, hi) = (arg(self, 0), arg(self, 1), arg(self, 2));
                format!("f_clamp({v}, {lo}, {hi})")
            }
            Intrinsic::Load4 => {
                let (dtype, at) = (self.buffer(&args[0]), arg(self, 1));
                let Expr::Var(name) = &args[0] else {
                    unreachable!()
                };
                match dtype {
                    DType::Bf16 => format!("f_u32x2({name} + {at})"),
                    _ => format!("f_f32x4({name} + {at})"),
                }
            }
            Intrinsic::Store4 => {
                let Expr::Var(name) = &args[0] else {
                    panic!("a store names its buffer");
                };
                let name = name.clone();
                format!("f_f32x4({name} + {}) = {}", arg(self, 1), arg(self, 2))
            }
            Intrinsic::Zero4 => "float4(0.0f, 0.0f, 0.0f, 0.0f)".into(),
        }
    }

    fn block(&mut self, stmts: &[Stmt]) {
        self.scope.push(HashMap::new());
        self.indent += 1;
        for s in stmts {
            self.stmt(s);
        }
        self.indent -= 1;
        self.scope.pop();
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, init } => {
                let ty = self.ty(init);
                // A zeroed four leaves the compiler to fill it.
                let text = match init {
                    Expr::Call(Intrinsic::Zero4, _) => format!("float4 {name};"),
                    _ => format!("{} {name} = {};", ty.c(), self.expr(init)),
                };
                self.line(text);
                self.bind(name, Kind::Value(ty));
            }
            Stmt::Array { name, ty, len } => {
                self.line(format!("{} {name}[{len}];", ty.c()));
                self.bind(name, Kind::Array(*ty));
            }
            Stmt::Shared { name, ty, len } => {
                self.line(format!("f_group({}, {name}, {len});", ty.c()));
                self.bind(name, Kind::Array(*ty));
            }
            Stmt::Assign {
                target,
                op,
                value: Expr::Call(Intrinsic::Sum32, args),
            } => {
                assert!(op.is_none() && args[0] == *target, "`v = sum32(v)`");
                let v = self.expr(target);
                let mut m = LANES / 2;
                while m > 0 {
                    self.line(format!("{v} += f_shfl_xor({v}, {m});"));
                    m /= 2;
                }
            }
            Stmt::Assign { target, op, value } => {
                let (t, v) = (self.expr(target), self.expr(value));
                let op = op.map_or(String::new(), |op| op.c().into());
                self.line(format!("{t} {op}= {v};"));
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                unroll,
                body,
            } => {
                // Loops over constants unroll unless told otherwise.
                let constant =
                    matches!((from, to, step), (Expr::Int(_), Expr::Int(_), Expr::Int(_)));
                match unroll {
                    0 if constant => self.line("#pragma unroll".into()),
                    0 => {}
                    1 => self.line("#pragma unroll".into()),
                    n => self.line(format!("#pragma unroll {n}")),
                }
                let (from, to, step) = (self.expr(from), self.expr(to), self.expr(step));
                self.line(format!(
                    "for (u32 {var} = {from}; {var} < {to}; {var} += {step}) {{"
                ));
                self.scope.push(HashMap::new());
                self.bind(var, Kind::Value(Ty::U32));
                self.block(body);
                self.scope.pop();
                self.line("}".into());
            }
            Stmt::If { cond, then, els } => {
                let cond = self.expr(cond);
                self.line(format!("if ({cond}) {{"));
                self.block(then);
                if !els.is_empty() {
                    self.line("} else {".into());
                    self.block(els);
                }
                self.line("}".into());
            }
            Stmt::Return => self.line("return;".into()),
            Stmt::Barrier => self.line("f_barrier();".into()),
            Stmt::Expr(e) => {
                let text = self.expr(e);
                self.line(format!("{text};"));
            }
        }
    }
}

/// The wider of two types in arithmetic.
fn wider(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        (Ty::F32, _) | (_, Ty::F32) => Ty::F32,
        (Ty::U64, _) | (_, Ty::U64) => Ty::U64,
        (Ty::F32x4, _) | (_, Ty::F32x4) => Ty::F32x4,
        _ => Ty::U32,
    }
}

/// A float literal the dialects accept.
fn float(x: f64) -> String {
    let text = format!("{x:?}");
    match text.contains('.') || text.contains('e') {
        true => format!("{text}f"),
        false => format!("{text}.0f"),
    }
}

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
            Expr::Call(i, args) => {
                for a in args {
                    self.expr(a);
                }
                match i {
                    Intrinsic::Rsqrt | Intrinsic::Exp | Intrinsic::Tanh => {
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
        }
    }
}

impl Body {
    /// The text of this body, with buffers and scalars declared.
    pub fn text(&self) -> Text {
        let mut p = Printer {
            scope: vec![HashMap::new()],
            lines: Vec::new(),
            indent: 1,
            lane: false,
            params: &self.params,
        };
        let (mut buffers, mut scalars) = (Vec::new(), Vec::new());
        for param in p.params {
            match param {
                Param::Buffer {
                    name,
                    dtype,
                    writes,
                } => {
                    p.bind(name, Kind::Buffer(*dtype));
                    let ty = match dtype {
                        DType::Bf16 => "u16",
                        DType::U32 => "u32",
                        DType::I32 => "int",
                        _ => "float",
                    };
                    buffers.push((name.clone(), ty, *writes));
                }
                Param::Scalar { name, ty } => {
                    p.bind(name, Kind::Value(*ty));
                    scalars.push((leak(name), ty.c()));
                }
            }
        }
        for s in &self.stmts {
            p.stmt(s);
        }
        let mut lines = p.lines;
        if p.lane {
            lines.insert(0, format!("    u32 lane = gid % {LANES};"));
        }
        Text {
            entry: self.entry.clone(),
            buffers,
            scalars,
            body: lines.join("\n") + "\n",
        }
    }

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

/// Scalar names outlive the text that uses them.
fn leak(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}
