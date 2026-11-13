//! A kernel body as a tree, printed and costed.

use crate::LANES;
use crate::dialect::Text;
use graph::DType;
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

    pub(crate) fn compares(self) -> bool {
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
    Pow,
    Cos,
    Sin,
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
    /// `if cond { a } else { b }` as a value.
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
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

    pub fn select(cond: Expr, a: Expr, b: Expr) -> Expr {
        Expr::Select(Box::new(cond), Box::new(a), Box::new(b))
    }
}

/// A buffer element's type as a value.
fn element(dtype: DType) -> Ty {
    match dtype {
        DType::U32 | DType::I32 => Ty::U32,
        _ => Ty::F32,
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
    /// A warp's zeroed block of `rows` by `cols` sums.
    TileAcc {
        name: String,
        rows: usize,
        cols: usize,
    },
    /// Adds `depth` staged columns of A times B to `acc`.
    TileBlock {
        acc: String,
        a: Expr,
        a_at: Expr,
        b: Expr,
        b_at: Expr,
        stride: Expr,
        depth: usize,
    },
    /// Runs `body` per sum this lane holds: `r`, `c`, `v`.
    TileEach {
        acc: String,
        body: Vec<Stmt>,
    },
    /// Every sum back to zero.
    TileZero {
        acc: String,
    },
    /// Every sum times `by`, an expression in its row `r`.
    TileScale {
        acc: String,
        by: Expr,
    },
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
    /// Tile sums over this many rows and columns.
    Tiles(usize, usize),
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
                    Kind::Buffer(dtype) => element(dtype),
                    _ => Ty::F32,
                },
                _ => Ty::F32,
            },
            Expr::Field(..) => Ty::F32,
            Expr::Select(_, a, _) => self.ty(a),
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
            // Past i32, a bare literal would widen the arithmetic.
            Expr::Int(n) if *n > i64::from(i32::MAX) => format!("{n}u"),
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
            Expr::Select(c, a, b) => {
                let (c, a, b) = (self.expr(c), self.expr(a), self.expr(b));
                format!("({c} ? {a} : {b})")
            }
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
            Intrinsic::Pow => format!("f_pow({}, {})", arg(self, 0), arg(self, 1)),
            Intrinsic::Cos => format!("f_cos({})", arg(self, 0)),
            Intrinsic::Sin => format!("f_sin({})", arg(self, 0)),
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
            // A condition settled at build time leaves one branch.
            Stmt::If {
                cond: Expr::Bool(taken),
                then,
                els,
            } => {
                let branch = if *taken { then } else { els };
                if !branch.is_empty() {
                    self.line("{".into());
                    self.block(branch);
                    self.line("}".into());
                }
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
            Stmt::TileAcc { name, rows, cols } => {
                // The tile macros read `lane`.
                self.lane = true;
                let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
                self.line(format!("tile_acc {name}_t[{tm}][{tn}];"));
                self.line(format!("float {name}[{tm}][{tn}][TILE_COUNT];"));
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!("            {name}[i][j][e] = 0.0f;"));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
                self.bind(name, Kind::Tiles(*rows, *cols));
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
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                let (tm, tn) = (format!("({rows}) / TILE_M"), format!("({cols}) / TILE_N"));
                let a = format!("({} + {})", self.expr(a), self.expr(a_at));
                let b = format!("({} + {})", self.expr(b), self.expr(b_at));
                let stride = self.expr(stride);
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line(format!("        TILE_ZERO({acc}_t[i][j]);"));
                self.line("    }".into());
                self.line("}".into());
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 kk = 0; kk < {depth}; kk += TILE_K) {{"));
                self.line(format!("    tile_a ta[{tm}];"));
                self.line(format!("    tile_b tb[{tn}];"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
                self.line(format!(
                    "        TILE_LOAD_A(ta[i], {a} + i * TILE_M * ({stride}) + kk, {stride});"
                ));
                self.line("    }".into());
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line(format!(
                    "        TILE_LOAD_B(tb[j], {b} + j * TILE_N * ({stride}) + kk, {stride});"
                ));
                self.line("    }".into());
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("        #pragma unroll".into());
                self.line(format!("        for (u32 j = 0; j < {tn}; j++) {{"));
                self.line(format!(
                    "            TILE_MMA({acc}_t[i][j], ta[i], tb[j]);"
                ));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
                // The unit's sums truncate, so blocks add up in floats.
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < {tm}; i++) {{"));
                self.line("    #pragma unroll".into());
                self.line(format!("    for (u32 j = 0; j < {tn}; j++) {{"));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!(
                    "            {acc}[i][j][e] += TILE_VAL({acc}_t[i][j], e);"
                ));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileEach { acc, body } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        #pragma unroll".into());
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line("            u32 r = i * TILE_M + TILE_ROW(e);".into());
                self.line("            u32 c = j * TILE_N + TILE_COL(e);".into());
                self.line(format!("            float v = {acc}[i][j][e];"));
                self.scope.push(HashMap::new());
                for (name, ty) in [("r", Ty::U32), ("c", Ty::U32), ("v", Ty::F32)] {
                    self.bind(name, Kind::Value(ty));
                }
                self.indent += 2;
                self.block(body);
                self.indent -= 2;
                self.scope.pop();
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileZero { acc } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line(format!("            {acc}[i][j][e] = 0.0f;"));
                self.line("        }".into());
                self.line("    }".into());
                self.line("}".into());
            }
            Stmt::TileScale { acc, by } => {
                let Kind::Tiles(rows, cols) = self.kind(acc) else {
                    panic!("`{acc}` is not a tile block");
                };
                self.scope.push(HashMap::new());
                self.bind("r", Kind::Value(Ty::U32));
                let by = self.expr(by);
                self.scope.pop();
                self.line("#pragma unroll".into());
                self.line(format!("for (u32 i = 0; i < ({rows}) / TILE_M; i++) {{"));
                self.line(format!(
                    "    for (u32 j = 0; j < ({cols}) / TILE_N; j++) {{"
                ));
                self.line("        for (u32 e = 0; e < TILE_COUNT; e++) {".into());
                self.line("            u32 r = i * TILE_M + TILE_ROW(e);".into());
                self.line(format!("            {acc}[i][j][e] *= {by};"));
                self.line("        }".into());
                self.line("    }".into());
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
    if x.is_infinite() {
        let bits = if x > 0.0 {
            "0x7f800000u"
        } else {
            "0xff800000u"
        };
        return format!("f_bits({bits})");
    }
    let text = format!("{x:?}");
    match text.contains('.') || text.contains('e') {
        true => format!("{text}f"),
        false => format!("{text}.0f"),
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
}

/// Scalar names outlive the text that uses them.
fn leak(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}
