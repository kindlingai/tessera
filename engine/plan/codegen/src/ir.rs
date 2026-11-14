//! A kernel body as a tree, printed and costed.

use crate::LANES;
use crate::dialect::{self, Text};
use crate::form::{Form, U8_FORM};
use crate::kept::Layout;
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
    /// A pointer into a buffer's bytes.
    Bytes,
}

impl Ty {
    fn c(self) -> &'static str {
        match self {
            Ty::F32 => "float",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::Bool => "bool",
            Ty::F32x4 => "float4",
            Ty::Bytes => dialect::BYTES,
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
    BitXor,
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
            BinOp::BitXor => "^",
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

/// The `Ty` of an intrinsic's Rust return type.
macro_rules! ty {
    (u32) => {
        Ty::U32
    };
    (F32x4) => {
        Ty::F32x4
    };
    (f32) => {
        Ty::F32
    };
    // A statement's value is never read.
    () => {
        Ty::F32
    };
}

/// What a kernel may call besides arithmetic, from the table.
macro_rules! intrinsics {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:ident),*)
        $(-> $ret:ident)? { device: $dev:expr, text: $text:expr, host: $host:expr,
        count: $count:expr $(,)? })*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Intrinsic {
            $($(#[$doc])* $var,)*
        }

        impl Intrinsic {
            pub const ALL: &[Intrinsic] = &[$(Intrinsic::$var),*];

            /// The type of the value a call gives.
            pub fn ty(self) -> Ty {
                match self {
                    $(Intrinsic::$var => ty!($($ret)?),)*
                }
            }

            /// Arguments a call takes, a method's receiver first.
            pub fn arity(self) -> usize {
                match self {
                    $(Intrinsic::$var => <[&str]>::len(&[$(stringify!($arg)),*]),)*
                }
            }

            fn print(self, p: &mut Printer, args: &[Expr]) -> String {
                let n = self.arity();
                assert_eq!(args.len(), n, "`{self:?}` takes {n} arguments");
                let text: fn(&mut Printer, &[Expr]) -> String = match self {
                    $(Intrinsic::$var => $text,)*
                };
                text(p, args)
            }
        }
    };
}

crate::intrinsic::table!(intrinsics);

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
    /// A helper call: its name, type and arguments.
    Fn(String, Ty, Vec<Expr>),
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
fn element(form: &dyn Form) -> Ty {
    match form.dtype().is_float() {
        true => Ty::F32,
        false => Ty::U32,
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
    /// A SIMD group's zeroed block of `rows` by `cols` sums.
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

/// A kernel parameter. Only a callee takes the last two.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Buffer {
        name: String,
        form: &'static dyn Form,
        writes: bool,
    },
    Scalar {
        name: String,
        ty: Ty,
    },
    /// The caller's group memory, by name.
    Shared {
        name: String,
        ty: Ty,
    },
    /// The caller's tile sums, by name.
    Tiles {
        name: String,
    },
    /// The caller's local array, by name.
    Array {
        name: String,
        ty: Ty,
    },
}

/// A whole kernel, before any dialect.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub entry: String,
    /// The kept-row layout `kept_row` reads, when a call does.
    pub layout: Option<&'static dyn Layout>,
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
    /// The value a kernel called from another gives back.
    pub ret: Option<Expr>,
    /// The helpers the body calls.
    pub helpers: Vec<Helper>,
}

/// A function the text defines once and kernels call.
#[derive(Clone, Debug, PartialEq)]
pub struct Helper {
    /// The name the text gives it.
    pub name: String,
    pub ret: Ty,
    /// The parameters, statements and value.
    pub body: Body,
}

/// Every name in scope and what it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Value(Ty),
    /// Four elements of this form, loaded at once.
    Wide(&'static dyn Form),
    /// A buffer parameter of this form.
    Buffer(&'static dyn Form),
    /// Group memory, or a local array, of this element type.
    Array(Ty),
    /// Tile sums over this many rows and columns.
    Tiles(usize, usize),
}

/// Prints a body as the dialect-neutral text.
struct Printer<'a> {
    /// Each name's kind and the name the text gives it.
    scope: Vec<HashMap<String, (Kind, String)>>,
    lines: Vec<String>,
    indent: usize,
    lane: bool,
    params: &'a [Param],
    layout: Option<&'static dyn Layout>,
}

impl Printer<'_> {
    fn kind(&self, name: &str) -> Kind {
        for scope in self.scope.iter().rev() {
            if let Some((k, _)) = scope.get(name) {
                return *k;
            }
        }
        panic!("`{name}` is not in scope")
    }

    /// The name the text gives `name`.
    fn c_name(&self, name: &str) -> String {
        let found = self.scope.iter().rev().find_map(|s| s.get(name));
        found.map_or_else(|| name.into(), |(_, c)| c.clone())
    }

    fn bind(&mut self, name: &str, kind: Kind) {
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), (kind, name.into()));
    }

    /// Binds a variable. A shadow gets a fresh C name.
    fn declare(&mut self, name: &str, kind: Kind) -> String {
        let taken = |p: &Self, c: &str| {
            let mut all = p.scope.iter().flat_map(|s| s.iter());
            all.any(|(n, (_, rendered))| n == c || rendered == c)
        };
        let mut c = name.to_string();
        let mut n = 0;
        while taken(self, &c) {
            n += 1;
            c = format!("{name}_{n}");
        }
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), (kind, c.clone()));
        c
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
                Kind::Wide(_) => Ty::F32x4,
                Kind::Buffer(form) if form.ty() == "u8" => Ty::Bytes,
                other => panic!("`{name}` is {other:?}, not a value"),
            },
            Expr::Neg(a) | Expr::Not(a) => self.ty(a),
            Expr::Bin(op, a, b) => match op.compares() {
                true => Ty::Bool,
                false => wider(self.ty(a), self.ty(b)),
            },
            Expr::Cast(ty, _) => *ty,
            Expr::Index(base, _) => match &**base {
                Expr::Var(name) => match self.kind(name) {
                    Kind::Array(ty) => ty,
                    Kind::Buffer(form) => element(form),
                    _ => Ty::F32,
                },
                _ => Ty::F32,
            },
            Expr::Field(..) => Ty::F32,
            Expr::Select(c, a, b) => match **c {
                Expr::Bool(false) => self.ty(b),
                _ => self.ty(a),
            },
            Expr::Fn(_, ty, _) => *ty,
            Expr::Call(i, _) => i.ty(),
        }
    }

    /// The form of a buffer parameter.
    fn buffer(&self, e: &Expr) -> &'static dyn Form {
        match e {
            Expr::Var(name) => match self.kind(name) {
                Kind::Buffer(form) => form,
                other => panic!("`{name}` is {other:?}, not a buffer"),
            },
            _ => panic!("a buffer is named directly"),
        }
    }

    /// The form behind a four-wide value from a buffer.
    fn wide(&self, e: &Expr) -> Option<&'static dyn Form> {
        match e {
            Expr::Var(name) => match self.kind(name) {
                Kind::Wide(form) => Some(form),
                _ => None,
            },
            Expr::Call(Intrinsic::Load4, args) => Some(self.buffer(&args[0])),
            _ => None,
        }
    }

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            // Past i32, a bare literal would widen the arithmetic.
            Expr::Int(n) if *n > i64::from(i32::MAX) => format!("{n}u"),
            Expr::Int(n) => n.to_string(),
            Expr::Float(x) => float(*x),
            Expr::Bool(b) => b.to_string(),
            Expr::Var(name) => self.c_name(name),
            Expr::Neg(a) => format!("(-{})", self.expr(a)),
            // On an integer, `!` flips every bit.
            Expr::Not(a) => match self.ty(a) {
                Ty::Bool => format!("(!{})", self.expr(a)),
                _ => format!("(~{})", self.expr(a)),
            },
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
                    Kind::Buffer(form) => form.load(name, &at),
                    _ => format!("{}[{at}]", self.c_name(name)),
                }
            }
            Expr::Field(base, h) => {
                let form = self.wide(base);
                let base = self.expr(base);
                match form {
                    Some(form) => form.part(&base, *h),
                    None => format!("{base}.{h}"),
                }
            }
            Expr::Call(i, args) => i.print(self, args),
            // A choice settled at build time leaves one branch.
            Expr::Select(c, a, b) if matches!(**c, Expr::Bool(_)) => {
                let taken = if **c == Expr::Bool(true) { a } else { b };
                self.expr(taken)
            }
            Expr::Select(c, a, b) => {
                let (c, a, b) = (self.expr(c), self.expr(a), self.expr(b));
                format!("({c} ? {a} : {b})")
            }
            Expr::Fn(name, _, args) => {
                let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                format!("{name}({})", args.join(", "))
            }
        }
    }

    /// `name(args)`, each argument printed in turn.
    fn spell(&mut self, name: &str, args: &[Expr]) -> String {
        let args: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
        format!("{name}({})", args.join(", "))
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
                // The initializer reads names bound before this one.
                let (ty, value, kind) = match (init, self.wide(init)) {
                    // A zeroed four leaves the compiler to fill it.
                    (Expr::Call(Intrinsic::Zero4, _), _) => ("float4", None, Kind::Value(ty)),
                    (_, Some(form)) => (form.wide(), Some(self.expr(init)), Kind::Wide(form)),
                    _ if ty == Ty::Bytes => (ty.c(), Some(self.expr(init)), Kind::Buffer(U8_FORM)),
                    _ => (ty.c(), Some(self.expr(init)), Kind::Value(ty)),
                };
                let name = self.declare(name, kind);
                match value {
                    Some(value) => self.line(format!("{ty} {name} = {value};")),
                    None => self.line(format!("{ty} {name};")),
                }
            }
            Stmt::Array { name, ty, len } => {
                let name = self.declare(name, Kind::Array(*ty));
                self.line(format!("{} {name}[{len}] = {{}};", ty.c()));
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
                self.scope.push(HashMap::new());
                let var = self.declare(var, Kind::Value(Ty::U32));
                self.line(format!(
                    "for (u32 {var} = {from}; {var} < {to}; {var} += {step}) {{"
                ));
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
        (Ty::Bytes, _) | (_, Ty::Bytes) => Ty::Bytes,
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
        assert!(self.ret.is_none(), "`{}` has a value: call it", self.entry);
        let mut p = Printer {
            scope: vec![HashMap::new()],
            lines: Vec::new(),
            indent: 1,
            lane: false,
            params: &self.params,
            layout: self.layout,
        };
        let (mut buffers, mut scalars) = (Vec::new(), Vec::new());
        let mut helpers: Vec<Helper> = Vec::new();
        for param in p.params {
            match param {
                Param::Buffer { name, form, writes } => {
                    p.bind(name, Kind::Buffer(*form));
                    buffers.push((name.clone(), form.ty(), *writes));
                    gather(&form.header(), &mut helpers);
                }
                Param::Scalar { name, ty } => {
                    p.bind(name, Kind::Value(*ty));
                    scalars.push((leak(name), ty.c()));
                }
                Param::Shared { name, .. } | Param::Tiles { name } | Param::Array { name, .. } => {
                    panic!("`{}` takes `{name}` from a caller", self.entry);
                }
            }
        }
        for s in &self.stmts {
            p.stmt(s);
        }
        gather(&self.helpers, &mut helpers);
        let mut lines = p.lines;
        if p.lane {
            lines.insert(0, format!("    u32 lane = gid % {LANES};"));
        }
        Text {
            entry: self.entry.clone(),
            buffers,
            scalars,
            headers: helpers.iter().map(Helper::text).collect(),
            body: lines.join("\n") + "\n",
        }
    }
}

/// Adds each helper after the ones it calls, once each.
fn gather(helpers: &[Helper], found: &mut Vec<Helper>) {
    for h in helpers {
        gather(&h.body.helpers, found);
        match found.iter().find(|f| f.name == h.name) {
            Some(f) => assert_eq!(*f, *h, "two helpers named `{}` differ", h.name),
            None => found.push(h.clone()),
        }
    }
}

impl Helper {
    /// The definition: `f_fn ret name(params) { .. }`.
    pub fn text(&self) -> String {
        let mut p = Printer {
            scope: vec![HashMap::new()],
            lines: Vec::new(),
            indent: 1,
            lane: false,
            params: &self.body.params,
            layout: None,
        };
        let mut params = Vec::new();
        for param in p.params {
            match param {
                Param::Buffer { name, form, .. } => {
                    p.bind(name, Kind::Buffer(*form));
                    params.push(dialect::pointer(form.ty(), name));
                }
                Param::Scalar { name, ty } => {
                    p.bind(name, Kind::Value(*ty));
                    params.push(format!("{} {name}", ty.c()));
                }
                Param::Shared { name, .. } | Param::Tiles { name } | Param::Array { name, .. } => {
                    panic!("`{}` takes `{name}` from a caller", self.name);
                }
            }
        }
        for s in &self.body.stmts {
            p.stmt(s);
        }
        let value = self.body.ret.as_ref().expect("a helper has a value");
        let value = p.expr(value);
        p.line(format!("return {value};"));
        assert!(!p.lane, "`{}` reads the lane: pass it in", self.name);
        dialect::function(self.ret.c(), &self.name, &params, &p.lines)
    }
}

/// Scalar names outlive the text that uses them.
fn leak(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}
