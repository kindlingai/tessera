//! A kernel body as a tree, printed and costed.

use crate::LANES;
use crate::dialect::{self, Text};
use crate::form::Load;
use crate::print::{Kind, Printer};
use crate::wgsl::store4;

/// A value's type, for the printer.
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
    pub(crate) fn c(self) -> &'static str {
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
    pub(crate) fn c(self) -> &'static str {
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

    pub fn compares(self) -> bool {
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

/// Calls a kernel may make besides arithmetic, from the table.
macro_rules! intrinsics {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:ident),*)
        $(-> $ret:ident)? { device: $dev:expr, text: $text:expr, wgsl: $wgsl:expr, host: $host:expr,
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

            pub(crate) fn print(self, p: &mut Printer, args: &[Expr]) -> String {
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
pub(crate) fn element(form: &dyn Load) -> Ty {
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
    /// A SIMD group's slab of A, kept in registers.
    TileHeld {
        name: String,
        rows: usize,
        depth: usize,
    },
    /// Copies a slab of A from group memory into `held`.
    TileHold {
        held: String,
        a: Expr,
        a_at: Expr,
        stride: Expr,
    },
    /// Adds `depth` staged columns of A times B to `acc`.
    TileBlock {
        acc: String,
        a: Expr,
        /// Ignored when `a` is a held slab.
        a_at: Expr,
        b: Expr,
        b_at: Expr,
        stride: Expr,
        depth: usize,
        /// Zero: f32. Else bf16 pairs, A in `parts` slabs.
        parts: usize,
        /// Words between A's slabs.
        apart: Expr,
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

/// A kernel parameter. The last two are for callees.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    Buffer {
        name: String,
        form: &'static dyn Load,
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

impl Param {
    pub fn name(&self) -> &str {
        match self {
            Param::Buffer { name, .. }
            | Param::Scalar { name, .. }
            | Param::Shared { name, .. }
            | Param::Tiles { name }
            | Param::Array { name, .. } => name,
        }
    }
}

/// A whole kernel, before any dialect.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    pub entry: String,
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

impl Body {
    /// The text of this body, with buffers and scalars declared.
    pub fn text(&self) -> Text {
        assert!(self.ret.is_none(), "`{}` has a value: call it", self.entry);
        let mut p = Printer::new(&self.params, false);
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
pub(crate) fn gather(helpers: &[Helper], found: &mut Vec<Helper>) {
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
        let mut p = Printer::new(&self.body.params, false);
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

/// Leaks a scalar name, so it outlives its source text.
fn leak(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}
