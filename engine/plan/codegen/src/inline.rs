//! One kernel's tree inlined into another's, at build time.

use crate::ir::{Body, Expr, Helper, Intrinsic, Param, Stmt};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The prefix of the fields a kernel's returned group declares.
pub const BUILT: &str = "self";
const BUILT_PREFIX: &str = "self__";

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

/// The variable that holds `field` of group `group`.
pub fn member(group: &str, field: &str) -> String {
    format!("{group}__{field}")
}

/// Parameters that pass together, one per field, scalars by value.
pub trait Group {
    /// Each field as a parameter named `{name}__{field}`.
    fn params(name: &str) -> Vec<Param>;
}

/// The caller's name for each field of group `arg`.
pub fn fields<G: Group>(arg: Expr) -> Vec<Expr> {
    let Expr::Var(name) = arg else {
        panic!("a group passes by name");
    };
    G::params(&name)
        .iter()
        .map(|p| Expr::var(p.name()))
        .collect()
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
        self.inline(pre, call, None);
    }

    /// A call bound by `let name`: its value or group.
    pub fn bind(&mut self, pre: &mut Vec<Stmt>, call: Inline, name: &str) -> Option<Expr> {
        self.inline(pre, call, Some(name))
    }

    /// A call as a value, after the statements in `pre`.
    pub fn expr(&mut self, pre: &mut Vec<Stmt>, call: Inline) -> Expr {
        let entry = match &call.callee {
            Callee::Kernel(body) => body.entry.clone(),
            Callee::Helper(h) => h.name.clone(),
        };
        match self.inline(pre, call, None) {
            Some(value) => value,
            None => panic!("`{entry}` has no value"),
        }
    }

    fn inline(&mut self, pre: &mut Vec<Stmt>, call: Inline, bound: Option<&str>) -> Option<Expr> {
        self.calls += 1;
        let Inline { callee, args } = call;
        let mut body = match callee {
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
        let suffix = format!("__{}", self.calls);
        let mut names = Names::default();
        for (param, arg) in body.params.into_iter().zip(args) {
            match param {
                // Buffers, group memory, tiles and arrays pass by name.
                Param::Buffer { name, .. }
                | Param::Shared { name, .. }
                | Param::Tiles { name }
                | Param::Array { name, .. } => {
                    let Expr::Var(named) = arg else {
                        panic!("`{}` takes `{name}` by name", body.entry);
                    };
                    if name != named {
                        names.insert(name, Target::Exact(named));
                    }
                }
                // A name the callee never assigns reads as itself.
                Param::Scalar { name, .. }
                    if matches!(arg, Expr::Var(_)) && !assigns(&body.stmts, &name) =>
                {
                    let Expr::Var(named) = arg else {
                        unreachable!()
                    };
                    if name != named {
                        names.insert(name, Target::Exact(named));
                    }
                }
                Param::Scalar { name, ty } => {
                    let fresh = with_suffix(&name, &suffix);
                    pre.push(Stmt::Let {
                        name: fresh.clone(),
                        init: Expr::cast(ty, arg),
                    });
                    names.insert(name, Target::Exact(fresh));
                }
            }
        }
        bind_declared(&body.stmts, bound, &mut names);
        let renamer = Rename {
            names,
            suffix: &suffix,
            call: None,
        };
        renamer.block(&mut body.stmts);
        pre.append(&mut body.stmts);
        for h in body.helpers {
            self.helper(h);
        }
        body.ret.map(|mut e| {
            renamer.expr(&mut e);
            e
        })
    }

    fn helper(&mut self, h: Helper) {
        if !self.helpers.iter().any(|k| k.name == h.name) {
            self.helpers.push(h);
        }
    }
}

fn with_suffix(name: &str, suffix: &str) -> String {
    let mut out = String::with_capacity(name.len() + suffix.len());
    out.push_str(name);
    out.push_str(suffix);
    out
}

fn bind_name(name: &str, bound: Option<&str>, names: &mut Names) {
    // A built group's fields take the bound name.
    let target = match (bound, name.strip_prefix(BUILT_PREFIX)) {
        (Some(to), Some(field)) => Target::Exact(member(to, field)),
        _ => Target::Suffix,
    };
    names.insert(name.to_owned(), target);
}

fn bind_declared(stmts: &[Stmt], bound: Option<&str>, names: &mut Names) {
    for s in stmts {
        match s {
            Stmt::Let { name, .. }
            | Stmt::Array { name, .. }
            | Stmt::Shared { name, .. }
            | Stmt::AtomicShared { name, .. }
            | Stmt::TileAcc { name, .. }
            | Stmt::TileHeld { name, .. } => bind_name(name, bound, names),
            Stmt::For { var, body, .. } => {
                bind_name(var, bound, names);
                bind_declared(body, bound, names);
            }
            Stmt::If { then, els, .. } => {
                bind_declared(then, bound, names);
                bind_declared(els, bound, names);
            }
            Stmt::TileEach { body, .. } => bind_declared(body, bound, names),
            _ => {}
        }
    }
}

/// Fast non-cryptographic hasher for short local symbol tables.
#[derive(Default)]
struct SymHash(u64);

impl Hasher for SymHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
        self.0 = h;
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<SymHash>>;

enum Target {
    Exact(String),
    Suffix,
}

/// Inline small map, promoting to a hash map when large.
enum Names {
    Small(Vec<(String, Target)>),
    Many(FastMap<String, Target>),
}

impl Default for Names {
    fn default() -> Self {
        Self::Small(Vec::new())
    }
}

impl Names {
    fn is_empty(&self) -> bool {
        match self {
            Self::Small(v) => v.is_empty(),
            Self::Many(m) => m.is_empty(),
        }
    }

    fn get(&self, key: &str) -> Option<&Target> {
        match self {
            Self::Small(v) => v.iter().find(|(k, _)| k == key).map(|(_, to)| to),
            Self::Many(m) => m.get(key),
        }
    }

    fn insert(&mut self, key: String, val: Target) {
        match self {
            Self::Small(v) => {
                if let Some((_, slot)) = v.iter_mut().find(|(k, _)| *k == key) {
                    *slot = val;
                    return;
                }
                if v.len() < 12 {
                    v.push((key, val));
                    return;
                }
                let mut map = FastMap::with_capacity_and_hasher(32, Default::default());
                for (k, old) in v.drain(..) {
                    map.insert(k, old);
                }
                map.insert(key, val);
                *self = Self::Many(map);
            }
            Self::Many(m) => {
                m.insert(key, val);
            }
        }
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

/// Rewrites each `intrinsic` call in `stmts` with `with` in place.
pub(crate) fn replace(stmts: &mut [Stmt], intrinsic: Intrinsic, with: Replace<'_>) {
    let names = Names::default();
    let call = Some((intrinsic, with));
    Rename {
        names,
        suffix: "",
        call,
    }
    .block(stmts);
}

/// Gives a call's value from its arguments.
type Replace<'a> = &'a dyn Fn(Vec<Expr>) -> Expr;

/// Rewrites a tree in place with renamed symbols.
struct Rename<'a> {
    names: Names,
    suffix: &'a str,
    /// Calls of this intrinsic become the function's value.
    call: Option<(Intrinsic, Replace<'a>)>,
}

impl Rename<'_> {
    fn name(&self, name: &mut String) {
        if self.names.is_empty() {
            return;
        }
        match self.names.get(name) {
            Some(Target::Exact(to)) => {
                name.clear();
                name.push_str(to);
            }
            Some(Target::Suffix) => name.push_str(self.suffix),
            None => {}
        }
    }

    fn block(&self, stmts: &mut [Stmt]) {
        for s in stmts {
            self.stmt(s);
        }
    }

    fn expr(&self, e: &mut Expr) {
        match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => {}
            Expr::Var(name) => self.name(name),
            Expr::Neg(a) | Expr::Not(a) | Expr::Cast(_, a) | Expr::Field(a, _) => self.expr(a),
            Expr::Bin(_, a, b) | Expr::Index(a, b) => {
                self.expr(a);
                self.expr(b);
            }
            Expr::Select(c, a, b) => {
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            Expr::Fn(_, _, args) => {
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Call(i, args) => {
                for a in args.iter_mut() {
                    self.expr(a);
                }
                if let Some((to, with)) = self.call
                    && to == *i
                {
                    let taken = std::mem::take(args);
                    *e = with(taken);
                }
            }
        }
    }

    fn stmt(&self, s: &mut Stmt) {
        match s {
            Stmt::Let { name, init } => {
                self.name(name);
                self.expr(init);
            }
            Stmt::Array { name, .. }
            | Stmt::Shared { name, .. }
            | Stmt::AtomicShared { name, .. }
            | Stmt::TileAcc { name, .. }
            | Stmt::TileHeld { name, .. } => self.name(name),
            Stmt::AtomicAdd { name, at, value } => {
                self.name(name);
                self.expr(at);
                self.expr(value);
            }
            Stmt::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                body,
                ..
            } => {
                self.name(var);
                self.expr(from);
                self.expr(to);
                self.expr(step);
                self.block(body);
            }
            Stmt::If { cond, then, els } => {
                self.expr(cond);
                self.block(then);
                self.block(els);
            }
            Stmt::Return | Stmt::Barrier => {}
            Stmt::Expr(e) => self.expr(e),
            Stmt::TileHold {
                held,
                a,
                a_at,
                stride,
            } => {
                self.name(held);
                self.expr(a);
                self.expr(a_at);
                self.expr(stride);
            }
            Stmt::TileBlock {
                acc,
                a,
                a_at,
                b,
                b_at,
                stride,
                apart,
                ..
            } => {
                self.name(acc);
                self.expr(a);
                self.expr(a_at);
                self.expr(b);
                self.expr(b_at);
                self.expr(stride);
                self.expr(apart);
            }
            Stmt::TileEach { acc, body } => {
                self.name(acc);
                self.block(body);
            }
            Stmt::TileZero { acc } => self.name(acc),
            Stmt::TileScale { acc, by } => {
                self.name(acc);
                self.expr(by);
            }
        }
    }
}
