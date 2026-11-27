//! A kernel's declaration: its arguments, shape and source.

use crate::extent::Extent;
use crate::plan::{Slot, Step};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tessera_codegen::ir::{Body, Expr, Helper, Param, Stmt};
use tessera_codegen::{Dialect, hash};
use tessera_engine_settings::ByBackend;
use tessera_graph::DimName;

/// One kernel argument, in the order its source declares.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A slot's buffer, an extent of elements in.
    Buffer(Slot, Extent),
    /// Device-mapped pinned host pointer for kept `Slot`, or zero.
    KeptHost(Slot),
    /// A plain value by its bytes, any `repr(C)` type.
    Value(Vec<u8>),
    /// A u32 read from a dimension at launch.
    Size(Extent),
}

impl Arg {
    /// A `repr(C)` value by its bytes.
    pub fn value<T: Copy>(value: &T) -> Arg {
        let at = std::ptr::from_ref(value).cast::<u8>();
        Arg::Value(unsafe { std::slice::from_raw_parts(at, size_of::<T>()) }.to_vec())
    }

    pub fn count(n: usize) -> Arg {
        Arg::value(&(n as u32))
    }

    /// A count, plain when it is already known.
    pub fn size(e: Extent) -> Arg {
        match e {
            Extent::Fixed(n) => Arg::count(n),
            e => Arg::Size(e),
        }
    }

    /// A slot's buffer, `skip` elements in.
    pub fn at(slot: Slot, skip: usize) -> Arg {
        Arg::Buffer(slot, Extent::Fixed(skip))
    }

    /// The value, when it is a u32.
    pub fn u32(&self) -> Option<u32> {
        match self {
            Arg::Value(b) if b.len() == 4 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            _ => None,
        }
    }

    /// The value, when it is an f32.
    pub fn f32(&self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }
}

impl From<Extent> for Shape {
    fn from(e: Extent) -> Shape {
        Shape::line(e)
    }
}

/// A kernel struct's field as a scalar argument.
pub trait ToArg {
    fn arg(&self) -> Arg;
}

impl ToArg for usize {
    fn arg(&self) -> Arg {
        Arg::count(*self)
    }
}

impl ToArg for u32 {
    fn arg(&self) -> Arg {
        Arg::value(self)
    }
}

impl ToArg for f32 {
    fn arg(&self) -> Arg {
        Arg::value(self)
    }
}

impl ToArg for Extent {
    fn arg(&self) -> Arg {
        Arg::size(*self)
    }
}

impl Shape {
    /// A line of `e` threads, plain when known.
    pub fn line(e: Extent) -> Shape {
        match e {
            Extent::Fixed(n) => Shape::Line(n),
            e => Shape::Runtime(e),
        }
    }

    /// Groups `(across, down)`, `threads` in each.
    pub fn grid((across, down): (usize, usize), threads: usize) -> Shape {
        Shape::Grid {
            across,
            down,
            threads,
        }
    }
}

/// How threads are laid out over the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::From)]
pub enum Shape {
    /// One thread per element, in groups of 256.
    Line(usize),
    #[from(ignore)]
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
    /// A line whose length is read at launch.
    #[from(ignore)]
    Runtime(Extent),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub args: Vec<Arg>,
    pub shape: Shape,
}

impl Launch {
    /// `body`'s arguments: buffers from `step`, scalars by name.
    pub fn of(
        body: &Body,
        step: &Step,
        overwrites: bool,
        scalar: &dyn Fn(&str) -> Arg,
        shape: Shape,
    ) -> Launch {
        let mut reads = 0..;
        let mut writes = 0..;
        let mut args = Vec::with_capacity(body.params.len());
        for p in &body.params {
            args.push(match p {
                Param::Buffer { writes: true, .. } if !overwrites => {
                    step.out_at(writes.next().expect("writes"))
                }
                Param::Buffer { .. } => step.read(reads.next().expect("reads")),
                Param::Scalar { name, .. } => scalar(name),
                _ => continue,
            });
        }
        Launch { args, shape }
    }

    /// One launch: the step's reads, its output, then `counts`.
    pub fn plain(step: &Step, counts: &[usize], shape: Shape) -> Launch {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend(counts.iter().copied().map(Arg::count));
        Launch { args, shape }
    }

    /// The same launch with every runtime size filled in.
    pub fn bound(self, size: &dyn Fn(DimName) -> Option<u32>) -> Option<Launch> {
        let count = |e: &Extent| Some(e.value(size)? as u32);
        let mut args = Vec::with_capacity(self.args.len());
        for arg in self.args {
            args.push(match arg {
                Arg::Size(e) => Arg::count(count(&e)? as usize),
                Arg::Buffer(slot, e) => Arg::at(slot, e.value(size)?),
                fixed => fixed,
            });
        }
        let shape = match self.shape {
            Shape::Runtime(e) => Shape::Line(e.value(size)?),
            fixed => fixed,
        };
        Some(Launch { args, shape })
    }
}

/// Where a kernel's source comes from.
pub enum Source {
    /// Hand-written, with one text per backend.
    Fixed {
        text: ByBackend<&'static str>,
        entry: &'static str,
    },
    /// A tree, rendered per dialect or run on the host.
    Made(Body),
}

impl Source {
    /// A tree under its own entry name.
    pub fn named(mut body: Body, entry: impl Into<String>) -> Source {
        body.entry = entry.into();
        Source::Made(body)
    }

    /// The kernel's entry symbol name.
    pub fn entry(&self) -> &str {
        match self {
            Source::Fixed { entry, .. } => entry,
            Source::Made(body) => &body.entry,
        }
    }

    /// Changes whenever the kernel's text does.
    pub fn hash(&self) -> u64 {
        match self {
            Source::Fixed { text, entry } => {
                (text.0.iter().chain([entry])).fold(0, |h: u64, text| h.rotate_left(1) ^ hash(text))
            }
            Source::Made(body) => made_hash(body),
        }
    }
}

type PrefixedCache = Mutex<HashMap<(usize, usize, String), &'static str>>;

/// Returns `base` prefixed with `prefix`, cached once per pair.
pub fn prefixed(prefix: &str, base: &'static str) -> &'static str {
    if prefix.is_empty() {
        return base;
    }
    static CACHE: OnceLock<PrefixedCache> = OnceLock::new();
    let map = CACHE.get_or_init(Mutex::default);
    let key = (base.as_ptr() as usize, base.len(), prefix.to_owned());
    let mut guard = map.lock().expect("prefixed sources");
    if let Some(&cached) = guard.get(&key) {
        return cached;
    }
    let leaked: &'static str = Box::leak(format!("{prefix}{base}").into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

/// Formats `#define` lines terminated by `#line 1`.
pub fn define_prefix<'a>(
    defs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, String> {
    use std::fmt::Write;
    let mut out = String::new();
    for (name, value) in defs {
        let valid_name = !name.is_empty()
            && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            && !name.as_bytes()[0].is_ascii_digit();
        if !valid_name {
            return Err(format!("invalid define name `{name}`"));
        }
        if value.is_empty() || value.contains(['\n', '\r', '\0']) {
            return Err(format!("invalid define value for `{name}`"));
        }
        let _ = writeln!(out, "#define {name} {value}");
    }
    if !out.is_empty() {
        out.push_str("#line 1\n");
    }
    Ok(out)
}

/// Pair of independent 64-bit hashes for structural IR caching.
struct PairHash(u64, u64);

impl PairHash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325, 0x9e37_79b9_7f4a_7c15)
    }

    fn u64(&mut self, v: u64) {
        self.0 = (self.0 ^ v).wrapping_mul(0x0100_0000_01b3);
        self.1 = self.1.rotate_left(13) ^ v.wrapping_mul(0x94d0_49bb_1331_11eb);
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        for &b in s.as_bytes() {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            self.1 = self.1.rotate_left(5) ^ u64::from(b).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }

    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Int(v) => {
                self.u64(1);
                self.u64(*v as u64);
            }
            Expr::Float(v) => {
                self.u64(2);
                self.u64(v.to_bits());
            }
            Expr::Bool(v) => {
                self.u64(3);
                self.u64(u64::from(*v));
            }
            Expr::Var(s) => {
                self.u64(4);
                self.str(s);
            }
            Expr::Neg(a) => {
                self.u64(5);
                self.expr(a);
            }
            Expr::Not(a) => {
                self.u64(6);
                self.expr(a);
            }
            Expr::Bin(op, a, b) => {
                self.u64(7);
                self.u64(*op as u64);
                self.expr(a);
                self.expr(b);
            }
            Expr::Cast(ty, a) => {
                self.u64(8);
                self.u64(*ty as u64);
                self.expr(a);
            }
            Expr::Index(a, b) => {
                self.u64(9);
                self.expr(a);
                self.expr(b);
            }
            Expr::Field(a, ch) => {
                self.u64(10);
                self.u64(*ch as u64);
                self.expr(a);
            }
            Expr::Call(i, args) => {
                self.u64(11);
                self.u64(*i as u64);
                self.u64(args.len() as u64);
                for a in args {
                    self.expr(a);
                }
            }
            Expr::Select(c, a, b) => {
                self.u64(12);
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            Expr::Fn(name, ty, args) => {
                self.u64(13);
                self.str(name);
                self.u64(*ty as u64);
                self.u64(args.len() as u64);
                for a in args {
                    self.expr(a);
                }
            }
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, init } => {
                self.u64(20);
                self.str(name);
                self.expr(init);
            }
            Stmt::Array { name, ty, len } => {
                self.u64(21);
                self.str(name);
                self.u64(*ty as u64);
                self.u64(*len as u64);
            }
            Stmt::Shared { name, ty, len } => {
                self.u64(22);
                self.str(name);
                self.u64(*ty as u64);
                self.u64(*len as u64);
            }
            Stmt::AtomicShared { name, len } => {
                self.u64(23);
                self.str(name);
                self.u64(*len as u64);
            }
            Stmt::AtomicAdd { name, at, value } => {
                self.u64(24);
                self.str(name);
                self.expr(at);
                self.expr(value);
            }
            Stmt::Assign { target, op, value } => {
                self.u64(25);
                self.expr(target);
                self.u64(op.map_or(u64::MAX, |o| o as u64));
                self.expr(value);
            }
            Stmt::For {
                var,
                from,
                to,
                step,
                unroll,
                body,
            } => {
                self.u64(26);
                self.str(var);
                self.expr(from);
                self.expr(to);
                self.expr(step);
                self.u64(*unroll as u64);
                self.stmts(body);
            }
            Stmt::If { cond, then, els } => {
                self.u64(27);
                self.expr(cond);
                self.stmts(then);
                self.stmts(els);
            }
            Stmt::Return => self.u64(28),
            Stmt::Barrier => self.u64(29),
            Stmt::Expr(e) => {
                self.u64(30);
                self.expr(e);
            }
            Stmt::TileAcc { name, rows, cols } => {
                self.u64(31);
                self.str(name);
                self.u64(*rows as u64);
                self.u64(*cols as u64);
            }
            Stmt::TileHeld { name, rows, depth } => {
                self.u64(32);
                self.str(name);
                self.u64(*rows as u64);
                self.u64(*depth as u64);
            }
            Stmt::TileHold {
                held,
                a,
                a_at,
                stride,
            } => {
                self.u64(33);
                self.str(held);
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
                depth,
                parts,
                apart,
            } => {
                self.u64(34);
                self.str(acc);
                self.expr(a);
                self.expr(a_at);
                self.expr(b);
                self.expr(b_at);
                self.expr(stride);
                self.u64(*depth as u64);
                self.u64(*parts as u64);
                self.expr(apart);
            }
            Stmt::TileEach { acc, body } => {
                self.u64(35);
                self.str(acc);
                self.stmts(body);
            }
            Stmt::TileZero { acc } => {
                self.u64(36);
                self.str(acc);
            }
            Stmt::TileScale { acc, by } => {
                self.u64(37);
                self.str(acc);
                self.expr(by);
            }
        }
    }

    fn stmts(&mut self, list: &[Stmt]) {
        self.u64(list.len() as u64);
        for s in list {
            self.stmt(s);
        }
    }

    fn body(&mut self, b: &Body) {
        self.str(&b.entry);
        self.u64(b.params.len() as u64);
        for p in &b.params {
            match p {
                Param::Buffer { name, form, writes } => {
                    self.u64(40);
                    self.str(name);
                    self.str(form.name());
                    self.u64(u64::from(*writes));
                }
                Param::Scalar { name, ty } => {
                    self.u64(41);
                    self.str(name);
                    self.u64(*ty as u64);
                }
                Param::Shared { name, ty } => {
                    self.u64(42);
                    self.str(name);
                    self.u64(*ty as u64);
                }
                Param::Tiles { name } => {
                    self.u64(43);
                    self.str(name);
                }
                Param::Array { name, ty } => {
                    self.u64(44);
                    self.str(name);
                    self.u64(*ty as u64);
                }
            }
        }
        self.stmts(&b.stmts);
        if let Some(r) = &b.ret {
            self.u64(1);
            self.expr(r);
        } else {
            self.u64(0);
        }
        self.u64(b.helpers.len() as u64);
        for Helper { name, ret, body } in &b.helpers {
            self.str(name);
            self.u64(*ret as u64);
            self.body(body);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct MadeSlot {
    h1: u64,
    h2: u64,
    source_hash: u64,
    used: bool,
}

const MADE_SLOTS: usize = 1024;

/// Cached CUDA-text hash for a generated `Body`, zero heap growth.
pub(crate) fn made_hash(body: &Body) -> u64 {
    static CACHE: OnceLock<Mutex<Box<[MadeSlot; MADE_SLOTS]>>> = OnceLock::new();
    let mut fp = PairHash::new();
    fp.body(body);
    let idx = (fp.0 as usize) & (MADE_SLOTS - 1);
    let cache = CACHE.get_or_init(|| Mutex::new(Box::new([MadeSlot::default(); MADE_SLOTS])));
    {
        let table = cache.lock().expect("made hash cache");
        let slot = table[idx];
        if slot.used && slot.h1 == fp.0 && slot.h2 == fp.1 {
            return slot.source_hash;
        }
    }
    let sh = hash(&body.text().render(Dialect::Cuda));
    let mut table = cache.lock().expect("made hash cache");
    table[idx] = MadeSlot {
        h1: fp.0,
        h2: fp.1,
        source_hash: sh,
        used: true,
    };
    sh
}
