//! A kernel's declaration: its arguments, shape and source.

use crate::extent::Extent;
use crate::plan::{Slot, Step};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tessera_codegen::hash;
use tessera_codegen::ir::{Body, Param};
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
            Source::Made(body) => body.hash(),
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
