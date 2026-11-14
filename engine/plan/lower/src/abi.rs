//! What a kernel declares: its arguments, shape and source.

use crate::extent::Extent;
use crate::plan::{Slot, Step};
use codegen::ir::{Body, Param};
use codegen::{Dialect, hash};
use graph::DimName;

/// One kernel argument, in the order its source declares.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A slot's buffer, an extent of elements in.
    Buffer(Slot, Extent),
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

impl From<usize> for Shape {
    fn from(n: usize) -> Shape {
        Shape::Line(n)
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
}

/// How threads are laid out over the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One thread per element, in groups of 256.
    Line(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
    /// A line whose length is read at launch.
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
        let mut args = Vec::with_capacity(body.params.len());
        for p in &body.params {
            args.push(match p {
                Param::Buffer { writes: true, .. } if !overwrites => step.out(),
                Param::Buffer { .. } => step.read(reads.next().expect("reads")),
                Param::Scalar { name, .. } => scalar(name),
                _ => continue,
            });
        }
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
    /// Hand-written, with one text per dialect.
    Fixed {
        metal: &'static str,
        cuda: &'static str,
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

    /// Changes whenever the kernel's text does.
    pub fn hash(&self) -> u64 {
        match self {
            Source::Fixed { metal, cuda, entry } => [metal, cuda, entry]
                .iter()
                .fold(0, |h: u64, text| h.rotate_left(1) ^ hash(text)),
            Source::Made(body) => hash(&body.text().render(Dialect::Cuda)),
        }
    }
}
