//! What a kernel declares: its arguments, shape and source.

use crate::extent::Extent;
use crate::plan::Slot;
use codegen::ir::Body;
use codegen::{Dialect, hash};
use graph::DimName;

/// One kernel argument, in the order its source declares.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A slot's buffer, this many elements in.
    Buffer(Slot, usize),
    U32(u32),
    F32(f32),
    /// One `uint2` on Metal; two arguments elsewhere.
    Pair(u32, u32),
    /// A small `repr(C)` struct, passed by value.
    Bytes(Vec<u8>),
    /// A count read from a dimension at launch.
    Size(Extent),
    /// A slot's buffer, an extent of elements in.
    Offset(Slot, Extent),
    /// Two extents as one `uint2`.
    Span(Extent, Extent),
}

impl Arg {
    /// The bytes of a `repr(C)` value.
    pub fn bytes<T: Copy>(value: &T) -> Arg {
        let at = std::ptr::from_ref(value).cast::<u8>();
        Arg::Bytes(unsafe { std::slice::from_raw_parts(at, size_of::<T>()) }.to_vec())
    }

    pub fn pair(a: usize, b: usize) -> Arg {
        Arg::Pair(a as u32, b as u32)
    }

    pub fn count(n: usize) -> Arg {
        Arg::U32(n as u32)
    }

    /// A count, plain when it is already known.
    pub fn size(e: Extent) -> Arg {
        match e {
            Extent::Fixed(n) => Arg::count(n),
            e => Arg::Size(e),
        }
    }

    /// A slot's buffer, `e` elements in.
    pub fn offset(slot: Slot, e: Extent) -> Arg {
        match e {
            Extent::Fixed(n) => Arg::Buffer(slot, n),
            e => Arg::Offset(slot, e),
        }
    }

    pub fn span(a: Extent, b: Extent) -> Arg {
        match (a, b) {
            (Extent::Fixed(a), Extent::Fixed(b)) => Arg::pair(a, b),
            (a, b) => Arg::Span(a, b),
        }
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
    /// The same launch with every runtime size filled in.
    pub fn bound(self, size: &dyn Fn(DimName) -> Option<u32>) -> Option<Launch> {
        let count = |e: &Extent| Some(e.value(size)? as u32);
        let mut args = Vec::with_capacity(self.args.len());
        for arg in self.args {
            args.push(match arg {
                Arg::Size(e) => Arg::U32(count(&e)?),
                Arg::Offset(slot, e) => Arg::Buffer(slot, e.value(size)?),
                Arg::Span(a, b) => Arg::Pair(count(&a)?, count(&b)?),
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
