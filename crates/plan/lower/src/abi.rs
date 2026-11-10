//! What a kernel declares: its arguments, shape and source.

use crate::emit::{Dialect, Text, hash};
use crate::plan::Slot;

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
}

/// How threads are laid out over the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// One thread per element, in groups of 256.
    Line(usize),
    /// One group of 256 threads per row.
    Rows(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub args: Vec<Arg>,
    pub shape: Shape,
}

/// Where a kernel's source comes from.
pub enum Source {
    /// Hand-written, with one text per dialect.
    Fixed {
        metal: &'static str,
        cuda: &'static str,
        entry: &'static str,
    },
    /// Written once, rendered per dialect.
    Made(Text),
}

impl Source {
    /// Changes whenever the kernel's text does.
    pub fn hash(&self) -> u64 {
        match self {
            Source::Fixed { metal, cuda, entry } => [metal, cuda, entry]
                .iter()
                .fold(0, |h: u64, text| h.rotate_left(1) ^ hash(text)),
            Source::Made(text) => hash(&text.render(Dialect::Cuda)),
        }
    }
}
