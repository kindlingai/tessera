//! Lowers a graph to kernel launches over device buffers.

mod abi;
mod assign;
mod cache;
mod cost;
pub mod cpu;
mod cut;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod device;
mod extent;
pub mod form;
mod kernel;
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod metal;
mod plan;
mod recipe;
pub mod rule;
mod run;
mod schedule;
mod select;
pub mod sites;
mod tree;

pub use abi::{Arg, Launch, Shape, Source, ToArg};
pub use assign::assign;
pub use cache::Cache;
pub use codegen::kept;
pub use codegen::{Dialect, LANES, Precision, Terms, Text, float, hash, hip_tiles, terms};
pub use cost::{Cost, Device, Sample};
pub use cut::cut;
pub use extent::Extent;
pub use form::{Form, Pins};
pub use gpukit::{Limits, Pick, Platform, Settings};
pub use kernel::{Kernel, KernelRef, Needs};
pub use kernel_macro::Kernel;
pub use mem::{Buffer, DType, Tensor, View, ViewMut};
pub use plan::{Loaded, Output, Plan, Slot, Step};
pub use recipe::{Launching, Recipe};
pub use rule::{Attn, Draft, RULES, Read, Rule, linked, rules};
pub use run::{Held, Run, open, precision};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Selected, Val, Value, select, select_upto};
pub use sim::{Cluster, Fit, Link, Load, Ops, Zone, ZoneKind};
pub use tree::{Node, T, Term, Tree};

use graph::{DimName, Graph};

/// Selects kernels, orders them, then assigns buffers.
pub fn lower(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Plan, Error> {
    lower_upto(g, entry, sizes, &[], device)
}

/// The same, with `upto` dimensions read at launch.
pub fn lower_upto(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Plan, Error> {
    let picked = select_upto(g, entry, sizes, upto, device)?;
    Ok(assign(&picked, &schedule(&picked)))
}

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The graph uses something no rule covers yet.
    Unsupported(String),
    /// A dimension has no size for this launch.
    Size(String),
    Gpu(gpukit::Error),
    /// A view did not fit its buffer or its use.
    Memory(mem::Error),
    /// A runtime dimension was never bound.
    Unbound,
    /// A stored device file could not be read.
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "no rule lowers {what}"),
            Error::Size(what) => write!(f, "{what} has no size in this launch"),
            Error::Gpu(e) => write!(f, "gpu: {e}"),
            Error::Memory(e) => write!(f, "memory: {e}"),
            Error::Unbound => write!(f, "a runtime dimension has no value"),
            Error::Io(e) => write!(f, "stored samples: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<gpukit::Error> for Error {
    fn from(e: gpukit::Error) -> Self {
        Error::Gpu(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<mem::Error> for Error {
    fn from(e: mem::Error) -> Self {
        Error::Memory(e)
    }
}
