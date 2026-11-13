//! Lowers a graph to kernel launches over device buffers.

mod abi;
mod assign;
mod cache;
mod cost;
mod cut;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod device;
mod form;
mod kernel;
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod metal;
mod plan;
pub mod rule;
mod run;
mod schedule;
mod select;
mod tree;

pub use abi::{Arg, Launch, Shape, Source};
pub use assign::assign;
pub use cache::Cache;
pub use codegen::{Dialect, Gen, LANES, Text, float, hash};
pub use cost::{Cost, Device, Sample};
pub use cut::cut;
pub use form::{Pins, Stored};
pub use kernel::{Kernel, KernelRef};
pub use mem::{Buffer, DType, Tensor, View, ViewMut, round};
pub use plan::{Loaded, Output, Plan, Slot, Step};
pub use rule::{Attn, Draft, RULES, Read, Rule, Site, rules};
pub use run::{Held, Run, open};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Selected, Val, Value, select};
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
    let picked = select(g, entry, sizes, device)?;
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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "no rule lowers {what}"),
            Error::Size(what) => write!(f, "{what} has no size in this launch"),
            Error::Gpu(e) => write!(f, "gpu: {e}"),
            Error::Memory(e) => write!(f, "memory: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<gpukit::Error> for Error {
    fn from(e: gpukit::Error) -> Self {
        Error::Gpu(e)
    }
}

impl From<mem::Error> for Error {
    fn from(e: mem::Error) -> Self {
        Error::Memory(e)
    }
}
