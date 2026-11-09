//! Lowers a graph to kernel launches over device buffers.

mod assign;
mod cost;
mod cut;
#[cfg(target_os = "macos")]
pub mod metal;
mod plan;
mod schedule;
mod select;

pub use assign::assign;
pub use cost::{Cost, Device, work};
pub use cut::cut;
pub use plan::{Kernel, MapStage, Output, Plan, Slot, Step};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Selected, THIN_ROWS, Val, Value, select};

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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "no rule lowers {what}"),
            Error::Size(what) => write!(f, "{what} has no size in this launch"),
            Error::Gpu(e) => write!(f, "gpu: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<gpukit::Error> for Error {
    fn from(e: gpukit::Error) -> Self {
        Error::Gpu(e)
    }
}

#[cfg(test)]
mod lower_test;
