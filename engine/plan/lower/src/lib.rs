//! Lowers a graph to kernel launches over device buffers.

mod abi;
mod assign;
mod body;
mod cache;
mod cost;
mod cut;
mod error;
mod extent;
pub mod form;
mod kernel;
pub mod kernels;
mod plan;
mod recipe;
pub mod rule;
mod schedule;
mod select;
pub mod sites;
mod tree;

pub use abi::{Arg, Launch, Shape, Source, ToArg};
pub use assign::assign;
pub use body::{GUESS, small, source};
pub use cache::Cache;
pub use cost::Device;
pub use cut::cut;
pub use error::{Size, Unsupported};
pub use extent::Extent;
pub use form::{Pins, Weight};
pub use kernel::{Kernel, KernelRef, Needs, UNPROVEN};
pub use plan::{Loaded, Output, Plan, Share, Slot, Step};
pub use recipe::{Launching, Recipe};
pub use rule::{Attn, Draft, Propose, RULES, Read, Rule, linked, rules};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Selected, Val, Value, select};
pub use sensai_codegen::kept;
pub use sensai_codegen::{Dialect, Terms, hip_tiles, terms};
pub use sensai_engine_settings::{Pick, Settings};
pub use sensai_kernel_macro::Kernel;
pub use sensai_mem::{Buffer, DType, Tensor, View};
pub use sensai_platform::{Limits, Platform};
pub use sensai_sim::{Cluster, Fit, Link, Zone, ZoneKind};
pub use sensai_work::Load;
pub use tree::{Node, T, Term, Tree};

use sensai_graph::{DimName, Graph};

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
pub(crate) fn lower_upto(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Plan, Error> {
    let picked = select::select_upto(g, entry, sizes, upto, device)?;
    Ok(assign(&picked, &schedule(&picked)))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The graph uses something without a rule yet.
    #[error("{0} has no lowering rule")]
    Unsupported(#[from] Unsupported),
    /// A dimension has no size for this launch.
    #[error("{0} has no size in this launch")]
    Size(#[from] Size),
    /// A view did not fit its buffer or its use.
    #[error("memory: {0}")]
    Memory(#[from] sensai_mem::Error),
    /// A runtime dimension was never bound.
    #[error("a runtime dimension has no value")]
    Unbound,
    /// A stored device file could not be read.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
