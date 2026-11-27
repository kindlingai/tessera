//! Lowers a graph to kernel launches over device buffers.

#![allow(
    clippy::len_without_is_empty,
    reason = "the workspace dead code check removes unused is_empty"
)]

mod abi;
mod assign;
mod body;
mod cache;
mod cost;
mod cut;
mod error;
mod extent;
pub mod form;
pub mod hazard;
mod kernel;
pub mod kernels;
mod plan;
mod recipe;
pub mod rule;
mod schedule;
mod select;
pub mod sites;
mod tree;

pub use abi::{Arg, Launch, Shape, Source, ToArg, define_prefix, prefixed};
pub use assign::assign;
pub use body::{GUESS, small, source};
pub use cache::Cache;
pub use cost::{ASKERS, Ask, Device, kernel_identity};
pub use cut::{cut, cut_region};
pub use error::{Size, Unsupported};
pub use extent::Extent;
pub use form::{Filed, Pins, Site, Weight};
pub use kernel::{COMMIT, Collective, Keeping, Kernel, KernelRef, Needs, PEEK, Resident, UNPROVEN};
pub use plan::{KeptInfo, Loaded, Output, Plan, Share, Slot, Step};
pub use recipe::{
    Compute, Dtype, KNOBS, Knob, Knobs, Launching, Recipe, Scoped, Wires, find_knob, labels,
    register_knob,
};
pub use rule::{Attn, Draft, Propose, Read, Rule, linked, rules};
pub use schedule::{peak, schedule};
pub use select::{Home, Op, Report, Selected, Val, Value, select};
pub use tessera_codegen::kept;
pub use tessera_codegen::{Dialect, Terms, hip_tiles, terms};
pub use tessera_engine_settings::{Backend, ByBackend, Pick, Reads, Settings};
pub use tessera_kernel_macro::Kernel;
pub use tessera_mem::{Buffer, DType, Tensor, View};
pub use tessera_platform::{Limits, Platform};
pub use tessera_sim::{Cluster, Fit, Link, Zone, ZoneKind};
pub use tessera_work::{Load, Ops};
pub use tree::{RING_CHUNK, T, Term, Tree, Wants, sliding_ring};

use tessera_graph::{DimName, Graph};

/// Selects kernels, orders them, then assigns buffers.
pub fn lower(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Plan, Error> {
    lower_upto(g, entry, Wants::ALL, sizes, &[], device)
}

/// The same, with `upto` dimensions read at launch.
pub(crate) fn lower_upto(
    g: &Graph,
    entry: usize,
    wants: Wants,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<Plan, Error> {
    let picked = select::select_upto(g, entry, wants, sizes, upto, device)?;
    Ok(assign(&picked, &schedule(&picked)))
}

/// Lowers `entry` and returns a fast-path diagnostic report.
pub fn explain(
    g: &Graph,
    entry: usize,
    sizes: &[(DimName, u64)],
    device: &Device,
) -> Result<Report, Error> {
    Ok(explain_upto(g, entry, Wants::ALL, sizes, &[], device)?.1)
}

/// The same, with `upto` dimensions read at launch.
pub(crate) fn explain_upto(
    g: &Graph,
    entry: usize,
    wants: Wants,
    sizes: &[(DimName, u64)],
    upto: &[DimName],
    device: &Device,
) -> Result<(Plan, Report), Error> {
    let (picked, report) = select::select_report(g, entry, wants, sizes, upto, device)?;
    let plan = assign(&picked, &schedule(&picked));
    Ok((plan, report))
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
    Memory(#[from] tessera_mem::Error),
    /// A runtime dimension was never bound.
    #[error("a runtime dimension has no value")]
    Unbound,
    /// A stored device file could not be read.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A required favored rule did not cover any step.
    #[error("favored rule `{0}` did not cover any step in the plan")]
    FavoredUnused(String),
}
