//! Metal backend: a device on the shared traits.

mod context;
#[cfg(test)]
mod context_test;
mod expand;
#[cfg(test)]
mod expand_test;
mod graph;
mod pipeline_cache;

pub use context::Context;
pub use expand::expand;
pub use graph::Graph;
