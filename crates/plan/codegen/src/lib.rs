//! Kernel text for every dialect, with its cost counted.

mod count;
mod dialect;
pub mod ir;
pub mod prelude;
mod tile;

pub use dialect::{Dialect, Text, float, hash};
pub use kernel_macro::kernel;

/// Threads that share sums by shuffles, on every backend.
pub const LANES: usize = 32;

#[cfg(test)]
mod ir_test;
