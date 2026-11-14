//! Kernel text for every dialect, with its cost counted.

pub mod block;
mod count;
mod dialect;
pub mod exec;
pub mod form;
pub mod inline;
pub mod ir;
pub mod kept;
pub mod numerics;
pub mod prelude;
mod tile;
pub mod trellis;

pub use dialect::{Dialect, Text, float, hash};
pub use form::Form;
pub use kept::Layout as Kept;
pub use kernel_macro::kernel;
pub use numerics::{Precision, Terms, terms};
pub use tile::hip as hip_tiles;

/// Threads that share sums by shuffles, on every backend.
pub const LANES: usize = 32;

#[cfg(test)]
mod block_test;
#[cfg(test)]
mod ir_test;
#[cfg(test)]
mod numerics_test;
