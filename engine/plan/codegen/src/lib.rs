//! Kernel text for every dialect, with its cost counted.

mod count;
mod dialect;
pub mod exec;
pub mod form;
pub mod ir;
pub mod numerics;
pub mod prelude;
mod tile;

pub use dialect::{Dialect, Text, float, hash};
pub use form::Form;
pub use kernel_macro::kernel;
pub use numerics::{Precision, Terms, terms};

/// Threads that share sums by shuffles, on every backend.
pub const LANES: usize = 32;

#[cfg(test)]
mod ir_test;
#[cfg(test)]
mod numerics_test;
