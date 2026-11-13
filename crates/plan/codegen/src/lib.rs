//! Kernel text for every dialect, with its cost counted.

mod builder;
mod dialect;
pub mod ir;
pub mod prelude;
mod tile;

pub use builder::{Gen, LANES};
pub use dialect::{Dialect, Text, float, hash};
pub use kernel_macro::kernel;

#[cfg(test)]
mod gen_test;
#[cfg(test)]
mod ir_test;
