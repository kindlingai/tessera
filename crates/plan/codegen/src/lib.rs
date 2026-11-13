//! Kernel text for every dialect, with its cost counted.

mod builder;
mod dialect;

pub use builder::{Gen, LANES};
pub use dialect::{Dialect, Text, float, hash};

#[cfg(test)]
mod gen_test;
