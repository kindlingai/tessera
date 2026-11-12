//! Runs a graph one op at a time.

mod cpu;
mod round;
mod run;
mod tensor;

pub use mem::Tensor;
pub use run::{Store, bind, run, run_kept};
pub use tensor::Error;

#[cfg(test)]
mod interp_test;
#[cfg(test)]
mod round_test;
