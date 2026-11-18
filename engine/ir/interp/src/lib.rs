//! Runs a graph one op at a time.

mod cpu;
#[cfg(test)]
mod cpu_test;
mod fold;
#[cfg(test)]
mod fold_test;
#[cfg(test)]
mod interp_test;
mod round;
#[cfg(test)]
mod round_test;
mod run;
#[cfg(test)]
mod run_test;
mod tensor;

pub use fold::fold;
pub use run::{Store, bind, run, run_kept, run_parts, run_ranks};
pub use sensai_mem::Tensor;
pub use tensor::Error;
