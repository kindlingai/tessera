//! Runs a graph one op at a time.

mod cpu;
mod round;
mod run;
mod tensor;

pub use cpu::Cpu;
pub use run::{Kernels, bind, run};
pub use tensor::{Data, Error, Tensor};

#[cfg(test)]
mod interp_test;
#[cfg(test)]
mod round_test;
