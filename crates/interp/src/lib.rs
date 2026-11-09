//! Runs a graph one op at a time.

mod cpu;
#[cfg(feature = "gpu")]
mod gpu;
mod run;
mod tensor;

pub use cpu::Cpu;
#[cfg(feature = "gpu")]
pub use gpu::{Gpu, gpu_ready};
pub use run::{Kernels, bind, run};
pub use tensor::{Data, Error, Tensor};

#[cfg(test)]
mod interp_test;
