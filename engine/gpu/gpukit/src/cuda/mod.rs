//! CUDA backend. libcuda and NVRTC load at runtime with dlopen.

mod context;
#[cfg(test)]
mod context_test;
pub mod driver;

pub use crate::launch::{KernelArgs, launch_1d};
pub use context::{Context, ContextConfig, Kernel, compile_ptx_with, nvrtc_version};

crate::device::backend_api!(Context);
