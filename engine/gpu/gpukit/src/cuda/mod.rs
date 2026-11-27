//! CUDA backend. libcuda and NVRTC load at runtime with dlopen.

mod context;
#[cfg(test)]
mod context_test;
pub mod driver;

pub use crate::launch::{KernelArgs, Launch, launch_1d, launch_grid};
pub use context::{Context, ContextConfig, Graph, Kernel, Module, compile_ptx_with, nvrtc_version};

crate::device::backend_api!(Context, Kernel);
