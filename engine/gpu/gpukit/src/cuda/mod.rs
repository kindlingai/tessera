//! CUDA backend. libcuda and NVRTC load at runtime with dlopen.

mod context;
pub mod driver;

pub use crate::launch::{
    KernelArgs, Launch, THREADS_PER_BLOCK, div_up, launch_1d, launch_grid, pod_bytes,
};
pub use context::{Context, ContextConfig, Graph, Kernel, Module};

crate::device::backend_api!(Context, Kernel);
