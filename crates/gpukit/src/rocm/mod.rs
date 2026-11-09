//! ROCm backend. HIP and hiprtc load at runtime with dlopen.

mod buffer;
mod cached;
mod context;
pub mod hip;
pub mod hiprtc;

pub use crate::launch::{
    KernelArgs, Launch, THREADS_PER_BLOCK, div_up, launch_1d, launch_1d_ranged, launch_grid,
    launch_rows, pod_bytes,
};
pub use buffer::{BufferPool, DeviceBuffer};
pub use cached::{cached_context, cached_kernel, cached_source_kernel};
pub use context::{Context, ContextConfig, Kernel, Module};
