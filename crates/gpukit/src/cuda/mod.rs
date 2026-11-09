//! CUDA backend; libcuda loads at runtime, so builds need nothing.

mod buffer;
mod cached;
mod context;
mod dispatch;
pub mod driver;
pub mod nvrtc;

pub use buffer::{BufferPool, DeviceBuffer};
pub use cached::{cached_context, cached_kernel, cached_source_kernel, pod_bytes};
pub use context::{Context, ContextConfig, Kernel, Module};
pub use dispatch::{
    KernelArgs, THREADS_PER_BLOCK, div_up, launch_1d, launch_1d_ranged, launch_grid, launch_rows,
};
