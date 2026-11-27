//! ROCm backend. HIP and hiprtc load at runtime with dlopen.

mod context;
pub mod hip;

pub use crate::launch::{KernelArgs, launch_1d};
pub use context::{Context, ContextConfig};

crate::device::backend_api!(Context);

/// The process-wide context, opened on first use.
pub fn cached_context() -> Result<&'static Context, crate::Error> {
    crate::device::cached_context()
}
