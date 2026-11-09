//! Shared test helpers for the per-op oracle checks.

use crate::backend;

/// True when a GPU dispatch should run.
pub fn gpu_available() -> bool {
    // Hosted macOS CI runners have no usable Metal device.
    backend::available().is_some()
        && !(cfg!(target_os = "macos") && std::env::var_os("CI").is_some())
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};
