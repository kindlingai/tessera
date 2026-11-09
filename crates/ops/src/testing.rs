//! Shared test helpers for the per-op oracle checks.

use crate::backend::{self, Backend};

/// True when a GPU dispatch should run.
pub fn gpu_available() -> bool {
    // Hosted macOS CI runners have no usable Metal device.
    let ci_mac = cfg!(target_os = "macos") && std::env::var_os("CI").is_some();
    let ready = match backend::available() {
        Some(Backend::Cuda) => gpukit::testing::cuda_ready(),
        Some(Backend::Metal) => true,
        None => false,
    };
    ready && !ci_mac
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};
