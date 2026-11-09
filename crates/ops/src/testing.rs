//! Shared test helpers for the per-op oracle checks.

/// True when a GPU dispatch should run.
pub fn gpu_available() -> bool {
    gpukit::testing::backend_ready()
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};
