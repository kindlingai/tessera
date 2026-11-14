//! GPU readiness, and the shared test helpers.

pub use testkit::{Noise, assert_oracle, cosine_f32, max_abs_diff, scratch};

/// Whether a CUDA driver loads on this host.
pub(crate) fn cuda_ready() -> bool {
    #[cfg(feature = "cuda")]
    return crate::cuda::driver::driver().is_ok();
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether a HIP runtime loads and sees a device.
pub(crate) fn rocm_ready() -> bool {
    #[cfg(feature = "rocm")]
    return crate::rocm::cached_context().is_ok();
    #[cfg(not(feature = "rocm"))]
    false
}
