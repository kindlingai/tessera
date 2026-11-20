//! Which GPU backend this build dispatches to.

pub use sensai_engine_settings::Backend;

/// The backend this build uses, if any.
pub fn available() -> Option<Backend> {
    if cfg!(target_os = "macos") {
        return Some(Backend::Metal);
    }
    // Runtimes load only at dispatch, unless both are built.
    match (cfg!(feature = "cuda"), cfg!(feature = "rocm")) {
        (true, true) if !cuda_ready() && rocm_ready() => Some(Backend::Rocm),
        (true, _) => Some(Backend::Cuda),
        (false, true) => Some(Backend::Rocm),
        (false, false) => None,
    }
}

/// Whether `backend` can run kernels here.
pub fn ready(backend: Backend) -> bool {
    match backend {
        // Hosted macOS runners lack a working Metal device.
        Backend::Metal => cfg!(target_os = "macos") && !ci(),
        Backend::Cuda => cuda_ready(),
        Backend::Rocm => rocm_ready(),
    }
}

/// This build's backend, when it can run kernels here.
pub fn picked() -> Option<Backend> {
    available().filter(|b| ready(*b))
}

/// Whether a CUDA driver loads on this host.
fn cuda_ready() -> bool {
    #[cfg(feature = "cuda")]
    return crate::cuda::driver::driver().is_ok();
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether a HIP runtime loads and sees a device.
fn rocm_ready() -> bool {
    #[cfg(feature = "rocm")]
    return crate::rocm::cached_context().is_ok();
    #[cfg(not(feature = "rocm"))]
    false
}

/// Whether a hosted CI runner is running this process.
#[allow(clippy::disallowed_methods, reason = "the one variable tests read")]
fn ci() -> bool {
    std::env::var_os("CI").is_some()
}
