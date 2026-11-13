//! Which GPU backend this build dispatches to.

/// A backend that can run kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
}

/// The backend this build uses, if any.
pub fn available() -> Option<Backend> {
    if cfg!(target_os = "macos") {
        return Some(Backend::Metal);
    }
    // Runtimes load only at dispatch, unless both are built.
    match (cfg!(feature = "cuda"), cfg!(feature = "rocm")) {
        (true, true) if !crate::testing::cuda_ready() && crate::testing::rocm_ready() => {
            Some(Backend::Rocm)
        }
        (true, _) => Some(Backend::Cuda),
        (false, true) => Some(Backend::Rocm),
        (false, false) => None,
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Metal => write!(f, "metal"),
            Backend::Cuda => write!(f, "cuda"),
            Backend::Rocm => write!(f, "rocm"),
        }
    }
}
