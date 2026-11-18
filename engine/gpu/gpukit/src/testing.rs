//! Whether each runtime, and the CI runner, is here.

use crate::backend::{Backend, picked};
use std::sync::LazyLock;

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

/// Whether a hosted CI runner is running this process.
#[allow(clippy::disallowed_methods, reason = "the one variable tests read")]
pub(crate) fn ci() -> bool {
    std::env::var_os("CI").is_some()
}

/// The backend `TESSERA_REQUIRE_GPU` names. GPU boxes set it.
#[allow(clippy::disallowed_methods, reason = "a test setting of the GPU boxes")]
static REQUIRED: LazyLock<Option<Backend>> = LazyLock::new(|| {
    let name = std::env::var("TESSERA_REQUIRE_GPU")
        .ok()
        .filter(|n| !n.is_empty())?;
    let all = [Backend::Metal, Backend::Cuda, Backend::Rocm];
    let found = all.into_iter().find(|b| b.to_string() == name);
    assert!(
        found.is_some(),
        "TESSERA_REQUIRE_GPU={name} names no backend"
    );
    found
});

/// Panics when this build cannot use the `TESSERA_REQUIRE_GPU` backend.
pub fn require_gpu() {
    let Some(want) = *REQUIRED else { return };
    let got = picked();
    assert_eq!(
        got,
        Some(want),
        "TESSERA_REQUIRE_GPU={want}, but this build picks {got:?}: build with --features sensai-run/{want}"
    );
}
