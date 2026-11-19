//! The GPU backend a test box must use.

use crate::backend::{Backend, picked};
use std::sync::LazyLock;

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
