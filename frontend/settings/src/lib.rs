//! Settings read once from the environment, passed down.

use gpukit::backend::Backend;
use gpukit::{Pick, Settings};
use std::path::PathBuf;

/// The only reader of the process environment.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    use std::env::{home_dir, var, var_os};
    // `CI` picks the host: hosted macOS runners lack Metal.
    let host = var("TESSERA_BACKEND").as_deref() == Ok("cpu")
        || (cfg!(target_os = "macos") && var_os("CI").is_some());
    // TESSERA_LANES, TESSERA_COSTS, TESSERA_QWEN3, TESSERA_EG2, TESSERA_SPACY,
    // TESSERA_NAMES and TESSERA_EMULATE fill fields.
    let lanes = var("TESSERA_LANES").ok().and_then(|v| v.parse().ok());
    let costs = var_os("TESSERA_COSTS")
        .map(PathBuf::from)
        .or_else(|| Some(home_dir()?.join(".cache/tessera/costs")));
    Settings {
        backend: if host { Pick::Host } else { Pick::Auto },
        host_lanes: lanes.unwrap_or(32),
        costs,
        qwen3: var_os("TESSERA_QWEN3").map(PathBuf::from),
        eg2: var_os("TESSERA_EG2").map(PathBuf::from),
        spacy: var_os("TESSERA_SPACY").map(PathBuf::from),
        names: var_os("TESSERA_NAMES").map(PathBuf::from),
        emulate: match var("TESSERA_EMULATE").as_deref() {
            Ok("metal") => Some(Backend::Metal),
            Ok("cuda") => Some(Backend::Cuda),
            Ok("rocm") => Some(Backend::Rocm),
            _ => None,
        },
    }
}
