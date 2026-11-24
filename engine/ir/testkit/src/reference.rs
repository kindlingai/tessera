use std::path::PathBuf;

/// The directory that holds every model's reference files.
#[allow(clippy::disallowed_methods, reason = "the cache lies under home")]
pub fn references() -> Option<PathBuf> {
    Some(std::env::home_dir()?.join(".cache/tessera/references"))
}

/// `model`'s reference files, when this host keeps them.
pub fn reference(model: &str) -> Option<PathBuf> {
    let dir = references()?.join(model);
    dir.is_dir().then_some(dir)
}
