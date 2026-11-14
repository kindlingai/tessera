use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A file tests may write, under the workspace's target directory.
pub fn scratch(file: &str) -> PathBuf {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../target/scratch"
    ));
    std::fs::create_dir_all(&dir).expect("a target directory");
    dir.join(file)
}

/// A new scratch file ending in `name`, holding `bytes`.
pub fn stage(name: &str, bytes: &[u8]) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = scratch(&format!("{}-{id}-{name}", std::process::id()));
    std::fs::write(&path, bytes).expect("a scratch file");
    path
}
