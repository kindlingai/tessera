//! Kernel files that replace planned kernels, read again on change.

use crate::Error;
use crate::kernels::{fixed_text, group};
use crate::run::{Outcome, Reload};
use std::collections::HashMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tessera_codegen::Dialect;
use tessera_gpukit::backend::Backend;
use tessera_lower::{Source, Step};

/// A file's change time, length and inode.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Version {
    modified: SystemTime,
    len: u64,
    inode: u64,
}

impl Version {
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            modified: meta.modified().ok()?,
            len: meta.len(),
            inode: inode(&meta),
        })
    }
}

#[cfg(unix)]
fn inode(meta: &Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(meta)
}

#[cfg(not(unix))]
fn inode(_: &Metadata) -> u64 {
    0
}

/// The name a kernel file takes, before its extension.
pub(crate) fn entry(source: &Source) -> &str {
    match source {
        Source::Fixed { entry, .. } => entry,
        Source::Made(body) => &body.entry,
    }
}

/// The text a GPU runner compiles for `step` on `backend`.
pub fn text(step: &Step, backend: Backend) -> String {
    let source = step.kernel.source();
    match (&source, fixed_text(&source, backend)) {
        (Source::Made(body), _) => body.text().bounded(Dialect::of(backend), Some(group(step))),
        (_, fixed) => fixed.unwrap_or_default().0.to_string(),
    }
}

/// A file's last seen version, and its kernel and text.
struct File<K> {
    seen: Option<Version>,
    kernel: Option<(K, String)>,
}

/// Kernel files in one directory.
pub(crate) struct Overrides<K> {
    dir: PathBuf,
    dialect: Dialect,
    files: HashMap<String, File<K>>,
}

impl<K> Overrides<K> {
    pub(crate) fn new(dir: PathBuf, dialect: Dialect) -> Self {
        Self {
            dir,
            dialect,
            files: HashMap::new(),
        }
    }

    pub(crate) fn path(&self, entry: &str) -> PathBuf {
        self.dir
            .join(format!("{entry}.{}", self.dialect.extension()))
    }

    /// Reads `entry`'s file again if it changed since last checked.
    pub(crate) fn check(
        &mut self,
        entry: &str,
        planned: impl FnOnce() -> String,
        compile: impl FnOnce(&str) -> Result<K, Error>,
    ) -> Option<Reload> {
        let path = self.path(entry);
        let now = Version::of(&path);
        let file = self.files.entry(entry.to_string()).or_insert(File {
            seen: None,
            kernel: None,
        });
        if file.seen == now {
            return None;
        }
        file.seen = now;
        let text = now.and_then(|_| std::fs::read_to_string(&path).ok());
        let outcome = match text.filter(|t| *t != planned()) {
            None => {
                // Nothing ran in its place, so nothing changes.
                file.kernel.take()?;
                Outcome::Planned
            }
            Some(text) => match compile(&text) {
                Ok(kernel) => {
                    file.kernel = Some((kernel, text));
                    Outcome::Loaded
                }
                Err(e) => Outcome::Failed(e.to_string()),
            },
        };
        Some(Reload { path, outcome })
    }

    pub(crate) fn kernel(&self, entry: &str) -> Option<&K> {
        Some(&self.files.get(entry)?.kernel.as_ref()?.0)
    }

    /// The text of `entry`'s loaded file.
    pub(crate) fn text(&self, entry: &str) -> Option<&str> {
        Some(&self.files.get(entry)?.kernel.as_ref()?.1)
    }

    /// Writes `text` as `entry`'s file, unless one exists.
    pub(crate) fn write(&self, entry: &str, text: &str) -> Result<Option<PathBuf>, Error> {
        let path = self.path(entry);
        if path.exists() {
            return Ok(None);
        }
        let failed = |e: std::io::Error| Error::KernelFile(path.clone(), e);
        std::fs::create_dir_all(&self.dir).map_err(failed)?;
        std::fs::write(&path, text).map_err(failed)?;
        Ok(Some(path))
    }
}
