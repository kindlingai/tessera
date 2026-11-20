//! Kernel files that replace planned kernels, read again on change.

use crate::Error;
use crate::run::{Outcome, Reload};
use sensai_codegen::Dialect;
use sensai_lower::Source;
use std::collections::HashMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

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

/// A planned kernel's text.
pub(crate) fn text(source: &Source, dialect: Dialect) -> String {
    match source {
        Source::Fixed { metal, .. } if dialect == Dialect::Metal => metal.to_string(),
        Source::Fixed { cuda, .. } => cuda.to_string(),
        Source::Made(body) => body.text().render(dialect),
    }
}

/// A file's last seen version, and its compiled kernel.
struct File<K> {
    seen: Option<Version>,
    kernel: Option<K>,
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
        let extension = match self.dialect {
            Dialect::Metal => "metal",
            _ => "cu",
        };
        self.dir.join(format!("{entry}.{extension}"))
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
                    file.kernel = Some(kernel);
                    Outcome::Loaded
                }
                Err(e) => Outcome::Failed(e.to_string()),
            },
        };
        Some(Reload { path, outcome })
    }

    pub(crate) fn kernel(&self, entry: &str) -> Option<&K> {
        self.files.get(entry)?.kernel.as_ref()
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
