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
pub(crate) struct Version {
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

/// A file change detected on disk before committing state.
pub(crate) enum Pending {
    /// File matches planned text with no active override.
    Unchanged {
        entry: String,
        seen: Option<Version>,
    },
    /// Active override was deleted or reverted to planned text.
    Revert {
        entry: String,
        path: PathBuf,
        seen: Option<Version>,
    },
    /// File changed to text differing from planned.
    Compile {
        entry: String,
        path: PathBuf,
        seen: Option<Version>,
        text: String,
    },
}

/// A compiled or reverted override change waiting to commit.
pub(crate) struct Ready<K> {
    entry: String,
    seen: Option<Version>,
    kernel: Option<Option<(K, String)>>,
    reload: Option<Reload>,
}

impl<K> Ready<K> {
    pub(crate) fn unchanged(entry: String, seen: Option<Version>) -> Self {
        Self {
            entry,
            seen,
            kernel: None,
            reload: None,
        }
    }

    pub(crate) fn reverted(entry: String, path: PathBuf, seen: Option<Version>) -> Self {
        Self {
            entry,
            seen,
            kernel: Some(None),
            reload: Some(Reload {
                path,
                outcome: Outcome::Planned,
            }),
        }
    }

    pub(crate) fn loaded(
        entry: String,
        path: PathBuf,
        seen: Option<Version>,
        kernel: K,
        text: String,
    ) -> Self {
        Self {
            entry,
            seen,
            kernel: Some(Some((kernel, text))),
            reload: Some(Reload {
                path,
                outcome: Outcome::Loaded,
            }),
        }
    }

    pub(crate) fn failed(entry: String, path: PathBuf, seen: Option<Version>, why: String) -> Self {
        Self {
            entry,
            seen,
            kernel: None,
            reload: Some(Reload {
                path,
                outcome: Outcome::Failed(why),
            }),
        }
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

    /// Inspects `entry`'s file without mutating recorded state.
    pub(crate) fn scan(&self, entry: &str, planned: impl FnOnce() -> String) -> Option<Pending> {
        let path = self.path(entry);
        let now = Version::of(&path);
        let prev = self.files.get(entry);
        if prev.and_then(|f| f.seen) == now {
            return None;
        }
        let text = now.and_then(|_| std::fs::read_to_string(&path).ok());
        Some(match text.filter(|t| *t != planned()) {
            None if prev.is_some_and(|f| f.kernel.is_some()) => Pending::Revert {
                entry: entry.to_string(),
                path,
                seen: now,
            },
            None => Pending::Unchanged {
                entry: entry.to_string(),
                seen: now,
            },
            Some(text) => Pending::Compile {
                entry: entry.to_string(),
                path,
                seen: now,
                text,
            },
        })
    }

    /// Commits one prepared change into recorded state.
    pub(crate) fn commit(&mut self, ready: Ready<K>) -> Option<Reload> {
        let file = self.files.entry(ready.entry).or_insert(File {
            seen: None,
            kernel: None,
        });
        file.seen = ready.seen;
        if let Some(kernel) = ready.kernel {
            file.kernel = kernel;
        }
        ready.reload
    }

    /// Reads `entry`'s file again if it changed since last checked.
    #[cfg(test)]
    pub(crate) fn check(
        &mut self,
        entry: &str,
        planned: impl FnOnce() -> String,
        compile: impl FnOnce(&str) -> Result<K, Error>,
    ) -> Option<Reload> {
        let ready = match self.scan(entry, planned)? {
            Pending::Unchanged { entry, seen } => Ready::unchanged(entry, seen),
            Pending::Revert { entry, path, seen } => Ready::reverted(entry, path, seen),
            Pending::Compile {
                entry,
                path,
                seen,
                text,
            } => match compile(&text) {
                Ok(kernel) => Ready::loaded(entry, path, seen, kernel, text),
                Err(e) => Ready::failed(entry, path, seen, e.to_string()),
            },
        };
        self.commit(ready)
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
