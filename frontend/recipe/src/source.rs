//! A recipe's place: a directory, or one git commit.

use crate::{Error, io, pinned};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, PartialEq)]
pub enum Source {
    Dir(PathBuf),
    /// A clone URL and a full commit hash.
    Git {
        url: String,
        commit: String,
    },
}

impl Source {
    /// `git+<url>#<commit>`, else a local directory.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let refuse = |why| Err(Error::Source(text.into(), why));
        let Some(rest) = text.strip_prefix("git+") else {
            if text.contains("://") {
                return refuse("a remote recipe starts with git+");
            }
            return Ok(Source::Dir(text.into()));
        };
        let Some((url, commit)) = rest.split_once('#') else {
            return refuse("a git recipe ends with #<commit>");
        };
        if !pinned(commit) {
            return refuse("pin a full 40-digit commit");
        }
        let (url, commit) = (url.into(), commit.to_ascii_lowercase());
        Ok(Source::Git { url, commit })
    }

    /// The local directory, cloned into `cache` first when needed.
    pub fn resolve(&self, cache: Option<&Path>) -> Result<PathBuf, Error> {
        match self {
            Source::Dir(dir) => Ok(dir.clone()),
            Source::Git { url, commit } => {
                let dir = cache.ok_or(Error::NoCache)?.join("git").join(commit);
                if !dir.exists() {
                    clone(url, commit, &dir)?;
                }
                Ok(dir)
            }
        }
    }
}

/// Fetches one commit beside `dir`, then renames the fetched folder.
fn clone(url: &str, commit: &str, dir: &Path) -> Result<(), Error> {
    let part = dir.with_extension("part");
    let _ = std::fs::remove_dir_all(&part);
    std::fs::create_dir_all(&part).map_err(io(&part))?;
    let steps: [&[&str]; 3] = [
        &["init", "-q"],
        &["fetch", "-q", "--depth", "1", url, commit],
        &["checkout", "-q", "--detach", "FETCH_HEAD"],
    ];
    for args in steps {
        run(&part, args)?;
    }
    std::fs::rename(&part, dir).map_err(io(dir))
}

/// Runs git in `dir`, keeping its error text on failure.
fn run(dir: &Path, args: &[&str]) -> Result<(), Error> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output();
    let out = out.map_err(io(dir))?;
    if out.status.success() {
        return Ok(());
    }
    let said = String::from_utf8_lossy(&out.stderr).trim().to_string();
    Err(Error::Command(format!("git {}", args.join(" ")), said))
}
