//! Rebases a head on a base in a work tree.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::git::{git, must};
use crate::hunk::Refusal;
use crate::lists::resolve;

pub struct Plan {
    pub tip: String,
    pub tree: String,
    /// Files with resolved conflicts, and the lock files.
    pub paths: BTreeSet<String>,
}

pub enum Outcome {
    Done(Plan),
    Refused(String),
}

const LOCK: &str = "Cargo.lock";

/// Replays the commits of `head` on `base` in `work`.
pub fn replay(base: &str, head: &str, work: &Path) -> Result<Outcome, String> {
    let here = Path::new(".");
    let base = must(
        here,
        &["rev-parse", "--verify", &format!("{base}^{{commit}}")],
    )?;
    let head = must(
        here,
        &["rev-parse", "--verify", &format!("{head}^{{commit}}")],
    )?;
    // An attributes file can select another merge for a path.
    let changed = must(here, &["diff", "--name-only", &format!("{base}...{head}")])?;
    if changed.lines().any(|path| path.ends_with(".gitattributes")) {
        return Ok(Outcome::Refused(
            "the branch changes `.gitattributes`".to_string(),
        ));
    }
    let dir = work.to_str().ok_or("the work path is not UTF-8")?;
    must(
        here,
        &["worktree", "add", "--quiet", "--detach", dir, &head],
    )?;

    let mut paths = BTreeSet::from([LOCK.to_string()]);
    let mut step = git(work, &["rebase", &base])?;
    loop {
        let state = must(
            work,
            &[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "rebase-merge",
            ],
        )?;
        if !Path::new(&state).exists() {
            if step.ok {
                break;
            }
            return Err(format!("git rebase: {}", step.stderr.trim()));
        }
        let files = unmerged(work)?;
        if files.is_empty() {
            return Err(format!("git rebase stopped: {}", step.stderr.trim()));
        }
        let commit = must(work, &["rev-parse", "--short", "REBASE_HEAD"])?;
        for (path, modes) in files {
            if let Some(Refusal { line, why }) = settle(work, &path, &modes)? {
                git(work, &["rebase", "--abort"])?;
                // A comment shows this text. The branch names the path.
                let plain = |c: char| c.is_ascii_alphanumeric() || "_./-".contains(c);
                let path = path.replace(|c| !plain(c), "?");
                let text = format!("`{path}:{line}`: {why}, in commit {commit}");
                return Ok(Outcome::Refused(text));
            }
            paths.insert(path);
        }
        let empty = git(work, &["diff", "--cached", "--quiet"])?.ok;
        step = git(
            work,
            &["rebase", if empty { "--skip" } else { "--continue" }],
        )?;
    }
    let tip = must(work, &["rev-parse", "HEAD"])?;
    if tip == base {
        // A push of the base would close the pull request.
        let text = "the base branch already holds each commit";
        return Ok(Outcome::Refused(text.to_string()));
    }
    Ok(Outcome::Done(Plan {
        tip,
        tree: must(work, &["rev-parse", "HEAD^{tree}"])?,
        paths,
    }))
}

/// Each unmerged path with its stage file modes.
fn unmerged(work: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let listed = git(work, &["ls-files", "--unmerged", "-z"])?;
    let mut files: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for entry in listed.stdout.split('\0') {
        let Some((stage, path)) = entry.split_once('\t') else {
            continue;
        };
        let mode = stage.split(' ').next().unwrap_or_default();
        files
            .entry(path.to_string())
            .or_default()
            .push(mode.to_string());
    }
    Ok(files)
}

/// Resolves one unmerged file and stages it.
fn settle(work: &Path, path: &str, modes: &[String]) -> Result<Option<Refusal>, String> {
    let refuse = |why| Ok(Some(Refusal { line: 1, why }));
    if path.rsplit('/').next() == Some(LOCK) {
        // The base side stays. The build adds the missing packages.
        if !git(work, &["checkout", "--ours", "--", path])?.ok {
            return refuse("the lock file is missing on the base branch");
        }
    } else {
        let regular = modes
            .iter()
            .all(|mode| mode == "100644" || mode == "100755");
        if modes.len() != 3 || !regular {
            return refuse("a side adds, removes or renames the file");
        }
        let file = work.join(path);
        let Ok(text) = fs::read_to_string(&file) else {
            return refuse("the file is not text");
        };
        match resolve(path, &text) {
            Ok(text) => fs::write(&file, text).map_err(|e| format!("{path}: {e}"))?,
            Err(refusal) => return Ok(Some(refusal)),
        }
    }
    must(work, &["add", "--", path])?;
    Ok(None)
}
