//! Rebases a head on a base in a work tree.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::code::{Pin, packages};
use crate::git::{git, must};
use crate::hunk::Refusal;
use crate::lists::resolve;

pub struct Plan {
    pub tip: String,
    pub tree: String,
    /// Files with resolved conflicts, and the lock files.
    pub paths: BTreeSet<String>,
    /// Locked versions that the branch set.
    pub pins: BTreeSet<Pin>,
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
    // A rebase drops merge commits and their changes.
    let span = format!("{base}..{head}");
    if !must(here, &["rev-list", "--merges", "--max-count=1", &span])?.is_empty() {
        return Ok(Outcome::Refused(
            "the branch has a merge commit".to_string(),
        ));
    }
    let pins = match pins(&base, &head)? {
        Ok(pins) => pins,
        Err(why) => return Ok(Outcome::Refused(why)),
    };
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
        pins,
    }))
}

/// The locked versions the branch set, or a refusal.
fn pins(base: &str, head: &str) -> Result<Result<BTreeSet<Pin>, String>, String> {
    let here = Path::new(".");
    let fork = must(here, &["merge-base", base, head])?;
    let mut locks = Vec::new();
    for commit in [fork.as_str(), base, head] {
        let lock = git(here, &["show", &format!("{commit}:{LOCK}")])?;
        locks.push(packages(&lock.stdout));
    }
    let (fork, base, head) = (&locks[0], &locks[1], &locks[2]);
    let plain = |text: &str| {
        text.chars()
            .all(|c| c.is_ascii_alphanumeric() || "_.+-".contains(c))
    };
    let mut pins = BTreeSet::new();
    for pin in head.difference(fork) {
        let (name, version) = pin;
        if !plain(name) || !plain(version) || name.starts_with('-') {
            return Ok(Err(
                "the lock file has an entry that is not plain".to_string()
            ));
        }
        let versions = |lock: &BTreeSet<Pin>| {
            let named = lock.iter().filter(|(other, _)| other == name);
            named
                .map(|(_, version)| version.clone())
                .collect::<Vec<_>>()
        };
        if !base.contains(pin) && versions(fork) != versions(base) {
            let why = format!("the branch and the base both change the locked version of `{name}`");
            return Ok(Err(why));
        }
        pins.insert(pin.clone());
    }
    Ok(Ok(pins))
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
