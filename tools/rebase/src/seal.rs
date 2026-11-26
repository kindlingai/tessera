//! The commit that holds the files of a build.

use std::fs;
use std::path::Path;

use crate::code::packages;
use crate::git::{git, must};
use crate::replay::Plan;

/// Commits a build's files. Does not run work tree code.
pub fn seal(work: &Path, plan: &Plan, from: &Path) -> Result<String, String> {
    let built = fs::read_to_string(from.join("tree")).unwrap_or_default();
    if built.trim() != plan.tree {
        return Err("the build used another tree".to_string());
    }
    let Ok(root) = from.join("files").canonicalize() else {
        return Ok(plan.tip.clone());
    };
    for path in &plan.paths {
        let Ok(file) = root.join(path).canonicalize() else {
            continue;
        };
        if !file.starts_with(&root) || !file.is_file() {
            return Err(format!("{path} in the build output is not a plain file"));
        }
        let file = file.to_str().ok_or("the build path is not UTF-8")?;
        let blob = must(work, &["hash-object", "-w", "--no-filters", file])?;
        let listed = must(work, &["ls-files", "--stage", "--", path])?;
        let mode = listed.split(' ').next().filter(|mode| !mode.is_empty());
        let mode = mode.unwrap_or("100644");
        must(
            work,
            &["update-index", "--add", "--cacheinfo", mode, &blob, path],
        )?;
    }
    let lock = packages(&must(work, &["show", ":Cargo.lock"])?);
    if let Some((name, version)) = plan.pins.difference(&lock).next() {
        return Err(format!("the lock lost {name} {version}, set by the branch"));
    }
    if git(work, &["diff", "--cached", "--quiet"])?.ok {
        return Ok(plan.tip.clone());
    }
    let tree = must(work, &["write-tree"])?;
    let note = "Sort merged lists and refresh Cargo.lock";
    must(work, &["commit-tree", &tree, "-p", &plan.tip, "-m", note])
}
