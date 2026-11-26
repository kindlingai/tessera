//! The files a build changes, and their commit.

use std::fs;
use std::path::Path;

use crate::git::{cargo, git, must};
use crate::replay::Plan;

/// Formats and builds, then copies changed files to `out`.
pub fn build(work: &Path, plan: &Plan, out: &Path, features: &[&str]) -> Result<(), String> {
    cargo(work, &["fmt", "--all"])?;
    for tool in ["check", "clippy"] {
        let mut args = vec![tool, "--workspace", "--all-targets"];
        args.extend_from_slice(features);
        cargo(work, &args)?;
    }
    let status = git(work, &["status", "--porcelain", "--no-renames", "-z"])?;
    let files = out.join("files");
    for entry in status.stdout.split('\0').filter(|entry| !entry.is_empty()) {
        let path = entry.get(3..).unwrap_or_default();
        if !plan.paths.contains(path) {
            return Err(format!(
                "the build changed {path}, a file without a conflict"
            ));
        }
        let to = files.join(path);
        let copied = to
            .parent()
            .map(fs::create_dir_all)
            .map(|made| made.and_then(|()| fs::copy(work.join(path), &to)));
        if !matches!(copied, Some(Ok(_))) {
            return Err(format!("cannot copy {path} to {}", to.display()));
        }
    }
    fs::create_dir_all(out)
        .and_then(|()| fs::write(out.join("tree"), &plan.tree))
        .map_err(|e| format!("{}: {e}", out.display()))
}

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
    if git(work, &["diff", "--cached", "--quiet"])?.ok {
        return Ok(plan.tip.clone());
    }
    let tree = must(work, &["write-tree"])?;
    let note = "Sort merged lists and refresh Cargo.lock";
    must(work, &["commit-tree", &tree, "-p", &plan.tip, "-m", note])
}
