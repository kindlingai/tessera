//! Which files the rules read, and the per-crate totals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{MAX_CRATE_FILES, MAX_CRATE_LINES, Violation};

/// Rust plus the GPU kernel languages.
pub const EXTENSIONS: &[&str] = &[
    "rs", "metal", "cu", "cuh", "h", "hpp", "c", "cc", "cpp", "wgsl", "hip",
];

pub struct Source {
    pub path: String,
    pub text: String,
    pub lines: usize,
}

impl Source {
    pub fn violation(&self, line: usize, rule: &'static str, message: String) -> Violation {
        Violation {
            path: self.path.clone(),
            line,
            rule,
            message,
        }
    }
}

pub fn repo_root() -> PathBuf {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output();
    match out {
        Ok(o) if o.status.success() => PathBuf::from(String::from_utf8_lossy(&o.stdout).trim()),
        _ => PathBuf::from("."),
    }
}

/// Tracked and untracked, unignored paths.
fn listed(root: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("git ls-files: {e}"))?;
    if !out.status.success() {
        return Err("git ls-files failed".into());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(String::from)
        .collect())
}

pub fn sources(root: &Path) -> Result<Vec<Source>, String> {
    let mut out = Vec::new();
    for path in listed(root)? {
        let ext = Path::new(&path).extension().and_then(|e| e.to_str());
        if !ext.is_some_and(|e| EXTENSIONS.contains(&e)) {
            continue;
        }
        // Deleted but still in the index.
        let Ok(text) = std::fs::read_to_string(root.join(&path)) else {
            continue;
        };
        let lines = text.lines().count();
        out.push(Source { path, text, lines });
    }
    Ok(out)
}

/// Directories holding a package manifest.
fn crate_dirs(root: &Path) -> Result<Vec<String>, String> {
    let mut dirs = Vec::new();
    for path in listed(root)? {
        let Some(dir) = path.strip_suffix("Cargo.toml") else {
            continue;
        };
        let text = std::fs::read_to_string(root.join(&path)).unwrap_or_default();
        if text.lines().any(|l| l.trim() == "[package]") {
            dirs.push(dir.to_string());
        }
    }
    Ok(dirs)
}

pub fn check_crates(root: &Path, sources: &[Source], out: &mut Vec<Violation>) {
    let dirs = match crate_dirs(root) {
        Ok(d) => d,
        Err(e) => {
            out.push(Violation {
                path: ".".into(),
                line: 1,
                rule: "crate-size",
                message: e,
            });
            return;
        }
    };
    let mut totals: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for src in sources {
        // A file belongs to its deepest enclosing crate.
        let owner = dirs
            .iter()
            .filter(|d| src.path.starts_with(d.as_str()))
            .max_by_key(|d| d.len());
        if let Some(dir) = owner {
            let t = totals.entry(dir.as_str()).or_default();
            t.0 += 1;
            t.1 += src.lines;
        }
    }
    for (dir, (files, lines)) in totals {
        let path = format!("{dir}Cargo.toml");
        if files > MAX_CRATE_FILES {
            out.push(Violation {
                path: path.clone(),
                line: 1,
                rule: "crate-files",
                message: format!("{files} source files (max {MAX_CRATE_FILES})"),
            });
        }
        if lines > MAX_CRATE_LINES {
            out.push(Violation {
                path,
                line: 1,
                rule: "crate-lines",
                message: format!("{lines} source lines (max {MAX_CRATE_LINES})"),
            });
        }
    }
}
