//! Which files the rules read, and the per-crate totals.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{LIMITS, Violation};

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
        if !ext.is_some_and(|e| LIMITS.extensions.iter().any(|x| x == e)) {
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
pub fn crate_dirs(root: &Path) -> Result<Vec<String>, String> {
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

/// The deepest crate directory holding `path`.
pub fn owner<'a>(dirs: &'a [String], path: &str) -> Option<&'a str> {
    dirs.iter()
        .filter(|d| path.starts_with(d.as_str()))
        .max_by_key(|d| d.len())
        .map(String::as_str)
}

fn language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("Rust"),
        "metal" => Some("Metal"),
        "cu" | "cuh" => Some("CUDA"),
        "h" | "hpp" | "c" | "cc" | "cpp" => Some("C"),
        "wgsl" => Some("WGSL"),
        "hip" => Some("HIP"),
        _ => None,
    }
}

/// A `macros/` file's language and its macros directory.
pub fn macro_dir(crate_dir: &str, path: &str) -> Option<(&'static str, String)> {
    let rel = Path::new(path.strip_prefix(crate_dir)?);
    let mut dir = PathBuf::from(crate_dir);
    for part in rel.parent()?.components() {
        dir.push(part);
        if part.as_os_str() == "macros" {
            return Some((language(rel)?, dir.to_string_lossy().into_owned()));
        }
    }
    None
}

pub fn check_crates(dirs: &[String], sources: &[Source], out: &mut Vec<Violation>) {
    let mut totals: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut macros: BTreeMap<(&str, &str), Vec<String>> = BTreeMap::new();
    for src in sources {
        let Some(dir) = owner(dirs, &src.path) else {
            continue;
        };
        if let Some((lang, mdir)) = macro_dir(dir, &src.path) {
            macros.entry((dir, lang)).or_default().push(mdir);
            continue;
        }
        let t = totals.entry(dir).or_default();
        t.0 += 1;
        t.1 += src.lines;
    }
    for ((dir, lang), mut mdirs) in macros {
        let path = format!("{dir}Cargo.toml");
        let max_macros = LIMITS.max_macro_files;
        if mdirs.len() > max_macros {
            out.push(Violation {
                path: path.clone(),
                line: 1,
                rule: "macro-files",
                message: format!("{} {lang} macro files (max {max_macros})", mdirs.len()),
            });
        }
        mdirs.sort();
        mdirs.dedup();
        if mdirs.len() > 1 {
            out.push(Violation {
                path,
                line: 1,
                rule: "macro-files",
                message: format!(
                    "{lang} macros in more than one directory: {}",
                    mdirs.join(", ")
                ),
            });
        }
    }
    for (dir, (files, lines)) in totals {
        let path = format!("{dir}Cargo.toml");
        let (max_files, max_lines) = (LIMITS.max_crate_files, LIMITS.max_crate_lines);
        if files > max_files {
            out.push(Violation {
                path: path.clone(),
                line: 1,
                rule: "crate-files",
                message: format!("{files} source files (max {max_files})"),
            });
        }
        if lines > max_lines {
            out.push(Violation {
                path,
                line: 1,
                rule: "crate-lines",
                message: format!("{lines} source lines (max {max_lines})"),
            });
        }
    }
}
