//! Runs each check as a program, and reads its JSON.

use std::path::{Path, PathBuf};
use std::process::Command;
use tessera_findings::Report;

/// The text style reads: staged, a range, or a message.
pub enum Changes {
    Staged,
    Range(String, String),
    Message(String),
}

pub type Check = fn(&Dirs, &Changes) -> Result<Report, String>;

/// Shared models and embeddings, and this checkout's reports.
pub struct Dirs {
    pub root: PathBuf,
    pub spacy: PathBuf,
    pub eg2: PathBuf,
    pub cache: PathBuf,
    pub out: PathBuf,
}

impl Dirs {
    /// Models under `shared`, or this checkout's `target/check`.
    pub fn new(root: &Path, shared: Option<&Path>) -> Self {
        let own = root.join("target/check");
        let base = shared.map_or(own.clone(), Path::to_path_buf);
        Self {
            root: root.to_path_buf(),
            spacy: base.join("spacy"),
            eg2: base.join("eg2-text"),
            cache: base.join("simcheck"),
            out: own.join("reports"),
        }
    }

    /// `path` relative to the repository.
    fn short(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

const FETCH: &str = "run `cargo run -p tessera-check -- fetch`";

/// Runs cargo to write `json`. Exit 1 means findings.
fn cargo(dirs: &Dirs, name: &str, args: &[&str], json: &Path) -> Result<Report, String> {
    eprintln!("tessera-check: running {name}");
    let out = Command::new("cargo")
        .args(args)
        .current_dir(&dirs.root)
        .output();
    let out = out.map_err(|e| format!("{name}: {e}"))?;
    if !matches!(out.status.code(), Some(0 | 1)) || !json.exists() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("{name} failed: {}", err.trim()));
    }
    Report::read(json)
}

fn skipped(tool: &str, title: &str, why: String) -> Report {
    Report {
        skipped: Some(why),
        ..Report::new(tool, title)
    }
}

/// A release build runs the whole-repository rules several times faster.
const LINT: [&str; 5] = ["run", "-q", "--release", "-p", "tessera-lint"];

pub fn lint(dirs: &Dirs, _: &Changes) -> Result<Report, String> {
    let json = dirs.out.join("lint.json");
    let path = json.to_string_lossy();
    let args = [&LINT[..], &["--", "--json", &path]].concat();
    cargo(dirs, "lint", &args, &json)
}

/// Paths the changes add or modify.
fn changed(dirs: &Dirs, changes: &Changes) -> Result<Vec<String>, String> {
    let mut args = vec!["diff", "--name-only", "--diff-filter=d"];
    let range;
    match changes {
        Changes::Staged => args.push("--cached"),
        Changes::Range(base, head) => {
            range = format!("{base}...{head}");
            args.push(&range);
        }
        Changes::Message(_) => return Ok(Vec::new()),
    }
    let out = Command::new("git")
        .args(&args)
        .current_dir(&dirs.root)
        .output();
    let out = out.map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {} failed", args.join(" ")));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().map(String::from).collect())
}

/// Lint's comment rules, on the changed files only.
pub fn comments(dirs: &Dirs, changes: &Changes) -> Result<Report, String> {
    let json = dirs.out.join("comments.json");
    let path = json.to_string_lossy();
    let files = changed(dirs, changes)?;
    let mut args = [&LINT[..], &["--", "comments", "--json", &path]].concat();
    args.extend(files.iter().map(String::as_str));
    let mut report = cargo(dirs, "comments", &args, &json)?;
    report.tool = "comments".into();
    report.title = "Comment rules on changed files".into();
    Ok(report)
}

pub fn style(dirs: &Dirs, changes: &Changes) -> Result<Report, String> {
    if !dirs.spacy.join("tokenizer").exists() {
        let why = format!("no spaCy pipeline in {}: {FETCH}", dirs.short(&dirs.spacy));
        return Ok(skipped("style", "Comment style", why));
    }
    let json = dirs.out.join("style.json");
    let (spacy, path) = (dirs.spacy.to_string_lossy(), json.to_string_lossy());
    let mut args = vec!["run", "-q", "--release", "-p", "tessera-style", "--"];
    match changes {
        Changes::Staged => args.extend(["staged", &spacy]),
        Changes::Range(base, head) => args.extend(["diff", &spacy, base, head]),
        Changes::Message(file) => args.extend(["message", &spacy, file]),
    }
    args.push(&path);
    cargo(dirs, "style", &args, &json)
}

pub fn simcheck(dirs: &Dirs, changes: &Changes) -> Result<Report, String> {
    if !dirs.eg2.join("config.json").exists() || !dirs.eg2.join("model.safetensors").exists() {
        let why = format!(
            "no EmbeddingGemma model in {}: {FETCH}",
            dirs.short(&dirs.eg2)
        );
        return Ok(skipped("simcheck", "Near-copies of code", why));
    }
    let json = dirs.out.join("simcheck.json");
    let (eg2, cache, path) = (
        dirs.eg2.to_string_lossy(),
        dirs.cache.to_string_lossy(),
        json.to_string_lossy(),
    );
    let manifest = "tools/simcheck/Cargo.toml";
    let mut args = vec![
        "run",
        "-q",
        "--release",
        "--manifest-path",
        manifest,
        "--",
        "check",
        &eg2,
        &cache,
        &path,
    ];
    // New files count from the range's base.
    if let Changes::Range(base, _) = changes {
        args.push(base);
    }
    cargo(dirs, "simcheck", &args, &json)
}

/// Downloads both models into the shared check folder.
pub fn fetch(dirs: &Dirs) -> Result<(), String> {
    for dir in [&dirs.spacy, &dirs.eg2] {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let (spacy, eg2) = (dirs.spacy.to_string_lossy(), dirs.eg2.to_string_lossy());
    let style = [
        "run",
        "-q",
        "--release",
        "-p",
        "tessera-style",
        "--",
        "fetch",
        &spacy,
    ];
    let manifest = "tools/simcheck/Cargo.toml";
    let simcheck = [
        "run",
        "-q",
        "--release",
        "--manifest-path",
        manifest,
        "--",
        "fetch",
        &eg2,
    ];
    for args in [&style[..], &simcheck[..]] {
        let status = Command::new("cargo")
            .args(args)
            .current_dir(&dirs.root)
            .status();
        if !status.map_err(|e| e.to_string())?.success() {
            return Err(format!("cargo {} failed", args.join(" ")));
        }
    }
    Ok(())
}
