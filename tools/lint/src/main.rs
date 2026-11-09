//! The repository's code-shape rules. AGENT.md lists them.

mod comments;
mod files;
mod kernel;
mod rust;

#[cfg(test)]
mod comments_test;
#[cfg(test)]
mod rust_test;

use std::process::ExitCode;

pub const MAX_COMMENT_WORDS: usize = 10;
pub const MAX_NAME_SEGMENTS: usize = 3;
pub const MAX_FILE_LINES: usize = 1000;
pub const MAX_CRATE_FILES: usize = 30;
pub const MAX_CRATE_LINES: usize = 15000;
/// Most of a file inline tests may fill.
pub const MAX_TEST_PERCENT: usize = 1;

pub struct Violation {
    pub path: String,
    pub line: usize,
    pub rule: &'static str,
    pub message: String,
}

fn main() -> ExitCode {
    let root = files::repo_root();
    let sources = match files::sources(&root) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tessera-lint: {e}");
            return ExitCode::from(2);
        }
    };
    let mut found = Vec::new();
    for src in &sources {
        check_file(src, &mut found);
    }
    files::check_crates(&root, &sources, &mut found);
    found.sort_by(|a, b| (&a.path, a.line).cmp(&(&b.path, b.line)));
    for v in &found {
        println!("{}:{}: {}: {}", v.path, v.line, v.rule, v.message);
    }
    if found.is_empty() {
        eprintln!("tessera-lint: {} files clean", sources.len());
        return ExitCode::SUCCESS;
    }
    eprintln!("tessera-lint: {} violations", found.len());
    ExitCode::FAILURE
}

fn check_file(src: &files::Source, out: &mut Vec<Violation>) {
    if src.lines > MAX_FILE_LINES {
        out.push(src.violation(
            1,
            "file-lines",
            format!("{} lines (max {MAX_FILE_LINES})", src.lines),
        ));
    }
    let is_rust = src.path.ends_with(".rs");
    comments::check(src, is_rust, out);
    if is_rust {
        rust::check(src, out);
    } else {
        kernel::check(src, out);
    }
}

/// Snake-case pieces of a name, ignoring leading or doubled underscores.
pub fn segments(name: &str) -> usize {
    let name = name.strip_prefix("r#").unwrap_or(name);
    name.split('_').filter(|s| !s.is_empty()).count()
}
