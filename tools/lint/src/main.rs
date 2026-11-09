//! The repository's code-shape rules. AGENT.md lists them.

mod comments;
mod files;
mod kernel;
mod names;
mod rust;

#[cfg(test)]
mod comments_test;
#[cfg(test)]
mod names_test;
#[cfg(test)]
mod rust_test;

use std::process::ExitCode;

pub const MAX_COMMENT_WORDS: usize = 10;
pub const MAX_NAME_SEGMENTS: usize = 3;
/// Allowed when another function shares the first or last segment.
pub const MAX_FAMILY_SEGMENTS: usize = 4;
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
    let dirs = match files::crate_dirs(&root) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("tessera-lint: {e}");
            return ExitCode::from(2);
        }
    };
    let (mut found, mut names) = (Vec::new(), Vec::new());
    for src in &sources {
        check_file(src, &mut found, &mut names);
    }
    names::check(&dirs, &names, &mut found);
    files::check_crates(&dirs, &sources, &mut found);
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

fn check_file(src: &files::Source, out: &mut Vec<Violation>, names: &mut Vec<names::Name>) {
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
        rust::check(src, out, names);
    } else {
        kernel::check(src, names);
    }
}
