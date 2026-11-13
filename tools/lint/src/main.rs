//! The repository's code-shape rules. AGENTS.md lists them.

mod comments;
mod files;
mod kernel;
mod macros;
mod names;
mod runs;
mod rust;
mod shape;
mod share;

#[cfg(test)]
mod comments_test;
#[cfg(test)]
mod macros_test;
#[cfg(test)]
mod names_test;
#[cfg(test)]
mod runs_test;
#[cfg(test)]
mod rust_test;
#[cfg(test)]
mod share_test;

use std::process::ExitCode;

/// The limits in the repository's lint.toml, compiled in.
#[derive(serde::Deserialize)]
pub struct Limits {
    pub max_comment_words: usize,
    pub max_name_segments: usize,
    pub max_family_segments: usize,
    pub max_file_lines: usize,
    pub max_crate_files_base: usize,
    pub max_crate_files_addl_module: usize,
    pub max_crate_lines: usize,
    pub max_macro_files: usize,
    pub min_macro_calls: usize,
    pub max_test_percent: usize,
    pub max_module_test_percent: usize,
    pub shape_run: usize,
    pub max_shape_copies: usize,
    pub extensions: Vec<String>,
}

pub static LIMITS: std::sync::LazyLock<Limits> = std::sync::LazyLock::new(|| {
    toml::from_str(include_str!("../../../lint.toml")).expect("lint.toml")
});

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
    let (mut found, mut names, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    for src in &sources {
        check_file(src, &mut found, &mut names, &mut tests);
    }
    share::check(&dirs, &sources, &tests, &mut found);
    names::check(&dirs, &names, &mut found);
    macros::check(&dirs, &sources, &mut found);
    files::check_crates(&dirs, &sources, &mut found);
    runs::check(&sources, &mut found);
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

fn check_file(
    src: &files::Source,
    out: &mut Vec<Violation>,
    names: &mut Vec<names::Name>,
    tests: &mut Vec<share::TestFile>,
) {
    let max = LIMITS.max_file_lines;
    if src.lines > max {
        out.push(src.violation(1, "file-lines", format!("{} lines (max {max})", src.lines)));
    }
    let is_rust = src.path.ends_with(".rs");
    comments::check(src, is_rust, out);
    if is_rust {
        rust::check(src, out, names, tests);
    } else {
        kernel::check(src, names);
    }
}
