//! The repository's code-shape rules. AGENTS.md lists them.

mod audit;
mod comments;
mod files;
mod kernel;
mod macros;
mod names;
mod readme;
mod runs;
mod rust;
mod share;
mod text;

#[cfg(test)]
mod comments_test;
#[cfg(test)]
mod macros_test;
#[cfg(test)]
mod names_test;
#[cfg(test)]
mod readme_test;
#[cfg(test)]
mod runs_test;
#[cfg(test)]
mod rust_test;
#[cfg(test)]
mod share_test;
#[cfg(test)]
mod text_test;

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
    pub shape_steps: Vec<(usize, usize)>,
    pub extensions: Vec<String>,
    pub hash_comment_files: Vec<String>,
    pub layers: Vec<Vec<String>>,
    pub layer_allow: Vec<audit::Allow>,
    pub warn_rules: Vec<String>,
    pub min_repeat_statements: usize,
    pub timing_exempt: Vec<String>,
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
    if std::env::args().nth(1).as_deref() == Some("tables") {
        return match readme::write(&root, &dirs) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("tessera-lint: {e}");
                ExitCode::from(2)
            }
        };
    }
    let (mut found, mut names, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    for src in &sources {
        check_file(src, &mut found, &mut names, &mut tests);
    }
    share::check(&dirs, &sources, &tests, &mut found);
    names::check(&dirs, &names, &mut found);
    macros::check(&dirs, &sources, &mut found);
    files::check_crates(&dirs, &sources, &mut found);
    runs::check(&sources, &mut found);
    match files::hash_sources(&root) {
        Ok(srcs) => srcs
            .iter()
            .for_each(|s| comments::check_hash(s, &mut found)),
        Err(e) => {
            eprintln!("tessera-lint: {e}");
            return ExitCode::from(2);
        }
    }
    if let Err(e) = readme::check(&root, &dirs, &mut found) {
        eprintln!("tessera-lint: {e}");
        return ExitCode::from(2);
    }
    if let Err(e) = audit::check(&root, &dirs, &sources, &mut found) {
        eprintln!("tessera-lint: {e}");
        return ExitCode::from(2);
    }
    found.sort_by(|a, b| (&a.path, a.line).cmp(&(&b.path, b.line)));
    for v in &found {
        println!("{}:{}: {}: {}", v.path, v.line, v.rule, v.message);
    }
    let args: Vec<String> = std::env::args().collect();
    if let [_, flag, path] = &args[..]
        && flag == "--json"
        && let Err(e) = report(&found).write(std::path::Path::new(path))
    {
        eprintln!("tessera-lint: {e}");
        return ExitCode::from(2);
    }
    let fails = found.iter().filter(|v| !is_note(v)).count();
    let notes = found.len() - fails;
    if fails == 0 {
        eprintln!("tessera-lint: {} files clean, {notes} notes", sources.len());
        return ExitCode::SUCCESS;
    }
    eprintln!("tessera-lint: {fails} violations, {notes} notes");
    ExitCode::FAILURE
}

/// A note advises and does not fail the run.
fn is_note(v: &Violation) -> bool {
    audit::level(v.rule, &LIMITS.warn_rules) == findings::Level::Note
}

/// Violations as findings, for tessera-check.
fn report(found: &[Violation]) -> findings::Report {
    let mut r = findings::Report::new("lint", "Code shape");
    for v in found {
        r.findings.push(findings::Finding {
            rule: v.rule.into(),
            level: audit::level(v.rule, &LIMITS.warn_rules),
            path: v.path.clone(),
            line: v.line,
            message: v.message.clone(),
            snippets: Vec::new(),
            fix: None,
        });
    }
    r
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
        text::check(src, out);
    } else {
        kernel::check(src, names);
    }
}
