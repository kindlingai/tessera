//! The repository's code-shape rules, listed in AGENTS.md.

mod audit;
mod comments;
mod files;
mod kernel;
mod macros;
mod names;
mod readme;
mod repo;
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
    pub max_description_words: usize,
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
    #[serde(default)]
    pub layer_allow: Vec<audit::Allow>,
    pub facade: audit::Facade,
    pub placement_gap: usize,
    pub thin_edge: audit::Thin,
    pub warn_rules: Vec<String>,
    pub min_repeat_statements: usize,
    pub repeat_exprs: audit::Exprs,
    pub timing_exempt: Vec<String>,
    #[serde(default)]
    pub allowed_type_copies: Vec<Vec<String>>,
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

/// A conflicted file skips the per-file rules and shape runs.
const CONFLICT: &str = "unresolved merge conflict. Resolve it, then run lint again.";

fn main() -> ExitCode {
    tessera_findings::main_of("tessera-lint", run)
}

/// `[comments] [--json <file>] [<path>...]`, or `tables`.
fn run(args: &[&str]) -> Result<bool, String> {
    let root = files::repo_root();
    if args == ["tables"] {
        return readme::write(&root, &files::crate_dirs(&root)?).map(|()| true);
    }
    let (only, rest) = match args {
        ["comments", rest @ ..] => (true, rest),
        _ => (false, args),
    };
    let (json, paths) = match rest {
        ["--json", file, paths @ ..] => (Some(*file), paths),
        _ => (None, rest),
    };
    let (mut found, count) = if only {
        comment_rules(&root, paths)?
    } else if paths.is_empty() {
        every_rule(&root)?
    } else {
        return Err("usage: tessera-lint [comments] [--json <file>] [<path>...] | tables".into());
    };
    found.sort_by(|a, b| (&a.path, a.line).cmp(&(&b.path, b.line)));
    for v in &found {
        println!("{}:{}: {}: {}", v.path, v.line, v.rule, v.message);
    }
    if let Some(file) = json {
        report(&found).write(std::path::Path::new(file))?;
    }
    let fails = found.iter().filter(|v| !is_note(v)).count();
    let notes = found.len() - fails;
    if fails == 0 {
        eprintln!("tessera-lint: {count} files clean, {notes} notes");
        return Ok(true);
    }
    eprintln!("tessera-lint: {fails} violations, {notes} notes");
    Ok(false)
}

/// The comment rules alone, on `paths`.
fn comment_rules(
    root: &std::path::Path,
    paths: &[&str],
) -> Result<(Vec<Violation>, usize), String> {
    let mut found = Vec::new();
    let named = |s: &files::Source| paths.contains(&s.path.as_str());
    let sources: Vec<files::Source> = files::sources(root)?.into_iter().filter(named).collect();
    for src in &sources {
        comments::check(src, src.path.ends_with(".rs"), &mut found);
    }
    for src in files::hash_sources(root)?.iter().filter(|s| named(s)) {
        comments::check_hash(src, &mut found);
    }
    Ok((found, sources.len()))
}

fn every_rule(root: &std::path::Path) -> Result<(Vec<Violation>, usize), String> {
    let sources = files::sources(root)?;
    let dirs = files::crate_dirs(root)?;
    let (mut found, mut names, mut tests) = (Vec::new(), Vec::new(), Vec::new());
    for src in &sources {
        match tessera_shape::conflict(&src.text) {
            Some(line) => found.push(src.violation(line, "conflict", CONFLICT.into())),
            None => check_file(src, &mut found, &mut names, &mut tests),
        }
    }
    share::check(&dirs, &sources, &tests, &mut found);
    names::check(&dirs, &names, &mut found);
    macros::check(&dirs, &sources, &mut found);
    files::check_crates(&dirs, &sources, &mut found);
    runs::check(&sources, &mut found);
    repo::copies::check(&dirs, &sources, &mut found);
    repo::revisions::check(root, &sources, &mut found);
    for src in &files::hash_sources(root)? {
        comments::check_hash(src, &mut found);
    }
    readme::check(root, &dirs, &mut found)?;
    audit::check(root, &dirs, &sources, &mut found)?;
    Ok((found, sources.len()))
}

/// A note advises and does not fail the run.
fn is_note(v: &Violation) -> bool {
    audit::level(v.rule, &LIMITS.warn_rules) == tessera_findings::Level::Note
}

/// Violations as findings, for tessera-check.
fn report(found: &[Violation]) -> tessera_findings::Report {
    let mut r = tessera_findings::Report::new("lint", "Code shape");
    for v in found {
        r.findings.push(tessera_findings::Finding {
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
