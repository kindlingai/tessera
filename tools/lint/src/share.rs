//! Limits `#[test]` lines against the code they test.

use std::collections::BTreeMap;
use std::path::Path;

use crate::files::{Source, owner};
use crate::rust::is_test_file;
use crate::{LIMITS, Violation};

/// Lines of `#[test]` functions in one test file.
pub struct TestFile {
    pub path: String,
    pub lines: usize,
}

/// `foo.rs` for `foo_test.rs`, or `foo/mod.rs` for `foo/foo_test.rs`.
fn module_of<'a>(path: &str, sources: &'a [Source]) -> Option<&'a Source> {
    let p = Path::new(path);
    let base = p.file_stem()?.to_str()?.strip_suffix("_test")?;
    let dir = p.parent()?;
    let mut paths = vec![dir.join(format!("{base}.rs"))];
    if dir.file_name().is_some_and(|d| d == base) {
        paths.push(dir.join("mod.rs"));
    }
    sources
        .iter()
        .find(|s| paths.iter().any(|c| Path::new(&s.path) == c))
}

fn over(tests: usize, code: usize) -> bool {
    tests * 100 > (tests + code) * LIMITS.max_module_test_percent
}

fn message(tests: usize, code: usize, what: &str) -> String {
    let max = LIMITS.max_module_test_percent;
    format!("{tests} test lines beside {code} lines of {what} (max {max}% of both)")
}

/// Unpaired test files count against their crate's other code.
pub fn check(dirs: &[String], sources: &[Source], tests: &[TestFile], out: &mut Vec<Violation>) {
    let mut pooled: BTreeMap<&str, usize> = BTreeMap::new();
    for t in tests {
        match module_of(&t.path, sources) {
            Some(m) if over(t.lines, m.lines) => out.push(Violation {
                path: t.path.clone(),
                line: 1,
                rule: "test-share",
                message: message(t.lines, m.lines, &m.path),
            }),
            Some(_) => {}
            None => {
                if let Some(dir) = owner(dirs, &t.path) {
                    *pooled.entry(dir).or_default() += t.lines;
                }
            }
        }
    }
    for (dir, lines) in pooled {
        let code = sources
            .iter()
            .filter(|s| owner(dirs, &s.path) == Some(dir) && !is_test_file(&s.path))
            .map(|s| s.lines)
            .sum();
        if over(lines, code) {
            out.push(Violation {
                path: format!("{dir}Cargo.toml"),
                line: 1,
                rule: "test-share",
                message: message(lines, code, "crate code"),
            });
        }
    }
}
