use crate::files::Source;
use crate::share::{TestFile, check};

fn src(path: &str, lines: usize) -> Source {
    Source {
        path: path.into(),
        text: String::new(),
        lines,
    }
}

/// Rules broken when `test` has `lines` test lines.
fn share(test: &str, lines: usize, sources: &[Source]) -> Vec<String> {
    let mut out = Vec::new();
    let dirs = ["c/".to_string()];
    check(
        &dirs,
        sources,
        &[TestFile {
            path: test.into(),
            lines,
        }],
        &mut out,
    );
    out.iter().map(|v| v.path.clone()).collect()
}

#[test]
fn tests_pair() {
    let code = [src("c/src/foo.rs", 75), src("c/src/bar/mod.rs", 75)];
    assert!(share("c/src/foo_test.rs", 25, &code).is_empty());
    assert_eq!(share("c/src/foo_test.rs", 26, &code), ["c/src/foo_test.rs"]);
    assert_eq!(
        share("c/src/bar/bar_test.rs", 26, &code),
        ["c/src/bar/bar_test.rs"]
    );
    assert_eq!(share("c/src/lone_test.rs", 51, &code), ["c/Cargo.toml"]);
    assert!(share("c/tests/end_test.rs", 50, &code).is_empty());
}
