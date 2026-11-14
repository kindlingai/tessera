use super::check;
use crate::audit::workspace::Pkg;
use crate::files::Source;

fn pkg(name: &str, dir: &str) -> Pkg {
    Pkg {
        name: name.into(),
        dir: dir.into(),
        manifest: String::new(),
        deps: Vec::new(),
    }
}

/// Messages for one test file, in crates `a` and `b-c`.
fn flagged(path: &str, text: &str) -> Vec<String> {
    let src = Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    };
    let mut out = Vec::new();
    let pkgs = [
        pkg("a", "x/a/"),
        pkg("b-c", "x/b/"),
        pkg("cal", "tools/calibrate/"),
    ];
    check(&pkgs, &[src], &mut out);
    out.into_iter().map(|v| v.message).collect()
}

const TEST: &str = "#[test]\nfn t() {}\n";

#[test]
fn own_crate_passes() {
    let own = format!("use a::run;\nuse b_c::Other;\n{TEST}");
    assert!(flagged("x/a/tests/t.rs", &own).is_empty());
    let inline = format!("use crate::run;\n{TEST}");
    assert!(flagged("x/a/src/run_test.rs", &inline).is_empty());
    let path = format!("{TEST}fn f() {{ a::run(); }}\n");
    assert!(flagged("x/a/tests/t.rs", &path).is_empty());
}

#[test]
fn other_crates_fail() {
    let text = format!("use b_c::Other;\nuse std::io;\n{TEST}");
    let found = flagged("x/a/tests/t.rs", &text);
    assert_eq!(
        found,
        ["tests only b_c: move them to the crate they test, not `a`"]
    );
    assert_eq!(flagged("x/a/tests/t.rs", TEST).len(), 1);
}

#[test]
fn helpers_and_modules_pass() {
    assert!(flagged("x/a/tests/common/mod.rs", "use b_c::Other;\n").is_empty());
    let nested = format!("use a::kernels::{{b_c::X}};\n{TEST}");
    assert!(flagged("x/a/tests/t.rs", &nested).is_empty());
    assert!(flagged("x/a/src/lib.rs", &format!("use b_c::X;\n{TEST}")).is_empty());
}

#[test]
fn folders_share_names() {
    let text = format!("use super::x;\n{TEST}");
    assert_eq!(flagged("x/a/tests/m/t.rs", &text).len(), 1);
    let one = |path: &str, text: &str| Source {
        path: path.into(),
        text: text.into(),
        lines: 1,
    };
    let files = [
        one("x/a/tests/m/mod.rs", "use a::run;\n"),
        one("x/a/tests/m/t.rs", &text),
    ];
    let mut out = Vec::new();
    check(&[pkg("a", "x/a/")], &files, &mut out);
    assert!(out.is_empty());
}

#[test]
fn clocks_fail() {
    let clock = |own: &str| format!("use {own}::run;\nuse std::time::Instant;\n{TEST}");
    let found = flagged("x/a/tests/t.rs", &clock("a"));
    assert_eq!(
        found,
        ["`Instant` in a test file: time code in a bench or tools/calibrate"]
    );
    assert!(flagged("tools/calibrate/tests/t.rs", &clock("cal")).is_empty());
    assert!(flagged("x/a/src/lib.rs", "use std::time::Instant;\n").is_empty());
}
