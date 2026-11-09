use std::collections::BTreeMap;

use crate::files::{Source, check_crates, macro_dir};
use crate::macros::{calls, check};

fn source(path: &str, text: &str) -> Source {
    Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    }
}

fn rules(sources: &[Source]) -> Vec<&'static str> {
    let dirs = ["crates/a/".to_string()];
    let mut out = Vec::new();
    check(&dirs, sources, &mut out);
    check_crates(&dirs, sources, &mut out);
    out.iter().map(|v| v.rule).collect()
}

#[test]
fn calls_count_bangs() {
    let mut c = BTreeMap::new();
    calls("a!(); x::b![1]; { a!{} } c + d".parse().unwrap(), &mut c);
    assert_eq!(
        (c.get("a"), c.get("b"), c.get("c")),
        (Some(&2), Some(&1), None)
    );
}

#[test]
fn macro_dirs_found() {
    let m = macro_dir("crates/a/", "crates/a/src/macros/x.rs");
    assert_eq!(m, Some(("Rust", "crates/a/src/macros".into())));
    assert_eq!(macro_dir("crates/macros/", "crates/macros/src/x.rs"), None);
}

const DEF: &str = "macro_rules! twice { ($e:expr) => { $e; twice!(@ $e) }; }\n";

#[test]
fn two_calls_pass() {
    let defs = source("crates/a/src/macros/mod.rs", DEF);
    let used = source("crates/a/src/lib.rs", "fn f() { twice!(1); twice!(2); }\n");
    assert!(rules(&[defs, used]).is_empty());
}

#[test]
fn own_calls_skip() {
    let defs = source("crates/a/src/macros/mod.rs", DEF);
    let used = source("crates/a/src/lib.rs", "fn f() { twice!(1); }\n");
    assert_eq!(rules(&[defs, used]), ["macro-calls"]);
}

#[test]
fn only_macro_rules() {
    let defs = source("crates/a/src/macros/mod.rs", "mod more;\nfn helper() {}\n");
    assert_eq!(rules(&[defs]), ["macro-content"]);
}

#[test]
fn macro_files_capped() {
    let many: Vec<Source> = (0..6)
        .map(|i| source(&format!("crates/a/src/macros/m{i}.rs"), "mod x;\n"))
        .collect();
    assert_eq!(rules(&many), ["macro-files"]);
}

#[test]
fn outside_crate_limits() {
    let mut sources: Vec<Source> = (0..30)
        .map(|i| source(&format!("crates/a/src/f{i}.rs"), "fn f() {}\n"))
        .collect();
    sources.push(source("crates/a/src/macros/mod.rs", "mod x;\n"));
    assert!(rules(&sources).is_empty());
}
