use super::unused::check;
use crate::files::Source;

fn src(path: &str, text: &str) -> Source {
    Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    }
}

/// Names flagged, with crates `a/` and `b/`.
fn flagged(files: &[(&str, &str)]) -> Vec<String> {
    let sources: Vec<Source> = files.iter().map(|(p, t)| src(p, t)).collect();
    let dirs = ["a/".to_string(), "b/".to_string()];
    let mut out = Vec::new();
    check(&dirs, &sources, &mut out);
    let quoted = |m: &str| m.split('`').nth(1).unwrap().to_string();
    let mut names: Vec<String> = out.iter().map(|v| quoted(&v.message)).collect();
    names.sort();
    names
}

const LIB: &str =
    "pub fn used() {}\npub fn dead() {}\npub struct Out;\npub fn make() -> Out { Out }\n";

#[test]
fn other_crates_decide() {
    let user = ("b/src/lib.rs", "fn f() { a::used(); a::make(); }\n");
    assert_eq!(flagged(&[("a/src/lib.rs", LIB), user]), ["dead"]);
}

#[test]
fn own_targets_count() {
    let test = (
        "a/tests/t.rs",
        "fn f() { a::used(); a::dead(); a::make(); }\n",
    );
    assert!(flagged(&[("a/src/lib.rs", LIB), test]).is_empty());
    let inline = ("a/src/lib_test.rs", "fn f() { used(); dead(); make(); }\n");
    assert_eq!(flagged(&[("a/src/lib.rs", LIB), inline]).len(), 4);
}

#[test]
fn kinds_are_found() {
    let lib = "#[macro_export]\nmacro_rules! m { () => {} }\npub use x::{One, two as Two};\n\
               pub const C: u8 = 1;\npub(crate) fn hidden() {}\npub mod inner { pub trait T {} }\n\
               #[cfg(test)]\nmod t { pub fn skip() {} }\n";
    let all = ["C", "One", "T", "Two", "m"];
    assert_eq!(flagged(&[("a/src/lib.rs", lib)]), all);
}

#[test]
fn binaries_export_nothing() {
    assert!(flagged(&[("a/src/main.rs", "pub fn go() {}\n")]).is_empty());
}

#[test]
fn macro_items_reach_types() {
    let lib = "pub struct Code;\npub struct Block<C>(C);\n\
               macro_rules! make { ($n:ident) => { pub const $n: Block<Code> = Block(Code); }; }\n\
               make!(NAME);\n";
    let user = ("b/src/lib.rs", "fn f() { let _ = a::NAME; }\n");
    assert!(flagged(&[("a/src/lib.rs", lib), user]).is_empty());
    assert_eq!(flagged(&[("a/src/lib.rs", lib)]), ["Block", "Code"]);
}

#[test]
fn bounds_reach_traits() {
    let lib = "pub trait Code {}\npub struct Block<C>(C);\n\
               impl<C: Code> Block<C> { pub fn new(c: C) -> Self { Self(c) } }\n";
    let user = ("b/src/lib.rs", "fn f() { let _ = a::Block::new(1); }\n");
    assert!(flagged(&[("a/src/lib.rs", lib), user]).is_empty());
}
