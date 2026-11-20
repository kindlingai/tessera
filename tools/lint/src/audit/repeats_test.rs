use super::repeats::{Exprs, check, exprs};
use crate::files::Source;

fn flagged(files: &[(&str, &str)]) -> Vec<String> {
    let sources: Vec<Source> = files
        .iter()
        .map(|(path, text)| Source {
            path: (*path).into(),
            text: (*text).into(),
            lines: text.lines().count(),
        })
        .collect();
    let mut out = Vec::new();
    check(&sources, &mut out);
    let limits = Exprs {
        min_tokens: 24,
        min_paths: 2,
        min_places: 3,
    };
    exprs(&sources, &limits, &mut out);
    out.into_iter().map(|v| v.path).collect()
}

const FOUR: &str = "{ let mut h = 1; for b in x { h ^= b; h = h.wrapping_mul(3); } h }";

fn func(name: &str, body: &str) -> String {
    format!("fn {name}(x: &[u8]) -> u64 {body}\n")
}

#[test]
fn copies_fail() {
    let (a, b) = (func("one", FOUR), func("two", &FOUR.replace('h', "acc")));
    assert_eq!(flagged(&[("a.rs", &a), ("b.rs", &b)]), ["a.rs"]);
}

#[test]
fn different_calls_differ() {
    let (a, b) = (
        func("one", FOUR),
        func("two", &FOUR.replace("wrapping_mul", "min")),
    );
    assert!(flagged(&[("a.rs", &a), ("b.rs", &b)]).is_empty());
}

#[test]
fn small_bodies_pass() {
    let body = "{ x.len().div_ceil(2) }";
    let (a, b) = (func("one", body), func("two", body));
    assert!(flagged(&[("a.rs", &a), ("b.rs", &b)]).is_empty());
}

#[test]
fn tests_and_traits_pass() {
    let test = format!("#[test]\n{}", func("t", FOUR));
    let imp = format!("impl T for S {{ {} }}", func("m", FOUR));
    assert!(
        flagged(&[
            ("a.rs", &test),
            ("b.rs", &test),
            ("c.rs", &imp),
            ("d.rs", &imp)
        ])
        .is_empty()
    );
}

#[test]
fn matches_thrice_fail() {
    let f = "fn f(p: &[S]) -> String {\n    let seg = |s: &S| match s {\n        \
             S::Field(f) => f.clone(),\n        S::Index(i) => i.to_string(),\n    };\n    \
             p.iter().map(seg).collect()\n}\n";
    assert!(flagged(&[("a.rs", f), ("b.rs", f)]).is_empty());
    assert_eq!(flagged(&[("a.rs", f), ("b.rs", f), ("c.rs", f)]), ["a.rs"]);
}
