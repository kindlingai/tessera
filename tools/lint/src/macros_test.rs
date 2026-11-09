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

/// Files named `crates/a/src/{prefix}{i}.rs` for `i` in `0..n`.
fn many(n: usize, prefix: impl Fn(usize) -> String, text: &str) -> Vec<Source> {
    (0..n)
        .map(|i| source(&format!("crates/a/src/{}{i}.rs", prefix(i)), text))
        .collect()
}

/// Source sets and the rules each breaks.
fn cases() -> Vec<(Vec<Source>, Vec<&'static str>)> {
    let defs = || source("crates/a/src/macros/mod.rs", DEF);
    let lib = |text| source("crates/a/src/lib.rs", text);
    let mut capped = many(25, |_| "f".into(), "fn f() {}\n");
    capped.push(source("crates/a/src/macros/mod.rs", "mod x;\n"));
    vec![
        (
            vec![defs(), lib("fn f() { twice!(1); twice!(2); }\n")],
            vec![],
        ),
        // Calls inside the definition do not count.
        (
            vec![defs(), lib("fn f() { twice!(1); }\n")],
            vec!["macro-calls"],
        ),
        (
            vec![source(
                "crates/a/src/macros/mod.rs",
                "mod more;\nfn helper() {}\n",
            )],
            vec!["macro-content"],
        ),
        (
            many(6, |_| "macros/m".into(), "mod x;\n"),
            vec!["macro-files"],
        ),
        (capped, vec![]),
        (many(26, |_| "f".into(), "fn f() {}\n"), vec!["crate-files"]),
        (many(26, |i| format!("d{}/f", i % 2), "fn f() {}\n"), vec![]),
    ]
}

#[test]
fn macro_rules_checked() {
    for (i, (sources, want)) in cases().into_iter().enumerate() {
        assert_eq!(rules(&sources), want, "case {i}");
    }
}
