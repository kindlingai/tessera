use super::{clashes, level};
use crate::files::Source;
use sensai_findings::Level;

#[test]
fn listed_rules_advise() {
    let warn = ["unused-pub".to_string()];
    assert!(level("unused-pub", &warn) == Level::Note);
    assert!(level("file-lines", &warn) == Level::Fail);
}

fn src(path: &str, text: &str) -> Source {
    Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    }
}

#[test]
fn merged_crates_clash() {
    let mac = "#[macro_export]\nmacro_rules! registry { () => {} }\n";
    let other = "#[macro_export]\nmacro_rules! other { () => {} }\n";
    let sources = [
        src("a/src/lib.rs", mac),
        src("b/src/form.rs", &format!("mod inner {{\n{mac}}}\n")),
        src("c/src/lib.rs", mac),
        src("b/src/lib.rs", other),
    ];
    let dirs = ["a/", "b/", "c/"].map(String::from);
    let groups = [vec!["a/".into(), "b/".into()]];
    let mut out = Vec::new();
    clashes(&dirs, &groups, &sources, &mut out);
    let found: Vec<_> = out.iter().map(|v| (v.path.as_str(), v.line)).collect();
    assert_eq!(found, [("a/src/lib.rs", 2), ("b/src/form.rs", 3)]);
    assert!(out[0].message.contains("b/src/form.rs:3"));
}
