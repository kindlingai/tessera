use crate::far::Lean;
use crate::pairs::Hit;
use crate::report::report;
use crate::windows::cut;
use findings::Level;
use std::path::Path;

#[test]
fn report_lists_each_kind() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wins = cut("src/fetch.rs", include_str!("fetch.rs"), 20, 7);
    let hit = Hit {
        a: 0,
        b: 1,
        score: 0.98,
        count: 2,
    };
    let pair = ("a.rs".to_string(), "b.rs".to_string());
    let lean = Lean {
        path: "a/x.rs".into(),
        home: "a".into(),
        own: 0.8,
        other: "b".into(),
        near: 0.9,
    };
    let r = report((&[hit], &[&pair], &[lean]), &wins, root, (20, 0.97)).unwrap();
    let rules: Vec<&str> = r.findings.iter().map(|f| f.rule.as_str()).collect();
    assert_eq!(rules, ["near-copy", "stale-allow", "new-file-lean"]);
    assert!(r.findings[0].snippets[0].text.starts_with("//! Downloads"));
    assert_eq!(
        r.findings[0].fix.as_deref(),
        Some("[\"src/fetch.rs\", \"src/fetch.rs\"],")
    );
    assert_eq!((r.fails(), r.findings[2].level == Level::Note), (2, true));
}
