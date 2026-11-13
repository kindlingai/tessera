use crate::shape::shape;
use crate::windows::{cut, overlap};

const SPACING: usize = 7;

#[test]
fn windows_follow_content() {
    let renamed = shape("let k = m.function(name)?;");
    assert_eq!(shape("let kernel = module.function(entry)?;"), renamed);
    let src = include_str!("fetch.rs");
    let before = cut("a.rs", src, 20, SPACING);
    let moved = format!("// moved\nfn x() {{}}\n\n{src}");
    let after: Vec<String> = cut("a.rs", &moved, 20, SPACING)
        .into_iter()
        .map(|w| w.text)
        .collect();
    // Inserted lines move anchors only within one run of them.
    for w in before.iter().filter(|w| w.line > 2 * SPACING) {
        assert!(after.contains(&w.text), "window at line {} moved", w.line);
    }
    assert_eq!(before[0].line, 1);
    assert!(overlap(&before[0], &before[1], 20));
    assert!(cut("a.rs", &"}\n".repeat(40), 20, SPACING).is_empty());
}
