use crate::pairs::Hit;
use crate::report::{MARKER, markdown};
use crate::windows::cut;
use std::path::Path;

#[test]
fn report_shows_hunks() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let wins = cut("src/fetch.rs", include_str!("fetch.rs"), 20, 7);
    let hit = Hit {
        a: 0,
        b: 1,
        score: 0.98,
        count: 2,
    };
    let md = markdown(
        &[hit],
        &[],
        &wins,
        root,
        (20, 0.97),
        Some("https://x/blob/abc"),
    )
    .unwrap();
    assert!(md.starts_with(MARKER) && md.contains("](https://x/blob/abc/src/fetch.rs#L1-L20)"));
    assert!(
        md.contains("````rust\n//! Downloads")
            && md.contains("[\"src/fetch.rs\", \"src/fetch.rs\"],")
    );
    let clean = markdown(&[], &[], &wins, root, (20, 0.97), None).unwrap();
    assert!(clean.ends_with("None found.\n"));
    let pair = ("a.rs".to_string(), "b.rs".to_string());
    let stale = markdown(&[], &[&pair], &wins, root, (20, 0.97), None).unwrap();
    assert!(stale.contains("allow nothing") && stale.contains("[\"a.rs\", \"b.rs\"],"));
}
