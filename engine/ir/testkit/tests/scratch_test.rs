//! Checks that scratch paths are normalized without parent segments.

use std::path::Component;
use tessera_testkit::scratch;

#[test]
fn scratch_path_is_normalized() {
    let path = scratch("normalized-check");
    assert!(
        !path.components().any(|c| c == Component::ParentDir),
        "scratch path still has parent segments: {}",
        path.display()
    );
    assert!(path.ends_with("target/scratch/normalized-check"));
}
