use super::{Pick, pick};
use crate::start::Options;

/// Owned flags parse, and `--draft` picks the drafter.
#[test]
fn flags_pick_entries() {
    let mut o = Options::default();
    assert!(o.switch("--guided"));
    assert!(o.flag("--image-size", "big".into()).is_none());
    assert!(o.flag("--task", "paint".into()).is_none());
    assert!(matches!(pick(&o).map(|e| &e.pick), Ok(Pick::Plain)));
    o.flag("--draft", "d".into()).unwrap();
    assert!(matches!(pick(&o).map(|e| &e.pick), Ok(Pick::Given(_))));
    assert!(o.flag("--mtp", "0".into()).is_none());
    o.flag("--mtp", "2".into()).unwrap();
    let conflict = pick(&o).err();
    assert_eq!(
        conflict.as_deref(),
        Some("--mtp cannot combine with --draft")
    );
    o.flag("--task", "image".into()).unwrap();
    assert!(pick(&o).is_err());
}
