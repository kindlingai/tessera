use super::{Pick, pick};
use crate::start::{Options, flags};

/// Owned flags parse, and the most given flags pick.
#[test]
fn flags_pick_entries() {
    let mut o = Options::default();
    assert!(o.switch("--guided"));
    assert!(!o.explain_lowering);
    assert!(o.switch("--explain-lowering"));
    assert!(o.explain_lowering);
    assert!(flags().contains("--explain-lowering"));
    assert!(o.flag("--image-size", "big".into()).is_none());
    assert!(o.flag("--max-batch-rows", "0".into()).is_none());
    o.flag("--max-batch-rows", "64".into()).unwrap();
    assert_eq!(o.max_batch_rows, Some(64));
    assert!(o.flag("--task", "paint".into()).is_none());
    assert!(matches!(pick(&o).map(|e| &e.pick), Ok(Pick::Plain)));
    o.flag("--draft", "d".into()).unwrap();
    assert!(matches!(pick(&o).map(|e| &e.pick), Ok(Pick::Given(_))));
    assert!(o.flag("--mtp", "0".into()).is_none());
    o.flag("--mtp", "2".into()).unwrap();
    let both = pick(&o).map(|e| e.pick);
    assert!(matches!(both, Ok(Pick::Given(flags)) if flags.len() == 2));
    o.flag("--task", "image".into()).unwrap();
    assert!(pick(&o).is_err());
}
