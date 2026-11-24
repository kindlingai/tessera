use super::generate::lowering;
use super::{Pick, pick};
use crate::start::{Options, flags};
use tessera_engine::form::{BF16_FORM, F32_FORM};
use tessera_engine::{Pins, Recipe};

/// Owned flags parse, and the most given flags pick.
#[test]
fn flags_pick_entries() {
    let mut o = Options::default();
    assert!(o.switch("--guided"));
    assert!(!o.explain_lowering);
    assert!(o.switch("--explain-lowering"));
    assert!(o.explain_lowering);
    assert!(flags().contains("--explain-lowering"));
    assert!(flags().contains("--kv-dtype"));
    assert!(o.flag("--image-size", "big".into()).is_none());
    assert!(o.flag("--max-batch-rows", "0".into()).is_none());
    o.flag("--max-batch-rows", "64".into()).unwrap();
    assert_eq!(o.max_batch_rows, Some(64));
    assert!(o.flag("--kv-dtype", "int4_g32".into()).is_none());
    assert!(o.flag("--kv-dtype", "bogus".into()).is_none());
    o.flag("--kv-dtype", "fp8".into()).unwrap();
    assert_eq!(o.kv_dtype.as_deref(), Some("fp8"));
    let base = Pins::all(BF16_FORM).pin("*.down", BF16_FORM);
    o.lower = Some(Recipe {
        pins: Pins::all(F32_FORM).kept(BF16_FORM),
        ..Default::default()
    });
    let merged = lowering(&o, Some(base.clone()));
    assert_eq!(merged.pins.weights.name(), "bf16");
    assert_eq!(merged.pins.named.len(), 1);
    assert_eq!(merged.pins.kept.name(), "fp8");
    o.kv_dtype = None;
    let from_recipe = lowering(&o, Some(base));
    assert_eq!(from_recipe.pins.kept.name(), "bf16");
    assert_eq!(from_recipe.pins.named.len(), 1);
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
