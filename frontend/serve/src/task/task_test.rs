use super::generate::lowering;
use super::{Pick, pick};
use crate::start::{Options, flags};
use tessera_engine::form::{BF16_FORM, F32_FORM};
use tessera_engine::{Compute, DType, Dtype, Pins, Recipe, Site};

/// Owned flags parse, and the most given flags pick.
#[test]
fn flags_pick_entries() {
    let mut o = Options::default();
    assert!(o.switch("--guided"));
    assert!(!o.explain_lowering);
    assert!(o.switch("--explain-lowering"));
    assert!(o.explain_lowering);
    assert!(flags().contains("--explain-lowering"));
    assert!(o.switch("--allow-failed-lowering"));
    assert!(o.allow_failed_lowering);
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

#[test]
fn kv_cache_tokens_pages() {
    let mut opts = Options::default();
    assert!(opts.flag("--kv-cache-tokens", "0".into()).is_none());
    assert!(opts.flag("--kv-cache-tokens", "1024".into()).is_some());
    let recipe = lowering(&opts, None);
    assert_eq!(recipe.kept.name(), "paged");
    assert_eq!(recipe.kept_rows, Some(1024));
}

#[test]
fn dtypes_fold() {
    let mut lower = Recipe::default();
    lower.dtype.push(Dtype::parse("@kept=bf16").unwrap());
    let mut opts = Options::default();
    opts.lower.replace(lower);
    for flag in ["@kept=fp8", "@norm=f32", ":operand=bf16", "*.down@sum=fp8"] {
        assert!(opts.flag("--dtype", flag.into()).is_some());
    }
    let recipe = lowering(&opts, None);
    let norm = recipe.pins.site_form(Site::Norm).map(|f| f.name());
    assert_eq!((recipe.pins.kept.name(), norm), ("fp8", Some("f32")));
    assert_eq!(recipe.sums.named, [("*.down".into(), DType::Fp8)]);
    assert_eq!(recipe.compute, Compute::Bf16);
    for bad in ["@sum=u8", "@nowhere=bf16", "a.*@norm=bf16", "@kept"] {
        assert!(opts.flag("--dtype", bad.into()).is_none(), "{bad}");
    }
    assert!(opts.flag("--shape", "p2p_sum:threads=256".into()).is_some());
    assert!(opts.flag("--shape", "p2p_sum:threads=7".into()).is_none());
    assert!(opts.flag("--shape", "p2p_sum:lanes=64".into()).is_none());
    assert_eq!(
        lowering(&opts, None).shape.get("p2p_sum", "threads"),
        Some(256)
    );
}
