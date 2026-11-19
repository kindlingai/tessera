use crate::source::{Origin, Source, split};

fn origin(value: &str) -> Option<Origin> {
    let given = ["--model".to_string(), value.to_string()];
    split(&given).ok()?.0
}

#[test]
fn parses_hub_models() {
    let hub = |v| match origin(v) {
        Some(Origin::Hub(spec)) => Some(spec),
        _ => None,
    };
    assert_eq!(hub("hf:Qwen/Qwen3-0.6B").unwrap(), "Qwen/Qwen3-0.6B");
    assert_eq!(hub("hf:a/b@main").unwrap(), "a/b@main");
    assert!(matches!(origin("models/qwen"), Some(Origin::Model(_))));
    for bad in ["hf:", "hf:a", "hf:a/", "hf:a/b/c", "hf:a/b@"] {
        assert!(origin(bad).is_none(), "{bad}");
    }
}

/// A recipe's module rules ride in the serve options.
#[test]
fn options_carry_rules() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/recipes/glm53-moe"
    );
    let Ok(mut source) = Source::open(Some(Origin::Recipe(dir.into()))) else {
        panic!("the recipe opens");
    };
    source.cache = Some(sensai_testkit::scratch("recipes"));
    let Ok(options) = source.options(&[]) else {
        panic!("the recipe gives options");
    };
    let lower = options.lower.unwrap();
    let names: Vec<&str> = lower.rules.iter().map(|r| r.name).collect();
    assert_eq!(names, ["glm53_moe"]);
    assert_eq!(lower.favor.as_deref(), Some("glm53_moe"));
    assert!(options.module.is_some());
}
