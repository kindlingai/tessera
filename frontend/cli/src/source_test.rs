use crate::source::{Origin, split};

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
