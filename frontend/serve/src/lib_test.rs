use crate::Slot;
use crate::mocks::{MockGenerator, call, json, loaded};
use serde_json::json;
use std::sync::{Arc, OnceLock};

#[test]
fn health_waits() {
    let empty: Slot = Arc::new(OnceLock::new());
    let (status, loading) = json(&empty, "/v1/health", None);
    assert_eq!(status, 503);
    assert_eq!(loading["error"]["type"], "unavailable_error");
    let (status, _) = call(&empty, "/v1/completions", Some(json!({"prompt": "a"})));
    assert_eq!(status, 503);
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    assert_eq!(json(&slot, "/ping", None).1, json!({"status": "ok"}));
    let (_, version) = json(&slot, "/version", None);
    assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));
}

#[test]
fn small_routes() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let (_, models) = json(&slot, "/v1/models", None);
    assert_eq!(
        (
            &models["data"][0]["id"],
            &models["data"][0]["max_model_len"]
        ),
        (&json!("m"), &json!(16))
    );
    let (_, tokens) = json(&slot, "/tokenize", Some(json!({"prompt": "ab"})));
    assert_eq!(
        tokens,
        json!({"count": 2, "max_model_len": 16, "tokens": [97, 98]})
    );
    let (_, text) = json(&slot, "/detokenize", Some(json!({"tokens": [97, 98]})));
    assert_eq!(text["prompt"], "ab");
    let (_, embedded) = json(
        &slot,
        "/v1/embeddings",
        Some(json!({"input": ["a", "bcd"]})),
    );
    assert_eq!(embedded["data"][1]["embedding"], json!([3.0]));
    assert_eq!(embedded["usage"]["prompt_tokens"], 2);
}

#[test]
fn metrics_text() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let (status, text) = call(&slot, "/metrics", None);
    let has = |line: &str| text.lines().any(|l| l == line);
    assert_eq!(status, 200);
    assert!(has("# TYPE vllm:kv_cache_size_tokens gauge"));
    assert!(has("# TYPE tessera:requests_running gauge"));
}
