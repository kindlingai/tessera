use crate::Slot;
use crate::mocks::{MockGenerator, call, json, loaded};
use serde_json::json;
use std::sync::{Arc, OnceLock};

fn auth_status(method: &str, path: &str, hdr: Option<(&str, &str)>) -> Option<u16> {
    let headers = hdr.into_iter().map(|(a, b)| (a.into(), b.into())).collect();
    let req = crate::http::Request {
        method: method.into(),
        path: path.into(),
        headers,
        body: Vec::new(),
    };
    crate::start::check_auth("k", &req).map(|r| r.status)
}

#[test]
fn api_key_auth() {
    let mut o = crate::start::Options::default();
    assert!(o.flag("--api-key", String::new()).is_none());
    assert!(o.flag("--api-key", "k".into()).is_some());
    assert_eq!(auth_status("GET", "/health", None), None);
    assert_eq!(auth_status("POST", "/v1/chat/completions", None), Some(401));
    assert_eq!(
        auth_status("POST", "/v1/messages", Some(("x-api-key", "x"))),
        Some(401)
    );
    assert_eq!(
        auth_status(
            "POST",
            "/v1/chat/completions",
            Some(("authorization", "Bearer k"))
        ),
        None
    );
    assert_eq!(
        auth_status("POST", "/v1/messages", Some(("x-api-key", "k"))),
        None
    );
}

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
    assert!(has("# TYPE tessera:http_connections_open gauge"));
    assert!(has("# TYPE tessera:completion_requests_total counter"));
}

#[test]
fn retrieve_model() {
    let empty: Slot = Arc::new(OnceLock::new());
    assert_eq!(json(&empty, "/v1/models/m", None).0, 503);
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let (status, found) = json(&slot, "/v1/models/m", None);
    assert_eq!(status, 200);
    assert_eq!(found["id"], "m");
    assert_eq!(found["object"], "model");
    assert_eq!(found["max_model_len"], 16);
    let (status, decoded) = json(&slot, "/v1/models/%6D", None);
    assert_eq!(status, 200);
    assert_eq!(decoded["id"], "m");
    let (status, missing) = json(&slot, "/v1/models/other", None);
    assert_eq!(status, 404);
    assert_eq!(missing["error"]["type"], "NotFoundError");
    assert_eq!(missing["error"]["param"], "model");
}

#[test]
fn warm_rows_buckets() {
    use crate::task::warm_rows;
    assert_eq!(warm_rows(1, None), vec![1]);
    assert_eq!(warm_rows(8, Some(4)), vec![1, 2, 4, 8]);
    assert_eq!(warm_rows(6, Some(12)), vec![1, 2, 4, 6, 12]);
}
