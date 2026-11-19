use crate::mocks::{MockGenerator, call, json, loaded};
use serde_json::{Value, json};
use std::sync::Arc;

#[test]
fn completion_whole() {
    let mock = Arc::new(MockGenerator::default());
    let slot = loaded(&mock, false);
    let body = json!({"model": "m", "prompt": "abcd", "stop": "x", "temperature": 0});
    let (status, got) = json(&slot, "/v1/completions", Some(body));
    assert_eq!(status, 200);
    assert_eq!(got["object"], "text_completion");
    assert_eq!(got["choices"][0]["text"], "hello");
    assert_eq!(got["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        got["usage"],
        json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 3}})
    );
    let asked = mock.asked.lock().unwrap().clone().unwrap();
    // The room the 16-token context leaves.
    assert_eq!((asked.max_tokens, asked.stop), (12, vec!["x".to_string()]));
}

#[test]
fn chat_streamed() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let body = json!({
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let (status, text) = call(&slot, "/v1/chat/completions", Some(body));
    assert_eq!(status, 200);
    let data: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let deltas: Vec<&Value> = chunks.iter().map(|c| &c["choices"][0]["delta"]).collect();
    assert_eq!(
        deltas[..3],
        [
            &json!({"role": "assistant", "content": ""}),
            &json!({"content": "he"}),
            &json!({"content": "llo"})
        ]
    );
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[3]["choices"], json!([]));
    assert_eq!(chunks[3]["usage"]["total_tokens"], 9);
    let objects: Vec<&Value> = chunks.iter().map(|c| &c["object"]).collect();
    assert!(objects.iter().all(|o| **o == "chat.completion.chunk"));
}

#[test]
fn refusals() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let cases = [
        (json!({"model": "other", "prompt": "a"}), 404, "model"),
        (json!({"prompt": "a", "tools": [{}]}), 400, "tools"),
        (
            json!({"prompt": "a", "temperature": 0.7}),
            400,
            "temperature",
        ),
        (
            json!({"prompt": "abcd", "max_tokens": 13}),
            400,
            "max_tokens",
        ),
    ];
    for (body, status, param) in cases {
        let (got, error) = json(&slot, "/v1/completions", Some(body));
        assert_eq!((got, &error["error"]["param"]), (status, &json!(param)));
        assert_eq!(error["error"]["code"], status);
    }
}

#[test]
fn unknown_fields() {
    let said = json!({"role": "user", "content": "hi", "name": "x", "reasoning": "r"});
    let body = json!({"messages": [said]});
    for (strict, status) in [(false, 200), (true, 400)] {
        let slot = loaded(&Arc::new(MockGenerator::default()), strict);
        let (got, answer) = json(&slot, "/v1/chat/completions", Some(body.clone()));
        assert_eq!(got, status);
        assert_eq!(strict, answer["error"]["param"] == "messages[0].name");
    }
}
