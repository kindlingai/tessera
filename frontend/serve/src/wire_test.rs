use crate::mocks::{MockGenerator, json, loaded};
use crate::wire::ResponseFormat;
use serde_json::json;
use std::sync::Arc;
use tessera_tasks::api::Format;

#[test]
fn formats_parse() {
    let schema = json!({"type": "object"});
    let named = json!({"name": "n", "strict": true, "schema": schema});
    let cases = [
        (json!({"type": "text"}), None),
        (json!({"type": "json_object"}), Some(Format::Object)),
        (
            json!({"type": "json_schema", "json_schema": named}),
            Some(Format::Schema(schema.clone())),
        ),
    ];
    for (given, want) in cases {
        let parsed: ResponseFormat = serde_json::from_value(given).unwrap();
        assert_eq!(parsed.format(), want);
    }
}

#[test]
fn bad_schemas_refused() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let format = json!({"type": "json_schema", "json_schema": {"schema": true}});
    let body = json!({"prompt": "a", "response_format": format});
    let (status, error) = json(&slot, "/v1/completions", Some(body));
    assert_eq!(
        (status, &error["error"]["param"]),
        (400, &json!("response_format"))
    );
}

#[test]
fn tool_wire_checks() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let tool = json!({"type": "function", "function": {"name": "ping"}});
    let cases = [
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "tools": [tool]}),
            "tools",
        ),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "required"}),
            "tool_choice",
        ),
        (
            json!({"messages": [{"role": "assistant", "tool_calls": [
                {"id": "1", "function": {"name": "ping", "arguments": "bad"}}
            ]}]}),
            "messages",
        ),
    ];
    for (body, param) in cases {
        let (status, error) = json(&slot, "/v1/chat/completions", Some(body));
        assert_eq!((status, &error["error"]["param"]), (400, &json!(param)));
    }
    let none_body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "none",
        "tools": [tool],
    });
    assert_eq!(json(&slot, "/v1/chat/completions", Some(none_body)).0, 200);
    let history = json!({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "ping", "arguments": "{\"x\":1}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "ping", "content": "ok"}
    ]});
    assert_eq!(json(&slot, "/tokenize", Some(history)).0, 200);
}

#[test]
fn clear_thinking_forwards() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"clear_thinking": false},
    });
    let (status, got) = json(&slot, "/tokenize", Some(body));
    assert_eq!(status, 200);
    assert_eq!(got["count"], "user:hi|clear=false".len());
}
