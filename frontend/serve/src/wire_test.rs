use crate::mocks::{MockGenerator, call, json, loaded, loaded_with};
use crate::wire::ResponseFormat;
use serde_json::json;
use std::sync::Arc;
use tessera_bundle::ResponseParser;
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

#[test]
fn thinking_keys() {
    let mock = Arc::new(MockGenerator::default());
    let slot = loaded(&mock, true);
    let body = json!({"prompt": "a", "thinking_presence_penalty": 0.25,
        "thinking_frequency_penalty": 0.5, "think_end_boost": "start=8|0.5x,per1k=2,max=10",
        "sequence_penalty": "answer=1:1.5:8:10"});
    assert_eq!(json(&slot, "/v1/completions", Some(body)).0, 200);
    let asked = mock.asked.lock().unwrap().clone().unwrap();
    let thinking = asked.thinking;
    assert!(thinking.frequency == 0.5 && thinking.boost.is_some());
    assert!(asked.sequence.answer.is_some());
    let bad = [
        ("thinking_frequency_penalty", json!(-2.5)),
        ("think_end_boost", json!(true)),
        ("think_end_boost", json!("start=1,per1k=1,max=101")),
        ("sequence_penalty", json!(true)),
        ("sequence_penalty", json!("guard=1:1.5:8:10")),
    ];
    for (key, value) in bad {
        let (status, error) = json(
            &slot,
            "/v1/completions",
            Some(json!({"prompt": "a", key: value})),
        );
        assert_eq!((status, &error["error"]["param"]), (400, &json!(key)));
    }
}

#[test]
fn reasoning_effort_forwards() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "low",
    });
    assert_eq!(
        json(&slot, "/v1/chat/completions", Some(body.clone())).0,
        200
    );
    assert_eq!(json(&slot, "/tokenize", Some(body)).0, 200);
}

#[test]
fn tool_choice_validation() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let tool = json!({"type": "function", "function": {"name": "ping"}});
    let bad_cases = [
        json!({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "bogus"}),
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "other"}},
        }),
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": ""}},
        }),
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
            "tool_choice": "required",
            "response_format": {"type": "json_object"},
        }),
    ];
    for body in bad_cases {
        let (status, error) = json(&slot, "/v1/chat/completions", Some(body));
        assert_eq!(
            (status, &error["error"]["param"]),
            (400, &json!("tool_choice"))
        );
    }
    // Model without tool parser still rejects forced tool_choice on tools.
    let req_body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "ping"}},
    });
    let (status, error) = json(&slot, "/v1/chat/completions", Some(req_body));
    assert_eq!((status, &error["error"]["param"]), (400, &json!("tools")));
}

#[test]
fn caps_parallel_calls() {
    let cfg = r#"{"response_template":{"fields":{"tool_calls":{
      "open_pattern":"<tool_call>\\s*(?P<name>[^<\\s]+)","close":"</tool_call>","content":"tags",
      "content_args":{"pattern":"<arg_key>(?P<key>[\\s\\S]*?)</arg_key>\\s*<arg_value>(?P<value>[\\s\\S]*?)</arg_value>"},
      "transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}}}}"#;
    let parser = ResponseParser::from_config(cfg).unwrap().unwrap();
    let mock = Arc::new(MockGenerator {
        pieces: vec![
            "<tool_call>ping<arg_key>a</arg_key><arg_value>one</arg_value></tool_call>".into(),
            "<tool_call>ping<arg_key>a</arg_key><arg_value>two</arg_value></tool_call>".into(),
        ],
        ..MockGenerator::default()
    });
    let slot = loaded_with(&mock, true, Some(parser));
    let tool = json!({"type": "function", "function": {"name": "ping"}});
    let mut body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool],
        "parallel_tool_calls": false,
    });
    let (status, got) = json(&slot, "/v1/chat/completions", Some(body.clone()));
    assert_eq!(status, 200);
    let calls = got["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["arguments"], "{\"a\":\"one\"}");
    body["stream"] = json!(true);
    let (s_status, sse) = call(&slot, "/v1/chat/completions", Some(body));
    assert_eq!(s_status, 200);
    assert!(sse.contains(r#"\"one\""#) && !sse.contains(r#"\"two\""#));
}

#[test]
fn openai_neutral_fields() {
    let slot = loaded(&Arc::new(MockGenerator::default()), true);
    let ok = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "store": false,
        "metadata": {},
        "service_tier": "auto",
        "prompt_cache_key": "session-1",
        "safety_identifier": "acct-1",
        "top_logprobs": null,
        "modalities": ["text"],
    });
    assert_eq!(json(&slot, "/v1/chat/completions", Some(ok)).0, 200);
    let default_tier = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "default",
    });
    assert_eq!(
        json(&slot, "/v1/chat/completions", Some(default_tier)).0,
        200
    );
    let bad = [
        ("store", json!(true)),
        ("metadata", json!({"k": "v"})),
        ("service_tier", json!("priority")),
        ("top_logprobs", json!(5)),
        ("modalities", json!(["text", "audio"])),
    ];
    for (key, value) in bad {
        let body = json!({"messages": [{"role": "user", "content": "hi"}], key: value});
        let (status, error) = json(&slot, "/v1/chat/completions", Some(body));
        assert_eq!((status, &error["error"]["param"]), (400, &json!(key)));
    }
}
