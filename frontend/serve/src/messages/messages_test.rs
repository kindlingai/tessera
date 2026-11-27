use super::{System, Turn, convert};
use crate::Slot;
use crate::mocks::{MockGenerator, call, json, loaded, loaded_with};
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use tessera_bundle::ResponseParser;

fn tool_parser() -> ResponseParser {
    let cfg = r#"{"response_template":{"fields":{
      "thinking":{"open":"<think>","close":"</think>","content":"text"},
      "tool_calls":{"open_pattern":"<call:(?P<name>\\w+)>","close":"</call>",
        "content":"json","transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}
    }}}"#;
    ResponseParser::from_config(cfg).unwrap().unwrap()
}

#[test]
fn messages_whole_and_errors() {
    let empty: Slot = Arc::new(OnceLock::new());
    let (status, body) = json(&empty, "/v1/messages", Some(json!({})));
    assert_eq!(
        (status, body["error"]["type"].as_str()),
        (503, Some("overloaded_error"))
    );
    let mock = Arc::new(MockGenerator::default());
    let slot = loaded(&mock, true);
    let req = json!({
        "model": "m",
        "max_tokens": 2,
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "h"}]}]
    });
    let (status, res) = json(&slot, "/v1/messages", Some(req));
    assert_eq!(
        (status, res["type"].as_str(), res["stop_reason"].as_str()),
        (200, Some("message"), Some("end_turn"))
    );
    assert_eq!(res["content"][0], json!({"type": "text", "text": "hello"}));
    assert_eq!(res["usage"]["output_tokens"], 2);
    let bad =
        json!({"model": "other", "max_tokens": 2, "messages": [{"role": "user", "content": "a"}]});
    assert_eq!(json(&slot, "/v1/messages", Some(bad)).0, 404);
    let zero =
        json!({"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "a"}]});
    assert_eq!(json(&slot, "/v1/messages", Some(zero)).0, 400);
}

#[test]
fn messages_tools_and_stream() {
    let turns: Vec<Turn> = serde_json::from_value(json!([
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "p"},
            {"type": "tool_use", "id": "t1", "name": "go", "input": {"x": 0}}
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    ]))
    .unwrap();
    let converted = convert(Some(System::Text("s".into())), turns).unwrap();
    assert_eq!(converted.len(), 3);
    assert_eq!(
        (
            converted[1].reasoning.as_deref(),
            converted[2].role.as_str()
        ),
        (Some("p"), "tool")
    );
    let mock = Arc::new(MockGenerator {
        pieces: vec![
            "<think>plan</think>".into(),
            "<call:go>{\"x\":1}</call>".into(),
        ],
        ..MockGenerator::default()
    });
    let slot = loaded_with(&mock, true, Some(tool_parser()));
    let body = json!({
        "model": "m",
        "max_tokens": 2,
        "tools": [{"name": "go", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "go"},
        "thinking": {"type": "enabled"},
        "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    });
    let (status, whole) = json(&slot, "/v1/messages", Some(body.clone()));
    assert_eq!(
        (status, whole["stop_reason"].as_str()),
        (200, Some("tool_use"))
    );
    assert_eq!(
        whole["content"][0],
        json!({"type": "thinking", "thinking": "plan"})
    );
    assert_eq!(
        (
            whole["content"][1]["name"].as_str(),
            &whole["content"][1]["input"]
        ),
        (Some("go"), &json!({"x": 1}))
    );
    let mut streamed = body;
    streamed["stream"] = Value::Bool(true);
    let (status, sse) = call(&slot, "/v1/messages", Some(streamed));
    assert_eq!(status, 200);
    for ev in [
        "event: message_start\n",
        "\"type\":\"thinking_delta\"",
        "\"type\":\"input_json_delta\"",
        "\"stop_reason\":\"tool_use\"",
        "event: message_stop\n",
    ] {
        assert!(sse.contains(ev), "missing {ev} in {sse}");
    }
}

#[test]
fn messages_stream_tool_deltas() {
    let cfg = r#"{"response_template":{"fields":{
      "tool_calls":{"open_pattern":"<tool_call>\\s*(?P<name>[^<\\s]+)","close":"</tool_call>",
        "content":"tags","content_args":{"pattern":"<arg_key>(?P<key>[\\s\\S]*?)</arg_key>\\s*<arg_value>(?P<value>[\\s\\S]*?)</arg_value>"},
        "transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}
    }}}"#;
    let parser = ResponseParser::from_config(cfg).unwrap().unwrap();
    let mock = Arc::new(MockGenerator {
        pieces: vec![
            "<tool_call>save\n<arg_key>body</arg_key><arg_value>".into(),
            "alpha ".into(),
            "beta".into(),
            "</arg_value><arg_key>ok</arg_key><arg_value>true</arg_value></tool_call>".into(),
        ],
        ..MockGenerator::default()
    });
    let slot = loaded_with(&mock, true, Some(parser));
    let body = json!({
        "model": "m",
        "max_tokens": 2,
        "stream": true,
        "tools": [{"name": "save", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "go"}]
    });
    let (status, sse) = call(&slot, "/v1/messages", Some(body));
    assert_eq!(status, 200);
    let mut partial = String::new();
    let mut deltas = 0usize;
    let mut started = false;
    let mut stopped = false;
    for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        let v: Value = serde_json::from_str(line).unwrap();
        match v["type"].as_str() {
            Some("content_block_start") if v["content_block"]["type"] == "tool_use" => {
                assert_eq!(v["content_block"]["name"], "save");
                started = true;
            }
            Some("content_block_delta") if v["delta"]["type"] == "input_json_delta" => {
                partial.push_str(v["delta"]["partial_json"].as_str().unwrap());
                deltas += 1;
            }
            Some("content_block_stop") if started => stopped = true,
            _ => {}
        }
    }
    assert!(started && stopped);
    assert!(deltas >= 3, "expected incremental input_json_delta: {sse}");
    assert_eq!(
        serde_json::from_str::<Value>(&partial).unwrap(),
        json!({"body": "alpha beta", "ok": true})
    );
}
