use super::super::response_test::GEMMA4_CFG;
use crate::{ResponseEvent, ResponseParser};
use serde_json::{Value, json};

pub(in crate::response) fn collect_calls(events: &[ResponseEvent]) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut open: Option<(String, String)> = None;
    for ev in events {
        match ev {
            ResponseEvent::ToolCall(tc) => {
                out.push((tc.function.name.clone(), tc.function.arguments.clone()));
            }
            ResponseEvent::ToolCallStart { name, .. } => {
                open = Some((name.clone(), String::new()));
            }
            ResponseEvent::ToolCallArgs(frag) => {
                if let Some((_, raw)) = &mut open {
                    raw.push_str(frag);
                }
            }
            ResponseEvent::ToolCallEnd => {
                if let Some((name, raw)) = open.take() {
                    let val: Value = serde_json::from_str(&raw)
                        .unwrap_or_else(|e| panic!("invalid JSON {raw:?}: {e}"));
                    out.push((name, val));
                }
            }
            ResponseEvent::Content(_) | ResponseEvent::Reasoning(_) => {}
        }
    }
    out
}

#[test]
fn streams_split_tokens() {
    let parser = ResponseParser::from_config(GEMMA4_CFG).unwrap().unwrap();
    let pieces = [
        "<|chan",
        "nel>thought\n",
        "plan",
        "<channel|>",
        "Hi ",
        "<|tool_",
        "call>call:get_",
        "temp{loc:<|\"|>",
        "Paris",
        "<|\"|>}<tool_",
        "call|>",
    ];
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    let mut saw_early_args = false;
    for (i, piece) in pieces.iter().enumerate() {
        stream.push(piece, &mut events);
        if i == 8 {
            saw_early_args = events
                .iter()
                .any(|e| matches!(e, ResponseEvent::ToolCallArgs(_)));
        }
    }
    stream.finish(&mut events);
    assert!(saw_early_args, "expected args before close tag: {events:?}");
    assert_eq!(
        events.first(),
        Some(&ResponseEvent::Reasoning("plan".into()))
    );
    assert!(events.contains(&ResponseEvent::Content("Hi ".into())));
    assert_eq!(
        collect_calls(&events),
        [("get_temp".into(), json!({"loc": "Paris"}))]
    );
}

#[test]
fn streams_glm53_tags() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("../glm53.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let pieces = [
        "<tool_call>submit_",
        "result\n",
        "<arg_key>text</arg_key><arg_value>",
        "hello ",
        "world\n\"quoted\"",
        "</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>",
        "<arg_key>flag</arg_key><arg_value>true</arg_value>",
        "<arg_key>note</arg_key><arg_value>",
        "true",
        " story</arg_value><arg_key>addr</arg_key><arg_value>",
        "123",
        " Main</arg_value><arg_key>meta</arg_key><arg_value>{\"k\":[1,2]}</arg_value>",
        "</tool_call><tool_call>ping</tool_call>",
    ];
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    let mut args_before_close = 0usize;
    let mut started_at = None;
    for (i, piece) in pieces.iter().enumerate() {
        let before = events.len();
        stream.push(piece, &mut events);
        if started_at.is_none()
            && events[before..]
                .iter()
                .any(|e| matches!(e, ResponseEvent::ToolCallStart { .. }))
        {
            started_at = Some(i);
        }
        if i <= 4 {
            args_before_close += events[before..]
                .iter()
                .filter(|e| matches!(e, ResponseEvent::ToolCallArgs(_)))
                .count();
        }
    }
    stream.finish(&mut events);
    assert_eq!(started_at, Some(1));
    assert!(
        args_before_close >= 3,
        "expected live text chunks: {events:?}"
    );
    let want = json!({
        "text": "hello world\n\"quoted\"",
        "days": 3,
        "flag": true,
        "note": "true story",
        "addr": "123 Main",
        "meta": {"k": [1, 2]},
    });
    assert_eq!(
        collect_calls(&events),
        [
            ("submit_result".into(), want.clone()),
            ("ping".into(), json!({})),
        ]
    );
    let parsed = parser.parse(&pieces.concat(), true, false);
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].function.arguments, want);
    assert_eq!(parsed.tool_calls[1].function.arguments, json!({}));
}

#[test]
fn streams_qwen35_tags() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("../qwen35.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let pieces = [
        "<tool_call>\n<function=write",
        "_file>\n",
        "<parameter=body>\n",
        "line one\n",
        "line two\n",
        "</para",
        "meter>\n<parameter=count>\n4\n</parameter>\n</function>\n</tool_call>",
    ];
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    let mut early_args = 0usize;
    for (i, piece) in pieces.iter().enumerate() {
        let before = events.len();
        stream.push(piece, &mut events);
        if i <= 4 {
            early_args += events[before..]
                .iter()
                .filter(|e| matches!(e, ResponseEvent::ToolCallArgs(_)))
                .count();
        }
    }
    stream.finish(&mut events);
    assert!(early_args >= 3, "expected incremental chunks: {events:?}");
    assert_eq!(
        collect_calls(&events),
        [(
            "write_file".into(),
            json!({"body": "line one\nline two", "count": 4}),
        )]
    );
}

#[test]
fn streams_qwen3_json() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("../qwen3.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let pieces = [
        "<tool_call>\n{\"name\": \"edit\", ",
        "\"arguments\": {\"path\": \"a.rs\", ",
        "\"body\": \"hello\"}",
        "}</tool_call>",
    ];
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    let mut early_args = 0usize;
    for (i, piece) in pieces.iter().enumerate() {
        let before = events.len();
        stream.push(piece, &mut events);
        if i <= 2 {
            early_args += events[before..]
                .iter()
                .filter(|e| matches!(e, ResponseEvent::ToolCallArgs(_)))
                .count();
        }
    }
    stream.finish(&mut events);
    assert!(
        early_args >= 2,
        "expected incremental JSON chunks: {events:?}"
    );
    assert_eq!(
        collect_calls(&events),
        [("edit".into(), json!({"path": "a.rs", "body": "hello"}))]
    );
    let rev = "<tool_call>\n{\"arguments\": {\"x\": 1}, \"name\": \"rev\"}</tool_call>";
    let parsed = parser.parse(rev, true, false);
    assert_eq!(parsed.tool_calls[0].function.name, "rev");
    assert_eq!(parsed.tool_calls[0].function.arguments, json!({"x": 1}));
}

#[test]
fn streams_truncated_call() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("../glm53.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    stream.push(
        "<tool_call>say\n<arg_key>text</arg_key><arg_value>partial",
        &mut events,
    );
    stream.finish(&mut events);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, ResponseEvent::ToolCallStart { .. }))
    );
    assert_eq!(events.last(), Some(&ResponseEvent::ToolCallEnd));
}
