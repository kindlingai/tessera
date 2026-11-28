use super::stream::stream_test::collect_calls;
use crate::{ResponseEvent, ResponseParser};
use serde_json::json;

pub(super) const GEMMA4_CFG: &str = r#"{
  "response_template": {
    "fields": {
      "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"], "content": "text"},
      "thinking": {"open": "<|channel>thought\n", "close": "<channel|>", "content": "text"},
      "tool_calls": {
        "open_pattern": "<\\|tool_call>call:(?P<name>\\w+)",
        "close": "<tool_call|>",
        "content": "json",
        "content_args": {"string_delims": [["<|\"|>", "<|\"|>"]], "unquoted_keys": true},
        "repeats": true,
        "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}
      }
    }
  }
}"#;

const JSON_CFG: &str = r#"{
  "response_template": {
    "fields": {
      "content": {"close": "<|im_end|>", "content": "text"},
      "thinking": {"open": "<think>\n", "close": "</think>", "content": "text"},
      "tool_calls": {"open": "<tool_call>\n", "close": "</tool_call>", "content": "json"}
    }
  }
}"#;

#[test]
fn parses_delimited_calls() {
    let parser = ResponseParser::from_config(GEMMA4_CFG).unwrap().unwrap();
    assert!(parser.supports_tools());
    let raw = "<|channel>thought\nNeed weather.\n<channel|>\
<|tool_call>call:weather{city:<|\"|>Tokyo, \"JP\"<|\"|>,flags:[true,null,3],opts:{a:1}}<tool_call|>\
<|tool_call>call:ping{}<tool_call|><|tool_response>ignored";
    let parsed = parser.parse(raw, true, false);
    assert_eq!(parsed.reasoning, "Need weather.\n");
    assert_eq!(parsed.content, "");
    assert_eq!(parsed.tool_calls.len(), 2);
    assert_eq!(parsed.tool_calls[0].function.name, "weather");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        json!({"city": "Tokyo, \"JP\"", "flags": [true, null, 3], "opts": {"a": 1}})
    );
    assert_eq!(parsed.tool_calls[1].function.name, "ping");
    assert_eq!(parsed.tool_calls[1].function.arguments, json!({}));
    assert!(parser.starts_thinking("...<tool_response|><|channel>thought\n"));
    let cont = parser.parse("done<channel|>15C", true, true);
    assert_eq!(
        (cont.reasoning.as_str(), cont.content.as_str()),
        ("done", "15C")
    );
}

#[test]
fn parses_standard_json_template() {
    let parser = ResponseParser::from_config(JSON_CFG).unwrap().unwrap();
    let raw = "<think>\nok</think><tool_call>\n{\"name\":\"sum\",\"arguments\":{\"x\":2}}</tool_call><|im_end|>";
    let parsed = parser.parse(raw, true, false);
    assert_eq!(parsed.reasoning, "ok");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "sum");
    assert_eq!(parsed.tool_calls[0].function.arguments, json!({"x": 2}));
    assert_eq!(
        parser.parse("<think>\n</think>hi", false, false).content,
        "hi"
    );
}

#[test]
fn strips_pattern_thinking_opener() {
    let cfg = r#"{"response_template": {"fields": {
      "thinking": {"open_pattern": "<think(?: \\w+)?>\n", "close": "</think>", "content": "text"}
    }}}"#;
    let parser = ResponseParser::from_config(cfg).unwrap().unwrap();
    let mut stream = parser.stream(false, false);
    let mut events = Vec::new();
    for piece in ["a<thi", "nk deep>", "\n", "plan", "</think>b<think>", "\n"] {
        stream.push(piece, &mut events);
    }
    stream.finish(&mut events);
    let say = |s: &str| ResponseEvent::Content(s.into());
    let plan = ResponseEvent::Reasoning("plan".into());
    assert_eq!(events, [say("a"), plan, say("b")]);
    let odd = parser.parse("<think!</think>x", false, false);
    assert_eq!(
        (odd.reasoning.as_str(), odd.content.as_str()),
        ("", "<think!</think>x")
    );
}

#[test]
fn deepseek_family_default() {
    let chat = "{% for m in messages %}{{ m['content'] or '' }}{% for tool in m['tool_calls'] or [] %}\
{{ '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + tool['function']['arguments'] }}\
{% endfor %}{% endfor %}";
    let mut template = crate::Template::new(chat.into()).unwrap();
    let parser = ResponseParser::builtin(&template).unwrap().unwrap();
    template
        .with_tool_prompt(parser.tool_prompt().unwrap())
        .unwrap();
    let call = crate::ToolCall {
        function: crate::FunctionCall {
            name: "ping".into(),
            arguments: json!({"a": 1}),
        },
        kind: "function".into(),
        id: None,
    };
    let reply = crate::Message {
        role: "assistant".into(),
        tool_calls: Some(vec![call]),
        ..Default::default()
    };
    let tools = [json!({"type": "function", "function": {"name": "ping", "parameters": {}}})];
    let options = crate::RenderOptions {
        tools: &tools,
        ..Default::default()
    };
    let text = template.render_options(&[reply], &options).unwrap();
    assert!(text.starts_with("## Tools\n") && text.contains("\n### ping\n"));
    assert!(text.ends_with("<｜tool▁call▁begin｜>function<｜tool▁sep｜>ping{\"a\":1}"));
    let pieces = [
        "Sure.",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "function<｜tool▁sep｜>get_",
        "weather\n```json\n{\"city\": \"Paris\"}\n```",
        "<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>ping\n```json\n{}\n```<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    ];
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    for piece in pieces {
        stream.push(piece, &mut events);
    }
    stream.finish(&mut events);
    assert_eq!(
        events.first(),
        Some(&ResponseEvent::Content("Sure.".into()))
    );
    let calls = collect_calls(&events);
    assert_eq!(
        calls,
        [
            ("get_weather".into(), json!({"city": "Paris"})),
            ("ping".into(), json!({})),
        ]
    );
}

#[test]
fn picks_builtin_format() {
    let cases = [
        ("{{ '<|tool_call>call:' + name }}", Some("gemma4")),
        ("{{ '<｜tool▁call▁begin｜>' + name }}", Some("deepseek")),
        ("{{ '<|tool_call|>' }}", Some("granite")),
        ("{{ '<tool_call>\\n<function=' + name }}", Some("qwen35")),
        ("{{ '<tool_call>' + name }}<arg_key>", Some("glm53")),
        ("{{ '<tool_call>[' }}", Some("smollm2")),
        ("{{ '<tool_call>\\n{\"name\": ' }}", Some("qwen3")),
        ("no tools", None),
    ];
    for (source, want) in cases {
        let chat = crate::Template::new(source.into()).unwrap();
        let got = super::formats::pick(&chat).unwrap().map(|(name, _)| name);
        assert_eq!(got, want, "{source}");
    }
}

#[test]
fn parses_arg_key_calls() {
    let raw = "<think>plan</think>Checking.<tool_call>f<arg_key>city</arg_key>\
<arg_value>Paris, FR</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
    let cfg = format!(r#"{{"response_template": {}}}"#, include_str!("glm53.json"));
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let parsed = parser.parse(raw, true, false);
    assert_eq!(
        (parsed.reasoning.as_str(), parsed.content.as_str()),
        ("plan", "Checking.")
    );
    assert_eq!(parsed.tool_calls[0].function.name, "f");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        json!({"city": "Paris, FR", "days": 3})
    );
}

#[test]
fn parses_tagged_parameter_calls() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("qwen35.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let raw = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call><|im_end|>";
    let parsed = parser.parse(raw, true, false);
    assert_eq!(parsed.content, "Let me check.\n\n");
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    assert_eq!(
        parsed.tool_calls[0].function.arguments,
        json!({"city": "New York", "days": 3, "note": "line one\nline two"})
    );
}

#[test]
fn parses_open_blocks() {
    let cfg = format!(
        r#"{{"response_template": {}}}"#,
        include_str!("granite.json")
    );
    let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
    let calls = r#"[{"name": "weather", "arguments": {"city": "Paris"}}, {"name": "ping", "arguments": {}}]"#;
    let mut stream = parser.stream(true, false);
    let mut events = Vec::new();
    stream.push("<tool", &mut events);
    stream.push(&format!("_call>{calls}"), &mut events);
    assert!(events.is_empty());
    stream.finish(&mut events);
    let parsed: Vec<_> = events
        .into_iter()
        .filter_map(|e| match e {
            ResponseEvent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].function.arguments, json!({"city": "Paris"}));
    let turn = crate::Message {
        role: "assistant".into(),
        tool_calls: Some(parsed),
        ..Default::default()
    };
    let inlined = parser.inline_calls(&[turn]);
    let text = inlined[0].content.as_deref().unwrap();
    let round = parser.parse(text, true, false);
    assert_eq!(round.tool_calls.len(), 2);
    assert_eq!(round.tool_calls[1].function.name, "ping");
}

#[test]
fn call_prefix_round_trips() {
    let cases = [
        (
            "glm53",
            "<tool_call><arg_key>",
            "\n<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
        ),
        ("qwen3", "<tool_call>", "{\"x\": 1}}</tool_call>"),
        (
            "qwen35",
            "<tool_call><function=",
            "\n<parameter=x>1</parameter></function></tool_call>",
        ),
        ("gemma4", "<|tool_call>call:", "{x:1}<tool_call|>"),
        (
            "deepseek",
            "<｜tool▁call▁begin｜>",
            "{\"x\": 1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        ),
        ("granite", "<|tool_call|>", "{\"x\": 1}}"),
        ("smollm2", "<tool_call>[", "{\"x\": 1}}</tool_call>"),
    ];
    for (name, hint, suffix) in cases {
        let tmpl = crate::Template::new(hint.into()).unwrap();
        let parser = ResponseParser::builtin(&tmpl).unwrap().expect(name);
        let req = parser.call_prefix(None).expect(name);
        assert!(!req.is_empty(), "{name} required prefix empty");
        let named = parser.call_prefix(Some("get_weather")).expect(name);
        let raw = format!("{named}{suffix}");
        let parsed = parser.parse(&raw, true, false);
        assert_eq!(parsed.tool_calls.len(), 1, "{name}: raw={raw:?}");
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather", "{name}");
    }
}
