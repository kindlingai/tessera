use crate::{ResponseEvent, ResponseParser};
use serde_json::json;

const GEMMA4_CFG: &str = r#"{
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
    for piece in pieces {
        stream.push(piece, &mut events);
    }
    stream.finish(&mut events);
    assert_eq!(
        events.first(),
        Some(&ResponseEvent::Reasoning("plan".into()))
    );
    assert!(events.contains(&ResponseEvent::Content("Hi ".into())));
    let call = events.iter().find_map(|e| match e {
        ResponseEvent::ToolCall(tc) => Some(tc),
        _ => None,
    });
    let call = call.expect("tool call event");
    assert_eq!(call.function.name, "get_temp");
    assert_eq!(call.function.arguments, json!({"loc": "Paris"}));
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
fn glm53_recipes_parse_tags() {
    let raw = "<think>plan</think>Checking.<tool_call>f<arg_key>city</arg_key>\
<arg_value>Paris, FR</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
    for name in ["glm53-moe", "glm53-whole"] {
        let path = format!(
            "{}/../../examples/recipes/{name}/recipe.toml",
            env!("CARGO_MANIFEST_DIR")
        );
        let recipe: serde_json::Value =
            toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let config = recipe["config"]["tokenizer_config.json"].to_string();
        let parser = ResponseParser::from_config(&config).unwrap().unwrap();
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
}
