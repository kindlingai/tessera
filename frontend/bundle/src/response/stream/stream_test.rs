use super::super::response_test::GEMMA4_CFG;
use super::TagDelims;
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

#[test]
fn streams_split_closers() {
    let cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think>|(?=<tool_call>)","content":"text"},"tool_calls":{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close_pattern":"</tool_call>|(?=<tool_call>)","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?(?:</parameter>|(?=<parameter=)|(?=</function>))"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}},"content":{"close_pattern":"<\\|im_end\\|>","content":"text"}}}}"#;
    let parser = ResponseParser::from_config(cfg).unwrap().unwrap();
    let full = "plan<tool_call>\n<function=echo>\n<parameter=a>one\n<parameter=b>two\n</function>\n<tool_call>\n<function=ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>ok<|im_end|>drop";
    let mut stream = parser.stream(true, true);
    let mut events = Vec::new();
    for ch in full.chars() {
        let mut buf = [0u8; 4];
        stream.push(ch.encode_utf8(&mut buf), &mut events);
    }
    stream.finish(&mut events);
    let reasoning: String = events
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::Reasoning(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    let content: String = events
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::Content(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "plan");
    assert_eq!(content, "ok");
    assert_eq!(
        collect_calls(&events),
        [
            ("echo".into(), json!({"a": "one", "b": "two"})),
            ("ping".into(), json!({"x": 1})),
        ]
    );
}

#[test]
fn streams_regex_and_numbers() {
    let think_cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think\\s*>","content":"text"},"tool_calls":{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close":"</tool_call>","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?</parameter>"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}}}}"#;
    let p = ResponseParser::from_config(think_cfg).unwrap().unwrap();
    let mut s = p.stream(true, true);
    let mut ev = Vec::new();
    for part in [
        "plan</think",
        " ><tool_call><function=echo><parameter=x>1</parameter></function></tool_call>",
    ] {
        s.push(part, &mut ev);
    }
    s.finish(&mut ev);
    let reasoning: String = ev
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::Reasoning(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "plan");
    assert_eq!(collect_calls(&ev), [("echo".into(), json!({"x": 1}))]);

    let end_cfg = r#"{"response_template":{"fields":{"content":{"close_pattern":"<end(?:_turn)?>","content":"text"}}}}"#;
    let end_p = ResponseParser::from_config(end_cfg).unwrap().unwrap();
    let mut es = end_p.stream(true, false);
    let mut eev = Vec::new();
    for part in ["answer<end", "_turn>trailer"] {
        es.push(part, &mut eev);
    }
    es.finish(&mut eev);
    let text: String = eev
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::Content(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "answer");

    let digits = format!("0.{}1", "0".repeat(65));
    let mut ns = p.stream(true, false);
    let mut nev = Vec::new();
    ns.push(
        &format!("<tool_call><function=echo><parameter=x>{digits}"),
        &mut nev,
    );
    ns.push(
        "</parameter><parameter=id>007</parameter></function></tool_call>",
        &mut nev,
    );
    ns.finish(&mut nev);
    let whole = p.parse(
        &format!(
            "<tool_call><function=echo><parameter=x>{digits}</parameter><parameter=id>007</parameter></function></tool_call>"
        ),
        true,
        false,
    );
    assert_eq!(
        collect_calls(&nev),
        [(
            "echo".into(),
            whole.tool_calls[0].function.arguments.clone()
        )]
    );
    assert_eq!(
        whole.tool_calls[0].function.arguments,
        json!({"x": 1e-66, "id": "007"})
    );
    check_tag_inline_flags();
    check_tag_capture_allowlist();
}

fn check_tag_inline_flags() {
    let base = r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter>";
    assert!(TagDelims::from_pattern(&format!("(?s){base}")).is_some());
    for raw in [
        include_str!("../qwen35.json"),
        include_str!("../glm53.json"),
    ] {
        let v: Value = serde_json::from_str(raw).unwrap();
        let pat = v["fields"]["tool_calls"]["content_args"]["pattern"]
            .as_str()
            .unwrap();
        let compiled = if pat.starts_with("(?") {
            pat.to_string()
        } else {
            format!("(?s){pat}")
        };
        assert!(TagDelims::from_pattern(&compiled).is_some(), "{compiled}");
    }
    for prefix in ["(?is)", "(?i)", "(?-s)", "(?U)", "(?x)", "(?m)", "(?s)(?i)"] {
        let pat = format!("{prefix}{base}");
        assert!(TagDelims::from_pattern(&pat).is_none(), "{pat}");
    }
    assert!(
        TagDelims::from_pattern(r"(?s)<parameter=(?P<key>(?i)[^>]+)>(?P<value>.*?)</parameter>")
            .is_none()
    );
    let tag_pat = format!("(?is){base}");
    let cases = [
        (
            [
                "<tool_call><function=ping><PARAMETER=x>1</PARAMETER></function>",
                " All done.",
            ],
            json!({"x": 1}),
        ),
        (
            [
                "<tool_call><function=ping><parameter=x>1</parameter><PARAMETER=y>2</PARAMETER></function>",
                " All done.",
            ],
            json!({"x": 1, "y": 2}),
        ),
    ];
    assert_tag_parity(&tag_pat, &cases);
}

fn check_tag_capture_allowlist() {
    let allowed = [
        r"(?s)<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter>",
        r"<arg_key>(?P<key>[\s\S]*?)</arg_key>\s*<arg_value>(?P<value>[\s\S]*?)</arg_value>",
    ];
    for pat in allowed {
        assert!(TagDelims::from_pattern(pat).is_some(), "{pat}");
    }
    let rejected = [
        r#"(?s)<parameter=(?P<key>[^>]+)>(?P<value>"[^"]*")</parameter>"#,
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>.*)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>.+)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>.+?)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>[^<]+)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>\w+)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>(?P<value>[\s\S]*)</parameter>",
        r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>\w+)>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>[^>\s]+)>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>[^<]+)>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)</key>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>[^>]+)>\s*(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>.+?)>(?P<value>.*?)</parameter>",
        r"(?s)<parameter=(?P<key>[\s\S]+?)>(?P<value>.*?)</parameter>",
        r"(?s)<k=(?P<key>[^>]+)>\s*<v>(?P<value>.*?)</v>",
        r"<arg_key>(?P<key>[\s\S]*?)</arg_key>\s* <arg_value>(?P<value>[\s\S]*?)</arg_value>",
    ];
    for pat in rejected {
        assert!(TagDelims::from_pattern(pat).is_none(), "{pat}");
    }
    let quoted_pat = r#"<parameter=(?P<key>[^>]+)>(?P<value>"[^"]*")</parameter>"#;
    let quoted_cases = [
        (
            [
                r#"<tool_call><function=ping><parameter=x>"a</parameter>b"</parameter></function>"#,
                " All done.",
            ],
            json!({"x": "a</parameter>b"}),
        ),
        (
            [
                r#"<tool_call><function=ping><parameter=x>"a</parameter>"#,
                r#"b"</parameter></function> All done."#,
            ],
            json!({"x": "a</parameter>b"}),
        ),
        (
            [
                r#"<tool_call><function=ping><parameter=x>"abc"</parameter></function>"#,
                " All done.",
            ],
            json!({"x": "abc"}),
        ),
    ];
    assert_tag_parity(quoted_pat, &quoted_cases);
    let qwen_pat = r"(?s)<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>";
    let key_edge_cases = [
        (
            [
                "<tool_call><function=ping><parameter=>1</parameter><parameter=y>2</parameter></function>",
                " All done.",
            ],
            json!({"y": 2}),
        ),
        (
            [
                "<tool_call><function=ping><parameter=bad\nkey>1</parameter><parameter=y>2</parameter></function>",
                " All done.",
            ],
            json!({"y": 2}),
        ),
        (
            [
                "<tool_call><function=ping><parameter=bad\n",
                "<parameter=y>2</parameter></function> All done.",
            ],
            json!({"y": 2}),
        ),
    ];
    assert_tag_parity(qwen_pat, &key_edge_cases);
    let wild_pat = r"(?s)<parameter=(?P<key>.+?)>(?P<value>.*?)</parameter>";
    let wild_cases = [
        (
            [
                "<tool_call><function=ping><parameter=>>1</parameter></function>",
                " All done.",
            ],
            json!({">": 1}),
        ),
        (
            [
                "<tool_call><function=ping><parameter=x>1</parameter></function>",
                " All done.",
            ],
            json!({"x": 1}),
        ),
    ];
    assert_tag_parity(wild_pat, &wild_cases);
    check_glm_key_parity();
}

fn check_glm_key_parity() {
    let glm_cases = [
        (
            [
                "<tool_call>ping\n<arg_key>x</arg_key>junk<arg_value>1</arg_value></tool_call>",
                " All done.",
            ],
            json!({}),
        ),
        (
            [
                "<tool_call>ping\n<arg_key>a</arg_key>b</arg_key> <arg_value>1</arg_value></tool_call>",
                " All done.",
            ],
            json!({"a</arg_key>b": 1}),
        ),
        (
            [
                "<tool_call>ping\n<arg_key>x</arg_key> \n\t<arg_value>1</arg_value></tool_call>",
                " All done.",
            ],
            json!({"x": 1}),
        ),
    ];
    for buffered in [false, true] {
        let mut tmpl: Value = serde_json::from_str(include_str!("../glm53.json")).unwrap();
        if buffered {
            let t = tmpl["fields"]["tool_calls"].as_object_mut().unwrap();
            t.remove("close");
            t.insert("close_pattern".into(), json!("[<]/tool_call>"));
        }
        let cfg = json!({"response_template": tmpl}).to_string();
        let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
        assert_parser_cases(&parser, &glm_cases);
    }
}

fn assert_tag_parity(tag_pat: &str, cases: &[([&str; 2], Value)]) {
    for close_pat in ["</function>", "[<]/function>"] {
        let cfg = json!({"response_template": {"fields": {"tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\n]+)>",
            "close_pattern": close_pat,
            "content": "xml-inline",
            "content_args": {"tag_pattern": tag_pat},
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        }}}})
        .to_string();
        let parser = ResponseParser::from_config(&cfg).unwrap().unwrap();
        assert_parser_cases(&parser, cases);
    }
}

fn assert_parser_cases(parser: &ResponseParser, cases: &[([&str; 2], Value)]) {
    for (pieces, want) in cases {
        let full = pieces.concat();
        let whole = parser.parse(&full, true, false);
        assert_eq!(whole.content, " All done.");
        assert_eq!(whole.tool_calls.len(), 1);
        assert_eq!(whole.tool_calls[0].function.name, "ping");
        assert_eq!(&whole.tool_calls[0].function.arguments, want);
        let mut stream = parser.stream(true, false);
        let mut events = Vec::new();
        for p in pieces {
            stream.push(p, &mut events);
        }
        stream.finish(&mut events);
        assert_stream_result(&events, want);
        let mut char_stream = parser.stream(true, false);
        let mut char_events = Vec::new();
        for ch in full.chars() {
            let mut buf = [0u8; 4];
            char_stream.push(ch.encode_utf8(&mut buf), &mut char_events);
        }
        char_stream.finish(&mut char_events);
        assert_stream_result(&char_events, want);
    }
}

fn assert_stream_result(events: &[ResponseEvent], want: &Value) {
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::Content(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, " All done.");
    assert_eq!(collect_calls(events), [("ping".into(), want.clone())]);
}
