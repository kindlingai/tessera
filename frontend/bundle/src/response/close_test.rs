use crate::{ResponseEvent, ResponseParser};
use serde_json::{Value, json};

#[test]
fn xml_inline_closers() {
    let cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think>|(?=<tool_call>)","content":"text"},"tool_calls":{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close_pattern":"</tool_call>|(?=<tool_call>)","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?(?:</parameter>|(?=<parameter=)|(?=</function>))"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}},"content":{"close":["<|im_end|>"],"content":"text"}}}}"#;
    let p = ResponseParser::from_config(cfg).unwrap().unwrap();
    assert!(p.supports_tools() && p.starts_thinking("<|im_start|>assistant\n<think>\n"));
    let prefix = p.call_prefix(Some("echo")).unwrap();
    assert_eq!(prefix, "<tool_call>\n<function=echo>\n");
    let raw = format!("{prefix}<parameter=word>\namber\n</parameter>\n</function>\n</tool_call>");
    let one = p.parse(&raw, true, false);
    assert_eq!(
        one.tool_calls[0].function.arguments,
        json!({"word": "amber"})
    );
    let think = p.parse(&format!("plan{raw}"), true, true);
    assert_eq!(
        (think.reasoning.as_str(), think.tool_calls.len()),
        ("plan", 1)
    );
    let adj = format!("{}{raw}", raw.trim_end_matches("</tool_call>"));
    assert_eq!(p.parse(&adj, true, false).tool_calls.len(), 2);
    let imp = "<tool_call>\n<function=echo>\n<parameter=a>one\n<parameter=b>two\n</function>\n</tool_call>";
    assert_eq!(
        p.parse(imp, true, false).tool_calls[0].function.arguments,
        json!({"a": "one", "b": "two"})
    );
}

#[test]
fn escaped_close_prefix() {
    let pipe_cfg = r#"{"response_template":{"fields":{"content":{"close_pattern":"<\\|im_end\\|>","content":"text"}}}}"#;
    let pipe = ResponseParser::from_config(pipe_cfg).unwrap().unwrap();
    assert_eq!(pipe.parse("1 < 2", true, false).content, "1 < 2");
    assert_eq!(pipe.parse("hi<|im_end|>x", true, false).content, "hi");
    let compact_cfg = r#"{"response_template":{"fields":{"tool_calls":{"open_pattern":"<tool_call><function=(?P<name>[^>\\n]+)>","close":"</tool_call>","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?</parameter>"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}}}}"#;
    let compact = ResponseParser::from_config(compact_cfg).unwrap().unwrap();
    let cp = compact.call_prefix(Some("echo")).unwrap();
    assert_eq!(cp, "<tool_call><function=echo>");
    let raw = format!("{cp}<parameter=x>1</parameter></function></tool_call>");
    assert_eq!(
        compact.parse(&raw, true, false).tool_calls[0]
            .function
            .arguments,
        json!({"x": 1})
    );
    let odd_cfg = r#"{"response_template":{"fields":{"tool_calls":{"open_pattern":"<call\\s+\\[(?P<name>\\w+)\\]>","close":"</call>","content":"json"}}}}"#;
    let odd = ResponseParser::from_config(odd_cfg).unwrap().unwrap();
    assert!(odd.call_prefix(Some("echo")).is_none());
}

fn parse_stream(
    parser: &ResponseParser,
    chunks: &[&str],
) -> (String, String, Vec<(String, Value)>) {
    let mut s = parser.stream(true, false);
    let mut ev = Vec::new();
    for part in chunks {
        s.push(part, &mut ev);
    }
    s.finish(&mut ev);
    let (mut reasoning, mut content, mut calls, mut cur) =
        (String::new(), String::new(), Vec::new(), None);
    for e in ev {
        match e {
            ResponseEvent::Reasoning(t) => reasoning.push_str(&t),
            ResponseEvent::Content(t) => content.push_str(&t),
            ResponseEvent::ToolCall(tc) => calls.push((tc.function.name, tc.function.arguments)),
            ResponseEvent::ToolCallStart { name, .. } => cur = Some((name, String::new())),
            ResponseEvent::ToolCallArgs(a) => cur.as_mut().unwrap().1.push_str(&a),
            ResponseEvent::ToolCallEnd => {
                let (n, a) = cur.take().unwrap();
                calls.push((n, serde_json::from_str(&a).unwrap()));
            }
        }
    }
    (reasoning, content, calls)
}

fn check_parity(parser: &ResponseParser, chunks: &[&str]) {
    let whole = parser.parse(&chunks.concat(), true, false);
    let want_calls: Vec<(String, Value)> = whole
        .tool_calls
        .into_iter()
        .map(|tc| (tc.function.name, tc.function.arguments))
        .collect();
    assert_eq!(
        parse_stream(parser, chunks),
        (whole.reasoning, whole.content, want_calls)
    );
}

#[test]
fn lookahead_closer_parity() {
    let think_cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think>(?!-literal)","content":"text"},"tool_calls":{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close":"</tool_call>","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?</parameter>"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}}}}"#;
    let p = ResponseParser::from_config(think_cfg).unwrap().unwrap();
    let sample = "<think>draft </think>-literal <tool_call><function=ping><parameter=x>1</parameter></function></tool_call> still thinking</think>answer";
    for i in 0..=sample.len() {
        check_parity(&p, &[&sample[..i], &sample[i..]]);
    }
    check_parity(
        &p,
        &[
            "<think>draft </think>",
            " <tool_call><function=ping><parameter=x>1</parameter></function></tool_call>",
        ],
    );
    let end_cfg = r#"{"response_template":{"fields":{"content":{"close_pattern":"<end>(?!-more)","content":"text"},"tool_calls":{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close":"</tool_call>","content":"xml-inline","content_args":{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?</parameter>"},"transform":{"type":"function","function":{"name":"{name}","arguments":"{content}"}}}}}}"#;
    let ep = ResponseParser::from_config(end_cfg).unwrap().unwrap();
    for tail in [
        " <tool_call><function=ping><parameter=x>1</parameter></function></tool_call>",
        "-more <tool_call><function=ping><parameter=x>1</parameter></function></tool_call>",
    ] {
        check_parity(&ep, &["answer<end>", tail]);
    }
    // Deferred tool-call closers must keep consecutive XML calls separate.
    let two = "<tool_call>\n<function=ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=ping>\n<parameter=x>2</parameter>\n</function>\n</tool_call> All done.";
    let want = vec![
        ("ping".to_string(), json!({"x": 1})),
        ("ping".to_string(), json!({"x": 2})),
    ];
    for pat in ["</tool_call>(?!-literal)", "</tool_call>(?!.*-literal)"] {
        let cfg = format!(
            r#"{{"response_template":{{"fields":{{"tool_calls":{{"open_pattern":"<tool_call>\\s*<function=(?P<name>[^>\\n]+)>","close_pattern":"{pat}","content":"xml-inline","content_args":{{"tag_pattern":"<parameter=(?P<key>[^>\\n]+)>\\n?(?P<value>.*?)\\n?</parameter>"}},"transform":{{"type":"function","function":{{"name":"{{name}}","arguments":"{{content}}"}}}}}}}}}}}}"#
        );
        let tp = ResponseParser::from_config(&cfg).unwrap().unwrap();
        let whole = tp.parse(two, true, false);
        let got_whole: Vec<(String, Value)> = whole
            .tool_calls
            .into_iter()
            .map(|tc| (tc.function.name, tc.function.arguments))
            .collect();
        assert_eq!(
            (whole.content.as_str(), got_whole),
            ("\n All done.", want.clone())
        );
        for i in 0..=two.len() {
            let (_, content, calls) = parse_stream(&tp, &[&two[..i], &two[i..]]);
            assert_eq!((content.as_str(), calls), ("\n All done.", want.clone()));
        }
    }
    // Competing longer alternation branch must not be bypassed mid-stream.
    let alt_cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think>\\s*</analysis>|</think>(?!-literal)","content":"text"}}}}"#;
    let ap = ResponseParser::from_config(alt_cfg).unwrap().unwrap();
    let alt_whole = ap.parse("<think>plan</think> </analysis>answer", true, false);
    let alt_stream = parse_stream(&ap, &["<think>plan</think> ", "</analysis>answer"]);
    assert_eq!(
        (alt_whole.reasoning.as_str(), alt_whole.content.as_str()),
        ("plan", "answer")
    );
    assert_eq!(
        (alt_stream.0.as_str(), alt_stream.1.as_str()),
        ("plan", "answer")
    );
    check_optional_delayed_spans();
}

fn check_optional_delayed_spans() {
    for pat in [
        r"</think>(?:\s*</analysis>)?(?!-literal)",
        r"</think>(?:\s*</analysis>)?",
    ] {
        let cfg = json!({"response_template": {"fields": {
            "reasoning_content": {"open": "<think>", "close_pattern": pat, "content": "text"},
        }}})
        .to_string();
        let p = ResponseParser::from_config(&cfg).unwrap().unwrap();
        let whole = p.parse("<think>plan</think> </analysis>answer", true, false);
        let split = parse_stream(&p, &["<think>plan</think> ", "</analysis>answer"]);
        assert_eq!(
            (whole.reasoning.as_str(), whole.content.as_str()),
            ("plan", "answer")
        );
        assert_eq!((split.0.as_str(), split.1.as_str()), ("plan", "answer"));
    }
    // Deferred content closer preserves empty UTF-8 lead tokens inside thinking.
    let delay_cfg = r#"{"response_template":{"fields":{"reasoning_content":{"open":"<think>","close_pattern":"</think>","content":"text"},"content":{"close_pattern":"<end>(?!.*literal)","content":"text"}}}}"#;
    let dp = ResponseParser::from_config(delay_cfg).unwrap().unwrap();
    let mut s = dp.stream(true, false);
    let mut ev = Vec::new();
    for piece in ["<end>-literal <think>plan", "", "é", " more</think>answer"] {
        s.push(piece, &mut ev);
    }
    s.finish(&mut ev);
    assert_eq!(s.reasoning_tokens(), 4);
    // Empty token counts after opener, not after closer.
    let mut after_open = dp.stream(true, false);
    ev.clear();
    for piece in ["<end>-literal <think>", "", "é</think>answer"] {
        after_open.push(piece, &mut ev);
    }
    after_open.finish(&mut ev);
    assert_eq!(after_open.reasoning_tokens(), 3);
    let mut after_close = dp.stream(true, false);
    ev.clear();
    for piece in ["<end>-literal <think>plan</think>", "", "é"] {
        after_close.push(piece, &mut ev);
    }
    after_close.finish(&mut ev);
    assert_eq!(after_close.reasoning_tokens(), 1);
    check_r7_closers();
}

fn check_r7_closers() {
    // Lookbehind after quantifier still retains left context until EOF.
    for pat in [r"</think\s*>(?<!raw</think>)", r"</think>(?<!raw</think>)"] {
        let cfg = json!({"response_template": {"fields": {
            "reasoning_content": {"open": "<think>", "close_pattern": pat, "content": "text"},
        }}})
        .to_string();
        let p = ResponseParser::from_config(&cfg).unwrap().unwrap();
        let whole = p.parse("<think>raw</think> first end</think>answer", true, false);
        let split = parse_stream(&p, &["<think>raw</think> first", " end</think>answer"]);
        assert_eq!(
            (whole.reasoning.as_str(), whole.content.as_str()),
            ("raw</think> first end", "answer")
        );
        assert_eq!(
            (split.0.as_str(), split.1.as_str()),
            ("raw</think> first end", "answer")
        );
    }
    // Competing longer literal holds short match until disambiguated.
    let lit_cfg = json!({"response_template": {"fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": "</think></analysis>|</think>",
            "content": "text",
        },
    }}})
    .to_string();
    let lp = ResponseParser::from_config(&lit_cfg).unwrap().unwrap();
    let sample = "<think>plan</think></analysis>answer";
    for i in 0..=sample.len() {
        check_parity(&lp, &[&sample[..i], &sample[i..]]);
    }
    // Earlier partial literal beats later short match across chunks.
    let early_cfg = json!({"response_template": {"fields": {
        "reasoning_content": {"open": "<think>", "close_pattern": "cd|abcdef", "content": "text"},
    }}})
    .to_string();
    let ep = ResponseParser::from_config(&early_cfg).unwrap().unwrap();
    check_parity(&ep, &["<think>planabcd", "efanswer"]);
}
