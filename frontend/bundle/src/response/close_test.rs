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
    // Lookbehind and class closers preserve XML context.
    check_xml_tool_closers(
        r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>",
        &[
            "<tool_call><function=ping><parameter=x>1</parameter></function></tool_call>",
            "\n<tool_call><function=ping><parameter=x>2</parameter></function></tool_call>",
            " All done.",
        ],
        &[
            r"(?<=</parameter></function>)</tool_call>",
            r"(?<=</function>)</tool_call>",
        ],
    );
    let implicit_tag = r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>))";
    check_xml_tool_closers(
        implicit_tag,
        &[
            "<tool_call><function=ping><parameter=x>1</function></tool_call>",
            "\n<tool_call><function=ping><parameter=x>2</function></tool_call>",
            " All done.",
        ],
        &[
            r"</function></tool_call>",
            r"[<]/function></tool_call>",
            r"(?<=[0-9])</function></tool_call>",
        ],
    );
    // Closer prefix bytes must not leak into parameter values.
    let tail_cfg = xml_tool_cfg(r"[T]AIL</function></tool_call>", implicit_tag);
    let tail_p = ResponseParser::from_config(&tail_cfg).unwrap().unwrap();
    let parsed = tail_p.parse(
        "<tool_call><function=ping><parameter=x>1TAIL</function></tool_call>",
        true,
        false,
    );
    assert_eq!(parsed.tool_calls[0].function.arguments, json!({}));
    check_r10_buffered_tags();
}

fn check_r10_buffered_tags() {
    // End-of-body `$` tag terminator keeps the final parameter before closer.
    let end_cfg = xml_tool_cfg(
        r"</function></tool_call>",
        r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)(?:</parameter>|$)",
    );
    let ep = ResponseParser::from_config(&end_cfg).unwrap().unwrap();
    let sample = "<tool_call><function=ping><parameter=y>2</parameter><parameter=x>1</function></tool_call> All done.";
    let want = vec![("ping".to_string(), json!({"y": 2, "x": 1}))];
    let whole = ep.parse(sample, true, false);
    assert_eq!(
        (
            whole.content.as_str(),
            whole.tool_calls[0].function.arguments.clone()
        ),
        (" All done.", json!({"y": 2, "x": 1}))
    );
    for i in 0..=sample.len() {
        let (_, content, calls) = parse_stream(&ep, &[&sample[..i], &sample[i..]]);
        assert_eq!((content.as_str(), calls), (" All done.", want.clone()));
    }
    // Zero-width closer defers buffered tag lookahead for split parity.
    let zero_cfg = xml_tool_cfg(
        r"(?=</function>)",
        r"[<]parameter=(?P<key>[^>]+)>(?P<value>.*?)(?:</parameter>|(?=</function></tool_call>))",
    );
    let zp = ResponseParser::from_config(&zero_cfg).unwrap().unwrap();
    let split = [
        "<tool_call><function=ping><parameter=y>2</parameter><parameter=x>1</function>",
        "</tool_call> All done.",
    ];
    let (_, content, calls) = parse_stream(&zp, &split);
    assert_eq!(
        (content.as_str(), calls),
        ("</function></tool_call> All done.", want.clone())
    );
    for i in 0..=sample.len() {
        check_parity(&zp, &[&sample[..i], &sample[i..]]);
    }
    check_r11_overlap_tags();
}

fn check_r11_overlap_tags() {
    // Overlapping value and block closers keep calls and text intact.
    let overlap_cases: &[(&str, &str, &[&str])] = &[
        (
            r"</parameter></function></tool_call>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)(?:</parameter>|(?=<parameter=)|(?=</function>))",
            &[
                "<tool_call><function=ping><parameter=x>1</parameter>",
                "</function></tool_call> next <tool_call><function=ping><parameter=x>2</parameter>",
                "</function></tool_call> All done.",
            ],
        ),
        (
            r"</function></tool_call>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)(?:</parameter>|</function>)",
            &[
                "<tool_call><function=ping><parameter=x>1</function>",
                "</tool_call> next <tool_call><function=ping><parameter=x>2</function>",
                "</tool_call> All done.",
            ],
        ),
        (
            r"</function></tool_call>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter></function>",
            &[
                "<tool_call><function=ping><parameter=x>1</parameter></function>",
                "</tool_call> next <tool_call><function=ping><parameter=x>2</parameter></function>",
                "</tool_call> All done.",
            ],
        ),
    ];
    for (close_pat, tag_pat, pieces) in overlap_cases {
        check_xml_tool_calls(tag_pat, pieces, &[close_pat], " next  All done.");
    }
    // Streamed string value with trailing newline handles overlapping block closer.
    let nl_cfg = xml_tool_cfg(
        r"</parameter></function></tool_call>",
        r"<parameter=(?P<key>[^>]+)>\n?(?P<value>.*?)\n?</parameter>",
    );
    let np = ResponseParser::from_config(&nl_cfg).unwrap().unwrap();
    let nl_pieces = [
        "<tool_call><function=ping><parameter=msg>\nhello\n</parameter>",
        "</function></tool_call> next <tool_call><function=ping><parameter=msg>\nbye\n</parameter>",
        "</function></tool_call> All done.",
    ];
    let want_nl = vec![
        ("ping".to_string(), json!({"msg": "hello"})),
        ("ping".to_string(), json!({"msg": "bye"})),
    ];
    let (_, content, calls) = parse_stream(&np, &nl_pieces);
    assert_eq!((content.as_str(), calls), (" next  All done.", want_nl));
    let nl_joined = nl_pieces.concat();
    for i in 0..=nl_joined.len() {
        check_parity(&np, &[&nl_joined[..i], &nl_joined[i..]]);
    }
    check_r12_tail_overlap();
}

fn check_r12_tail_overlap() {
    // Longer value closer extending past shorter block closer defers split.
    let cases: &[(&str, &str, &[&str], &str, Value)] = &[
        (
            r"</function></tool_call>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter></function></tool_call>\n",
            &[
                "<tool_call><function=ping><parameter=x>1</parameter></function></tool_call>",
                "\n All done.",
            ],
            "\n All done.",
            json!({"x": 1}),
        ),
        (
            r"</function></tool_call>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter></function></tool_call>\n",
            &[
                "<tool_call><function=ping><parameter=x>hello</parameter></function></tool_call>",
                "\n All done.",
            ],
            "\n All done.",
            json!({"x": "hello"}),
        ),
        (
            r"</function>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter></function></tool_call>",
            &[
                "<tool_call><function=ping><parameter=x>1</parameter></function>",
                "</tool_call> All done.",
            ],
            "</tool_call> All done.",
            json!({"x": 1}),
        ),
        (
            r"</function>",
            r"<parameter=(?P<key>[^>]+)>(?P<value>.*?)</parameter></function></tool_call>",
            &[
                "<tool_call><function=ping><parameter=x>1</parameter></function>",
                " All done.",
            ],
            " All done.",
            json!({"x": "1</parameter>"}),
        ),
    ];
    for (close_pat, tag_pat, pieces, want_content, want_args) in cases {
        let cfg = xml_tool_cfg(close_pat, tag_pat);
        let p = ResponseParser::from_config(&cfg).unwrap().unwrap();
        let joined = pieces.concat();
        let want = vec![("ping".to_string(), want_args.clone())];
        let whole = p.parse(&joined, true, false);
        assert_eq!(
            (
                whole.content.as_str(),
                whole.tool_calls[0].function.arguments.clone()
            ),
            (*want_content, want_args.clone())
        );
        let (_, content, calls) = parse_stream(&p, pieces);
        assert_eq!((content.as_str(), calls), (*want_content, want));
        for i in 0..=joined.len() {
            check_parity(&p, &[&joined[..i], &joined[i..]]);
        }
    }
}

fn xml_tool_cfg(close_pat: &str, tag_pat: &str) -> String {
    json!({"response_template": {"fields": {"tool_calls": {
        "open_pattern": r"<tool_call>\s*<function=(?P<name>[^>\n]+)>",
        "close_pattern": close_pat,
        "content": "xml-inline",
        "content_args": {"tag_pattern": tag_pat},
        "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
    }}}})
    .to_string()
}

fn check_xml_tool_closers(tag_pat: &str, pieces: &[&str], pats: &[&str]) {
    check_xml_tool_calls(tag_pat, pieces, pats, "\n All done.");
}

fn check_xml_tool_calls(tag_pat: &str, pieces: &[&str], pats: &[&str], want_content: &str) {
    let joined = pieces.concat();
    let want = vec![
        ("ping".to_string(), json!({"x": 1})),
        ("ping".to_string(), json!({"x": 2})),
    ];
    for pat in pats {
        let cfg = xml_tool_cfg(pat, tag_pat);
        let tp = ResponseParser::from_config(&cfg).unwrap().unwrap();
        let whole = tp.parse(&joined, true, false);
        let got: Vec<(String, Value)> = whole
            .tool_calls
            .into_iter()
            .map(|tc| (tc.function.name, tc.function.arguments))
            .collect();
        assert_eq!((whole.content.as_str(), got), (want_content, want.clone()));
        let (_, content, calls) = parse_stream(&tp, pieces);
        assert_eq!((content.as_str(), calls), (want_content, want.clone()));
        for i in 0..=joined.len() {
            check_parity(&tp, &[&joined[..i], &joined[i..]]);
        }
    }
}
