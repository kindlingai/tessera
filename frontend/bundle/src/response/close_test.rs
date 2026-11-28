use crate::ResponseParser;
use serde_json::json;

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
