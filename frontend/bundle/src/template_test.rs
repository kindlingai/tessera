use crate::{FunctionCall, Message, RenderOptions, ResponseParser, Template, ToolCall};
use serde_json::json;

const CHATML: &str = "{% for m in messages %}\
{{ '<|im_start|>' + m.role + '\n' + m.content.strip() + '<|im_end|>\n' }}{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}\
{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}";

fn say(role: &str, content: &str) -> Message {
    Message::text(role, content)
}

#[test]
fn renders_turns() {
    let t = Template::new(CHATML.into()).unwrap();
    let turns = [say("system", "Be brief."), say("user", " Hi ")];
    let want = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(t.render(&turns, true).unwrap(), want);
    assert_eq!(
        t.render(&turns, false).unwrap(),
        format!("{want}<think>\n\n</think>\n\n")
    );
}

#[test]
fn strftime_reads_clock() {
    let fixed: jiff::Zoned = "2026-03-04T10:00:00+00:00[UTC]".parse().unwrap();
    let clock = std::sync::Arc::new(move || fixed.clone());
    let t = Template::with_clock("{{ strftime_now('%d %B %Y') }}".into(), clock).unwrap();
    assert_eq!(t.render(&[], true).unwrap(), "04 March 2026");
}

#[test]
fn stops_at_break() {
    let source = "{% for m in messages %}{% if m.role == 'user' %}{% break %}{% endif %}{{ m.role }}{% endfor %}";
    let t = Template::new(source.into()).unwrap();
    let turns = [say("system", ""), say("user", ""), say("tool", "")];
    assert_eq!(t.render(&turns, true).unwrap(), "system");
}

#[test]
fn renders_reasoning_effort() {
    let source = "{% set e = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' %}{{ e }}";
    let t = Template::new(source.into()).unwrap();
    assert_eq!(t.render_with(&[], true, None).unwrap(), "max");
    assert_eq!(t.render_with(&[], true, Some("low")).unwrap(), "low");
    assert_eq!(t.render_with(&[], true, Some("high")).unwrap(), "high");
}

#[test]
fn strips_generation_tags() {
    let source =
        "{% for m in messages %}{%- generation %}{{ m.content }}{%- endgeneration %}{% endfor %}";
    let t = Template::new(source.into()).unwrap();
    assert_eq!(t.render(&[say("assistant", "ok")], true).unwrap(), "ok");
}

#[test]
fn renders_tools_and_dictsort() {
    let source = "{{ bos_token }}{% for t in tools %}{{ t.function.name }}:\
{% for k, v in t.function.parameters.properties | dictsort %}{{ k }}={{ v.type }},{% endfor %}{% endfor %}\
{% for m in messages %}{% if m.get('tool_calls') %}{% for tc in m.tool_calls %}\
call:{{ tc.function.name }}({% for k, v in tc.function.arguments | dictsort %}{{ k }}={{ v }}{% endfor %})\
{% endfor %}{% endif %}{% endfor %}{{ tools[0].function.parameters | tojson(ensure_ascii=False) }}";
    let t = Template::new(source.into())
        .unwrap()
        .with_tokens(Some("<bos>".into()), None);
    let tools = [json!({"type": "function", "function": {
        "name": "weather",
        "parameters": {"properties": {"z": {"type": "int"}, "a": {"type": "string"}}}
    }})];
    let call = ToolCall {
        id: Some("c1".into()),
        kind: "function".into(),
        function: FunctionCall {
            name: "weather".into(),
            arguments: json!({"city": "Tokyo"}),
        },
    };
    let msg = Message {
        role: "assistant".into(),
        tool_calls: Some(vec![call]),
        ..Default::default()
    };
    let opts = RenderOptions {
        tools: &tools,
        ..Default::default()
    };
    let got = t.render_options(&[msg], &opts).unwrap();
    let json = r#"{"properties": {"z": {"type": "int"}, "a": {"type": "string"}}}"#;
    assert_eq!(
        got,
        format!("<bos>weather:a=string,z=int,call:weather(city=Tokyo){json}")
    );
}

#[test]
fn renders_like_hugging_face() {
    let source = "{{ tools[0] | tojson }}|{{ tools[0] | tojson(indent=1) }}|\
{{ messages[0].reasoning_content }}|<tool_call>";
    let t = Template::new(source.into()).unwrap();
    let tools = [json!({"z": "it's <a> & b", "a": [1, 2]})];
    let msg = Message {
        role: "assistant".into(),
        reasoning: Some("why".into()),
        ..Default::default()
    };
    let opts = RenderOptions {
        tools: &tools,
        ..Default::default()
    };
    let want = "{\"z\": \"it's <a> & b\", \"a\": [1, 2]}|\
{\n \"z\": \"it's <a> & b\",\n \"a\": [\n  1,\n  2\n ]\n}|why|<tool_call>";
    assert_eq!(t.render_options(&[msg], &opts).unwrap(), want);
    let parser = ResponseParser::builtin(&t).unwrap().unwrap();
    let raw =
        "<think>\nhm\n</think>\n\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>";
    let parsed = parser.parse(raw, true, false);
    assert_eq!(
        (parsed.reasoning.as_str(), parsed.content.as_str()),
        ("hm", "")
    );
    assert_eq!(parsed.tool_calls[0].function.name, "f");
    let plain = Template::new("no tools".into()).unwrap();
    assert!(ResponseParser::builtin(&plain).unwrap().is_none());
}

#[test]
fn forwards_clear_thinking() {
    let source = "{% if clear_thinking is defined %}{{ clear_thinking }}{% else %}unset{% endif %}";
    let t = Template::new(source.into()).unwrap();
    let msg = [Message::text("user", "hi")];
    let at = |clear_thinking| {
        let opts = RenderOptions {
            clear_thinking,
            ..Default::default()
        };
        t.render_options(&msg, &opts).unwrap()
    };
    assert_eq!(at(None), "unset");
    assert_eq!(at(Some(false)), "False");
    assert_eq!(at(Some(true)), "True");
}
