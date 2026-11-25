use crate::{FunctionCall, Message, RenderOptions, Template, ToolCall};
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
