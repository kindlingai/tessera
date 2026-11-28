use serde_json::json;
use tessera_bundle::{FunctionCall, Message, RenderOptions, Template, ToolCall};

#[test]
fn tojson_ascii_escape() {
    let src = "{{ tools[0] | tojson(ensure_ascii=True) }}|\
               {{ tools[1] | tojson(ensure_ascii=True, indent=2) }}|\
               {{ tools[0] | tojson(ensure_ascii=True, indent=36) }}";
    let tmpl = Template::new(src.into()).unwrap();
    let tools = [json!({"ok": "plain"}), json!({"jp": "日本", "em": "🚀"})];
    let opts = RenderOptions {
        tools: &tools,
        ..Default::default()
    };
    let got = tmpl.render_options(&[], &opts).unwrap();
    let parts: Vec<&str> = got.split('|').collect();
    assert_eq!(parts[0], r#"{"ok": "plain"}"#);
    assert_eq!(
        parts[1],
        "{\n  \"jp\": \"\\u65e5\\u672c\",\n  \"em\": \"\\ud83d\\ude80\"\n}"
    );
    let pad = " ".repeat(36);
    assert_eq!(parts[2], format!("{{\n{pad}\"ok\": \"plain\"\n}}"));
}

#[test]
fn generation_and_args() {
    // Plain word "generation" without Jinja tag stays intact.
    let kept = Template::new("next generation: {{ messages[0].content }}".into()).unwrap();
    let msg = [Message::text("user", "ok")];
    assert_eq!(kept.render(&msg, true).unwrap(), "next generation: ok");

    // Template joining arguments as text triggers stringification retry.
    let src = "{% for m in messages %}{% for tc in m.tool_calls %}\
               {{ tc.function.name }}:{{ tc.function.arguments + '!' }}{% endfor %}{% endfor %}";
    let tmpl = Template::new(src.into()).unwrap();
    let call = ToolCall {
        id: None,
        kind: "function".into(),
        function: FunctionCall {
            name: "f".into(),
            arguments: json!({"a": 1}),
        },
    };
    let m = Message {
        role: "assistant".into(),
        tool_calls: Some(vec![call]),
        ..Default::default()
    };
    assert_eq!(tmpl.render(&[m], true).unwrap(), "f:{\"a\":1}!");
}
