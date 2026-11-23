use crate::{Message, Template};

const CHATML: &str = "{% for m in messages %}\
{{ '<|im_start|>' + m.role + '\n' + m.content.strip() + '<|im_end|>\n' }}{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}\
{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}";

fn say(role: &str, content: &str) -> Message {
    Message {
        role: role.into(),
        content: content.into(),
    }
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
