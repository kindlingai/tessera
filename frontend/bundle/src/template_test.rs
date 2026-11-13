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
