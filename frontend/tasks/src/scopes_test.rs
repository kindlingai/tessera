use crate::api::Params;
use crate::scopes::{Boost, Thinking, guide};
use tessera_bundle::ResponseParser;
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Edit, Penalties};

#[test]
fn boost_specs() {
    let start = |spec| Boost::parse(spec).unwrap().unwrap().start.tokens(1000);
    assert_eq!(start("start=16384,per1k=2,max=10"), 16384);
    assert_eq!(start("start=0.5x,per1k=2,max=10"), 500);
    assert_eq!(start(" start = 16384|0.5x , per1k=2,max=10"), 500);
    assert_eq!(Boost::parse("start=1,per1k=0,max=10"), Ok(None));
    for spec in [
        "start",
        "start=0|0.5x",
        "start=1.5x",
        "max=101",
        "max=1,max=2",
        "x=1",
    ] {
        assert!(Boost::parse(spec).is_err(), "{spec}");
    }
}

/// Edits change as block markers move the scope.
#[test]
fn scoped_edits() {
    let vocab = r#"{"model": {"type": "WordLevel", "unk_token": "a",
        "vocab": {"a": 0, "<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}}}"#;
    let config = r#"{"response_template": {"fields": {
        "thinking": {"open": "<think>\n", "close": "</think>"},
        "tool_calls": {"open": "<tool_call>", "close": "</tool_call>", "content": "json"}}}}"#;
    let tokenizer = Hf::from_json(vocab.as_bytes()).unwrap();
    let params = Params {
        max_tokens: 8,
        penalties: Penalties {
            presence: 0.5,
            ..Penalties::default()
        },
        thinking: Thinking {
            presence: 1.0,
            boost: Boost::parse("start=1,per1k=1000,max=10").unwrap(),
            ..Thinking::default()
        },
        starts_thinking: true,
        ..Params::default()
    };
    let refused = guide(None, &tokenizer, &params, &[]).err().unwrap();
    assert_eq!(refused.1, "thinking_presence_penalty");
    let parser = ResponseParser::from_config(config).unwrap();
    let mut guide = guide(parser.as_ref(), &tokenizer, &params, &[]).unwrap();
    let edit = |id, add| Edit { id, add, rep: 1.0 };
    let steps = [
        (0, vec![edit(0, -1.5), edit(2, 1.0)]),
        (0, vec![edit(0, -1.5), edit(2, 2.0)]),
        (2, vec![edit(0, -0.5), edit(2, -0.5)]),
        (3, vec![]),
    ];
    for (id, edits) in steps {
        guide.accept(id).unwrap();
        assert_eq!(guide.edits(), edits, "after {id}");
    }
}
