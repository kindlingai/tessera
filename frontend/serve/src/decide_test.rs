use crate::decide::{Kind, answer, answers, questions, system, template};
use crate::mocks::{MockGenerator, MockTokens, json, loaded};
use serde_json::{Value, json};
use std::sync::Arc;

fn asked(value: Value) -> Result<Vec<crate::decide::Question>, String> {
    questions(Some(&value)).map_err(|r| r.message)
}

#[test]
fn questions_check() {
    let got = asked(json!({
        "b": {"type": "noul", "criteria": {"true": "now", "false": null}},
        "a": {"type": "choice", "criteria": {"x": "first", "y": null}},
        "s": {"type": "score", "instructions": {"k": 1}, "criteria": ["lo", "mid", "hi"]},
    }))
    .unwrap();
    let ids: Vec<&str> = got.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(ids, ["b", "a", "s"]);
    assert_eq!(got[0].choices[0], ("yes".into(), Some("now".into())));
    assert_eq!(
        (got[1].kind, got[1].labels.clone()),
        (Kind::Choice, vec!["A".into(), "B".into()])
    );
    assert_eq!(
        (got[2].labels[2].as_str(), got[2].instructions.as_str()),
        ("3", "{\"k\":1}")
    );
    let refused = [
        (json!({}), "questions: needs"),
        (json!({"a:b": {"type": "noul"}}), "must be non-empty"),
        (json!({"q": {"type": "maybe"}}), "unknown type \"maybe\""),
        (
            json!({"q": {"type": "choice", "criteria": ["x"]}}),
            "choice criteria",
        ),
        (
            json!({"q": {"type": "score", "criteria": ["one"]}}),
            "at least two",
        ),
    ];
    for (value, want) in refused {
        let message = asked(value).unwrap_err();
        assert!(message.contains(want), "{message}");
    }
}

#[test]
fn layout_and_slots() {
    let qs = asked(json!({
        "go": {"type": "noul", "instructions": "Go?"},
        "pick": {"type": "choice", "criteria": {"red": "warm", "blue": null}},
    }))
    .unwrap();
    let text = system(&qs, Some("Be brief."));
    assert!(text.contains("\nQuestion go: Go?\n  yes\n  no\n"));
    assert!(text.contains("  A: red (warm)\n  B: blue\n"));
    assert!(text.ends_with("\"id: label\".\n\nBe brief."));
    assert_eq!(answers(&qs, &[1, 1]), "go: no\npick: B");
    // One id per character: `no` is not one token.
    let message = template(&MockTokens, &qs).unwrap_err().message;
    assert_eq!(
        message,
        "question \"go\": label \"no\" is not a single token"
    );
    let (ids, places) = template(&MockTokens, &qs[1..]).unwrap();
    assert_eq!(places[0].at, ids.len() - 1);
    assert_eq!(places[0].ids, [u32::from('A'), u32::from('B')]);
}

#[test]
fn shapes_answers() {
    let qs = asked(json!({
        "n": {"type": "noul"},
        "s": {"type": "score", "criteria": ["lo", "hi"]},
    }))
    .unwrap();
    let twice = 2f32.ln();
    let near = |v: &Value, want: f64| (v.as_f64().unwrap() - want).abs() < 1e-6;
    assert!(near(&answer(&qs[0], &[twice, 0.0])["noul"], 2.0 / 3.0));
    let score = answer(&qs[1], &[0.0, twice]);
    assert_eq!(score["legend"], json!({"0": "lo", "1": "hi"}));
    assert!(near(&score["score"], 2.0 / 3.0) && near(&score["confidence"], 2.0 / 3.0));
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let body = json!({"model": "jev", "state": {"k": 1},
        "questions": {"c": {"type": "choice", "criteria": {"x": null, "y": null}}}});
    // The prompt passes the mocks' 16-token context.
    let (status, got) = json(&slot, "/v1/systemone", Some(body));
    assert_eq!((status, &got["error"]["param"]), (400, &json!("state")));
    let (status, _) = json(&slot, "/v1/systemone", Some(json!({"questions": {}})));
    assert_eq!(status, 422);
}
