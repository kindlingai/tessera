use crate::Error;
use crate::api::{Finish, Format, Generator, Params, Refused, Step, ToolChoice};
use crate::generate::{Generate, Queued, Steer};
use crate::guided::Guides;
use std::sync::{Arc, Mutex, mpsc};
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Event, Guide, Penalties, Prefix, Reason, Sampling};

struct Open;

impl Guides for Open {
    fn guide(&self, _: &Format) -> Result<Box<dyn Guide>, Error> {
        Ok(Box::new(Prefix(Vec::new())))
    }
}

#[test]
fn speculative_engine_gate() {
    let file = br#"{"model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}}"#;
    let (jobs, queue) = mpsc::channel();
    let queued = |steer| Queued {
        generate: Arc::new(Generate {
            tokenizer: Hf::from_json(file).unwrap(),
            template: None,
            response: None,
            stop: Vec::new(),
            steer,
            sampling: false,
        }),
        jobs: Mutex::new(jobs.clone()),
        sent: None,
    };
    let solo = queued(Steer::Guided(Some(Arc::new(Open))));
    let batched = queued(Steer::Unguided("batched speculative decoding"));
    let penalties = Penalties {
        presence: 0.5,
        ..Penalties::default()
    };
    let sampling = Sampling {
        temperature: 0.8,
        ..Sampling::default()
    };
    let format = Params {
        format: Some(Format::Object),
        penalties,
        ..Params::default()
    };
    solo.generate(vec![0], format.clone(), Box::new(|_| true))
        .unwrap();
    assert!(queue.try_recv().unwrap().0.guide.is_some());
    let cases = [
        (&batched, format, "response_format"),
        (
            &batched,
            Params {
                penalties,
                ..Params::default()
            },
            "presence_penalty",
        ),
        (
            &batched,
            Params {
                min_tokens: 4,
                ..Params::default()
            },
            "min_tokens",
        ),
        (
            &batched,
            Params {
                tool_choice: ToolChoice::Required,
                ..Params::default()
            },
            "tool_choice",
        ),
        (
            &solo,
            Params {
                sampling,
                ..Params::default()
            },
            "temperature",
        ),
    ];
    for (queued, params, field) in cases {
        let error = queued
            .generate(vec![0], params, Box::new(|_| true))
            .unwrap_err();
        assert_eq!(error.downcast_ref::<Refused>().unwrap().1, field);
    }
    check_stop_token_identity();
}

fn check_stop_token_identity() {
    let file = r#"{"model":{"type":"WordLevel","vocab":{"preface ":0,"<think>é":1},"unk_token":"preface "},"decoder":{"type":"Fuse"}}"#
        .as_bytes();
    let (jobs, queue) = mpsc::channel();
    let queued = Queued {
        generate: Arc::new(Generate {
            tokenizer: Hf::from_json(file).unwrap(),
            template: None,
            response: None,
            stop: Vec::new(),
            steer: Steer::Guided(None),
            sampling: false,
        }),
        jobs: Mutex::new(jobs),
        sent: None,
    };
    let (tx, rx) = mpsc::channel();
    let params = Params {
        stop: vec!["ZZ".into()],
        ..Params::default()
    };
    queued
        .generate(vec![0], params, Box::new(move |step| tx.send(step).is_ok()))
        .unwrap();
    let (_, mut engine_sink) = queue.recv().unwrap();
    assert!(engine_sink(Event::Token(0)));
    assert!(engine_sink(Event::Token(1)));
    assert!(!engine_sink(Event::Done(Reason::Stop)));
    let steps: Vec<Step> = rx.try_iter().collect();
    assert_eq!(
        steps,
        vec![
            Step::Text {
                token: 1,
                text: "preface".into()
            },
            Step::Text {
                token: 1,
                text: " ".into()
            },
            Step::Text {
                token: 2,
                text: "<think>".into()
            },
            Step::Text {
                token: 2,
                text: "é".into()
            },
            Step::Done {
                text: String::new(),
                finish: Finish::Stop,
                tokens: 2,
                cached: 0,
            },
        ]
    );
}
