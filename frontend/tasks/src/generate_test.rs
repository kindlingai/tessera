use crate::api::{Generator, Params, Refused};
use crate::generate::{Generate, Queued};
use std::sync::{Arc, Mutex, mpsc};
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Penalties, Sampling};

#[test]
fn greedy_engine_refusals() {
    let file = br#"{"model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"}}"#;
    let generate = Generate {
        tokenizer: Hf::from_json(file).unwrap(),
        template: None,
        response: None,
        stop: Vec::new(),
        guides: None,
        edits: false,
        sampling: false,
    };
    let (jobs, _queue) = mpsc::channel();
    let queued = Queued {
        generate: Arc::new(generate),
        jobs: Mutex::new(jobs),
    };
    let penalties = Penalties {
        presence: 0.5,
        ..Penalties::default()
    };
    let sampling = Sampling {
        temperature: 0.8,
        ..Sampling::default()
    };
    let cases = [
        (
            Params {
                penalties,
                ..Params::default()
            },
            "presence_penalty",
        ),
        (
            Params {
                sampling,
                ..Params::default()
            },
            "temperature",
        ),
    ];
    for (params, field) in cases {
        let error = queued
            .generate(vec![0], params, Box::new(|_| true))
            .unwrap_err();
        assert_eq!(error.downcast_ref::<Refused>().unwrap().1, field);
    }
}
