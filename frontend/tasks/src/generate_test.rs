use crate::Error;
use crate::api::{Format, Generator, Params, Refused};
use crate::generate::{Generate, Queued, Steer};
use crate::guided::Guides;
use std::sync::{Arc, Mutex, mpsc};
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Guide, Penalties, Prefix, Sampling};

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
}
