//! Seeded sampling and logit edits through the batch scheduler.

use tessera_engine::{
    Allowed, Batch, Edit, Engine, Event, Fair, Fault, Guide, Penalties, Recipe, Request, Sampling,
};
use tessera_lower::{Settings, Weight};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

fn answer(engine: &mut Engine, req: Request) -> Vec<u32> {
    let mut batch = Batch::new(engine, Box::new(Fair));
    let (tx, rx) = std::sync::mpsc::channel();
    batch.submit(req, Box::new(move |e| tx.send(e).is_ok()));
    while batch.step().unwrap() {}
    let mut ids = Vec::new();
    for e in rx.try_iter() {
        if let Event::Token(id) = e {
            ids.push(id);
        }
    }
    ids
}

fn request(sampling: Sampling, guide: Option<Box<dyn Guide>>) -> Request {
    Request {
        tokens: vec![3, 9, 4],
        max_tokens: 6,
        sampling,
        guide,
        ..Request::default()
    }
}

fn engine() -> Engine {
    let g = stepper::<Norm, Swiglu>(&Config::toy(2, 16));
    let stored: Vec<Weight> = weights(&g, &mut Noise(41))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let mut engine = Engine::open(g, 16, &Recipe::default(), &Settings::default()).unwrap();
    engine.load(&stored).unwrap();
    engine
}

/// Allows any token and makes fixed edits.
struct Edits(Vec<Edit>);

impl Guide for Edits {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(Allowed::Any)
    }

    fn accept(&mut self, _: u32) -> Result<(), Fault> {
        Ok(())
    }

    fn edits(&mut self) -> Vec<Edit> {
        self.0.clone()
    }
}

#[test]
fn sampling_respects_seed() {
    let mut engine = engine();
    let sampling = |seed| Sampling {
        temperature: 0.9,
        top_p: 0.95,
        top_k: 4,
        min_p: 0.05,
        seed: Some(seed),
    };
    let a = answer(&mut engine, request(sampling(42), None));
    let b = answer(&mut engine, request(sampling(42), None));
    assert_eq!(a.len(), 6);
    assert_eq!(a, b);
}

#[test]
fn edits_steer_greedy() {
    let mut engine = engine();
    let greedy = Sampling::default();
    let plain = answer(&mut engine, request(greedy, None));
    let none = Some(Box::new(Edits(Vec::new())) as Box<dyn Guide>);
    assert_eq!(answer(&mut engine, request(greedy, none)), plain);
    let ban = Edit {
        id: plain[0],
        add: -1e9,
        rep: 1.0,
    };
    let banned = answer(
        &mut engine,
        request(greedy, Some(Box::new(Edits(vec![ban])))),
    );
    assert!(!banned.contains(&plain[0]), "{banned:?}");
    let once = Penalties {
        presence: 1e9,
        ..Penalties::default()
    };
    let mut fresh = answer(&mut engine, request(greedy, once.guide(&[])));
    fresh.sort_unstable();
    fresh.dedup();
    assert_eq!(fresh.len(), 6);
}
