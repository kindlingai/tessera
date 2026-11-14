//! The engine's decode loop against the interpreter.

use engine::{Batch, Engine, Event, Fair, Reason, Request, Row};
use gpukit::testing::Noise;
use graph::Graph;
use interp::{Store, Tensor, run_kept};
use lower::{Recipe, Weight, kept};
use models::qwen3::{Config, stepper};

fn weights(g: &Graph, noise: &mut Noise) -> Vec<Tensor> {
    let size = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().as_const().unwrap() as usize;
    let fill = |w: &graph::Weight| {
        let shape: Vec<usize> = w.ty.shape.0.iter().map(size).collect();
        Tensor::f32(
            shape.clone(),
            (0..shape.iter().product()).map(|_| noise.draw()).collect(),
        )
    };
    g.weights.iter().map(fill).collect()
}

/// The interpreter's likeliest id per row of one launch.
fn step(g: &Graph, weights: &[Tensor], store: &mut Store, rows: &[Row]) -> Vec<u32> {
    let column = |of: fn(&Row) -> u32| Tensor::u32(vec![rows.len()], rows.iter().map(of).collect());
    let feeds = [
        column(|r| r.token),
        column(|r| r.position),
        column(|r| r.slot),
    ];
    let got = run_kept(g, weights, store, 0, &feeds).unwrap();
    got[0]
        .indices()
        .unwrap()
        .iter()
        .map(|i| *i as u32)
        .collect()
}

/// Greedy decoding through the interpreter in slot 0.
fn reference(g: &Graph, weights: &[Tensor], prompt: &[u32], n: usize) -> Vec<u32> {
    let (mut store, mut out, mut next) = (Store::default(), Vec::new(), prompt.to_vec());
    for _ in 0..n {
        let past = out.len() + prompt.len() - next.len();
        let row = |(i, t): (usize, &u32)| Row {
            slot: 0,
            token: *t,
            position: (past + i) as u32,
        };
        let rows: Vec<Row> = next.iter().enumerate().map(row).collect();
        next = vec![*step(g, weights, &mut store, &rows).last().unwrap()];
        out.push(next[0]);
    }
    out
}

/// A toy engine with `slots` lanes, its graph and weights.
fn open(seed: u64, slots: u64) -> (Graph, Vec<Tensor>, Engine) {
    open_on(seed, slots, Recipe::default())
}

fn open_on(seed: u64, slots: u64, recipe: Recipe) -> (Graph, Vec<Tensor>, Engine) {
    let c = Config::toy(slots, 16);
    let g = stepper(&c);
    let weights = weights(&g, &mut Noise(seed));
    let mut engine = Engine::open(stepper(&c), 16, &recipe, &settings::read()).unwrap();
    let stored: Vec<Weight> = weights.iter().cloned().map(Weight::Floats).collect();
    engine.load(&stored).unwrap();
    (g, weights, engine)
}

/// The events of one answer to `request`.
fn events(engine: &mut Engine, request: &Request) -> Vec<Event> {
    let mut events = Vec::new();
    engine.generate(request, &mut |e| events.push(e)).unwrap();
    events
}

fn ids(events: &[Event]) -> Vec<u32> {
    let id = |e: &Event| match e {
        Event::Token(id) => Some(*id),
        Event::Done(_) => None,
    };
    events.iter().filter_map(id).collect()
}

fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

#[test]
fn engine_matches_interpreter() {
    let (g, weights, mut engine) = open(11, 1);
    let prompt = [3u32, 9, 4];
    let want = reference(&g, &weights, &prompt, 6);
    let request = Request {
        tokens: prompt.to_vec(),
        max_tokens: 6,
        stop: Vec::new(),
    };
    let got = events(&mut engine, &request);
    assert_eq!(
        (ids(&got), got.last()),
        (want.clone(), Some(&Event::Done(Reason::Length)))
    );
    // A stop id ends the answer, left out.
    let stopped = Request {
        stop: vec![want[2]],
        ..request.clone()
    };
    let got = events(&mut engine, &stopped);
    assert_eq!(
        (ids(&got).len(), got.last()),
        (2, Some(&Event::Done(Reason::Stop)))
    );
    // The context fills before a long answer ends.
    let long = Request {
        max_tokens: 100,
        ..request
    };
    assert_eq!(
        events(&mut engine, &long).last(),
        Some(&Event::Done(Reason::Context))
    );
}

#[test]
fn rows_mix_slots() {
    // The interpreter keeps rows densely. Any layout must agree.
    for kept in [kept::DENSE, kept::INTERLEAVED] {
        let recipe = Recipe {
            kept,
            ..Recipe::default()
        };
        let (g, weights, mut engine) = open_on(13, 4, recipe);
        // Two prompts at once, decodes side by side, then alone.
        let launches = [
            vec![row(2, 5, 0), row(2, 7, 1), row(0, 9, 0)],
            vec![row(0, 4, 1), row(2, 3, 2)],
            vec![row(2, 6, 3)],
            vec![row(0, 8, 2), row(2, 1, 4)],
        ];
        let mut store = Store::default();
        for rows in launches {
            let want = step(&g, &weights, &mut store, &rows);
            assert_eq!(engine.step(&rows).unwrap(), want, "{rows:?} on {kept:?}");
        }
        // Slots and positions past the graph's bounds are refused.
        assert!(engine.step(&[row(4, 1, 0)]).is_err());
        assert!(engine.step(&[row(0, 1, 16)]).is_err());
    }
}

/// Several answers through one batch, each against its own reference.
fn batched(seed: u64, slots: u64, budget: usize, cancel_after: Option<usize>) {
    let (g, weights, mut engine) = open(seed, slots);
    let prompts: [&[u32]; 5] = [
        &[3, 9, 4],
        &[1, 2, 3, 4, 5, 6],
        &[7],
        &[8, 8],
        &[2, 4, 6, 8],
    ];
    let wants: Vec<Vec<u32>> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| reference(&g, &weights, p, 3 + i))
        .collect();
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    batch.budget(budget);
    let mut taken = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let request = Request {
            tokens: prompt.to_vec(),
            max_tokens: 3 + i,
            stop: Vec::new(),
        };
        let (events, receiver) = std::sync::mpsc::channel();
        let mut sent = 0;
        batch.submit(
            request,
            Box::new(move |e| {
                sent += 1;
                events.send(e).is_ok() && cancel_after.is_none_or(|n| sent < n)
            }),
        );
        taken.push(receiver);
    }
    while batch.step().unwrap() {}
    for (i, receiver) in taken.into_iter().enumerate() {
        let got: Vec<Event> = receiver.try_iter().collect();
        match cancel_after {
            None => assert_eq!(
                (ids(&got), got.last()),
                (wants[i].clone(), Some(&Event::Done(Reason::Length))),
                "answer {i}"
            ),
            Some(n) => assert_eq!(
                (ids(&got), got.last()),
                (wants[i][..n].to_vec(), Some(&Event::Done(Reason::Stop))),
                "answer {i}"
            ),
        }
    }
}

#[test]
fn batch_matches_single() {
    // More answers than slots, with whole prompts in one step.
    batched(17, 3, 64, None);
    // Prompts in chunks of two rows, shared across answers.
    batched(19, 2, 2, None);
    // Cancelled answers free their slots for the rest.
    batched(23, 2, 64, Some(2));
}
