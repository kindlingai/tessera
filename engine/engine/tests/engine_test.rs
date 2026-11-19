//! The engine's decode loop against the interpreter.

use sensai_engine::{
    Allowed, Batch, Engine, Event, Fair, Fault, Guide, Prefix, Reason, Request, Row,
};
use sensai_graph::{Graph, Op};
use sensai_interp::{Store, Tensor, run_kept};
use sensai_lower::{Recipe, Settings, Weight, kept, rules};
use sensai_models::decoder::{Config, Moe, Swiglu, stepper};
use sensai_models::{mla, small};
use sensai_run::testing::runs;
use sensai_testkit::{Noise, weights};
use sensai_trace::DimName::Picks;
use sensai_trace::Norm;
use std::sync::{Arc, Mutex};

/// Tokens, positions and slots of `rows`.
fn feeds(rows: &[Row]) -> Vec<Tensor> {
    let column = |of: fn(&Row) -> u32| Tensor::u32(vec![rows.len()], rows.iter().map(of).collect());
    vec![
        column(|r| r.token),
        column(|r| r.position),
        column(|r| r.slot),
    ]
}

/// The interpreter's likeliest id per row of one launch.
fn step(g: &Graph, weights: &[Tensor], store: &mut Store, rows: &[Row]) -> Vec<u32> {
    let got = run_kept(g, weights, store, 0, &feeds(rows)).unwrap();
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
fn open(seed: u64, slots: u64, settings: &Settings) -> (Graph, Vec<Tensor>, Engine) {
    open_on(seed, slots, false, Recipe::default(), settings)
}

/// The graph stays unmasked. The engine's takes masks if `masked`.
fn open_on(
    seed: u64,
    slots: u64,
    masked: bool,
    recipe: Recipe,
    settings: &Settings,
) -> (Graph, Vec<Tensor>, Engine) {
    let mut c = Config::toy(slots, 16);
    let g = stepper::<Norm, Swiglu>(&c);
    let weights = weights(&g, &mut Noise(seed));
    c.stack.masked = masked;
    let mut engine = Engine::open(stepper::<Norm, Swiglu>(&c), 16, &recipe, settings).unwrap();
    let stored: Vec<Weight> = weights.iter().cloned().map(Weight::Floats).collect();
    engine.load(&stored).unwrap();
    (g, weights, engine)
}

/// The events of one answer to `request`.
fn events(engine: &mut Engine, request: Request) -> Vec<Event> {
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
    for settings in runs() {
        answers_match(&settings);
    }
}

/// One answer, then a stopped and an overlong one.
fn answers_match(settings: &Settings) {
    let (g, weights, mut engine) = open(11, 1, settings);
    let prompt = [3u32, 9, 4];
    let want = reference(&g, &weights, &prompt, 6);
    let request = |max_tokens, stop| Request {
        tokens: prompt.to_vec(),
        max_tokens,
        stop,
        guide: None,
    };
    let got = events(&mut engine, request(6, Vec::new()));
    assert_eq!(
        (ids(&got), got.last()),
        (want.clone(), Some(&Event::Done(Reason::Length)))
    );
    // A stop id ends the answer, left out.
    let got = events(&mut engine, request(6, vec![want[2]]));
    assert_eq!(
        (ids(&got).len(), got.last()),
        (2, Some(&Event::Done(Reason::Stop)))
    );
    // The context fills before a long answer ends.
    assert_eq!(
        events(&mut engine, request(100, Vec::new())).last(),
        Some(&Event::Done(Reason::Context))
    );
}

#[test]
fn rows_mix_slots() {
    // The interpreter keeps rows densely. Layouts and rules must agree.
    let mut recipes = Vec::new();
    for kept in [kept::DENSE, kept::INTERLEAVED] {
        recipes.push(Recipe {
            kept,
            ..Recipe::default()
        });
    }
    for rule in rules() {
        recipes.push(Recipe {
            favor: Some(rule.name.into()),
            ..Recipe::default()
        });
    }
    let runs = runs();
    let pairs = runs
        .iter()
        .flat_map(|s| recipes.iter().map(move |r| (s, r)));
    for (settings, recipe) in pairs {
        let (g, weights, mut engine) = open_on(13, 4, false, recipe.clone(), settings);
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
            assert_eq!(engine.step(&rows).unwrap(), want, "{rows:?} on {recipe:?}");
        }
        // Slots and positions past the graph's bounds are refused.
        assert!(engine.step(&[row(4, 1, 0)]).is_err());
        assert!(engine.step(&[row(0, 1, 16)]).is_err());
    }
}

/// Several answers through one batch, each against its own reference.
fn batched(seed: u64, slots: u64, budget: usize, cancel_after: Option<usize>, s: &Settings) {
    let (g, weights, mut engine) = open(seed, slots, s);
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
            guide: None,
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
    for s in runs() {
        // More answers than slots, with whole prompts in one step.
        batched(17, 3, 64, None, &s);
        // Prompts in chunks of two rows, shared across answers.
        batched(19, 2, 2, None, &s);
        // Cancelled answers free their slots for the rest.
        batched(23, 2, 64, Some(2), &s);
    }
}

/// Low-rank attention computes its keys at each launch.
#[test]
fn mla_matches_interpreter() {
    let graphs = [
        mla::stepper::<mla::Moe>(&small::deepseek()),
        mla::stepper::<mla::RotatedMoe>(&small::glm53()),
    ];
    let launches = [
        vec![row(1, 5, 0), row(1, 7, 1), row(0, 9, 0)],
        vec![row(0, 4, 1), row(1, 3, 2)],
        vec![row(1, 8, 3), row(0, 2, 2), row(0, 6, 3), row(0, 1, 4)],
    ];
    for settings in runs() {
        for g in &graphs {
            let weights = weights(g, &mut Noise(3));
            let stored: Vec<Weight> = weights.iter().cloned().map(Weight::Floats).collect();
            let mut engine = Engine::open(g.clone(), 16, &Recipe::default(), &settings).unwrap();
            engine.load(&stored).unwrap();
            let mut store = Store::default();
            for rows in &launches {
                let want = step(g, &weights, &mut store, rows);
                assert_eq!(engine.step(rows).unwrap(), want);
            }
        }
    }
}

#[test]
fn kept_rows_resume() {
    for s in runs() {
        let (g, weights, mut engine) = open(31, 2, &s);
        let mut batch = Batch::new(&mut engine, Box::new(Fair));
        batch.reuse();
        batch.budget(1);
        let first = reference(&g, &weights, &[3, 9, 4], 3);
        let longer = [&[3, 9, 4], &first[..]].concat();
        // Each prompt, and the steps its new rows take.
        let cases: [(&[u32], usize); 3] = [(&[3, 9, 4], 5), (&longer, 3), (&[3, 9, 7, 1], 4)];
        for (prompt, steps) in cases {
            let (events, taken) = std::sync::mpsc::channel();
            let request = Request {
                tokens: prompt.to_vec(),
                max_tokens: 3,
                ..Request::default()
            };
            batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
            let mut ran = 0;
            while batch.step().unwrap() {
                ran += 1;
            }
            let got: Vec<Event> = taken.try_iter().collect();
            let want = reference(&g, &weights, prompt, 3);
            assert_eq!((ids(&got), ran), (want, steps));
        }
    }
}

/// Trimmed steps return the whole graph's rows at their picks.
#[test]
fn trim_keeps_picks() {
    let graphs = [
        stepper::<Norm, Swiglu>(&small::qwen3()),
        stepper::<(), Moe>(&small::granite()),
        mla::stepper::<mla::Moe>(&small::deepseek()),
        mla::stepper::<mla::RotatedMoe>(&small::glm53()),
    ];
    let launches = [
        (vec![row(1, 5, 0), row(1, 7, 1), row(0, 9, 0)], vec![1, 2]),
        (vec![row(0, 4, 1), row(1, 3, 2)], vec![0, 1]),
    ];
    for g in graphs {
        let trimmed = sensai_graph::trim(&g, 0).unwrap();
        let picked = |n: &sensai_graph::Node| n.op == Op::Matmul && n.ty.shape.0[0] == Picks.into();
        assert!(trimmed.nodes.iter().any(picked));
        let weights = weights(&g, &mut Noise(3));
        let (mut whole, mut cut) = (Store::default(), Store::default());
        for (rows, picks) in &launches {
            let mut feeds = feeds(rows);
            let want = run_kept(&g, &weights, &mut whole, 0, &feeds).unwrap();
            feeds.push(Tensor::u32(vec![picks.len()], picks.clone()));
            let got = run_kept(&trimmed, &weights, &mut cut, 0, &feeds).unwrap();
            let (want, got) = (want[1].floats().unwrap(), got[1].floats().unwrap());
            let width = got.len() / picks.len();
            for (at, p) in picks.iter().enumerate() {
                let p = *p as usize;
                assert_eq!(got[at * width..][..width], want[p * width..][..width]);
            }
        }
    }
}

/// Allows one fixed set, and keeps each accepted id.
struct Fixed(Allowed, Arc<Mutex<Vec<u32>>>);

impl Guide for Fixed {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(self.0.clone())
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        self.1.lock().map_err(|_| "poisoned")?.push(id);
        Ok(())
    }
}

#[test]
fn heads_agree() {
    let mut c = Config::toy(1, 16);
    c.stack.masked = true;
    let trimmed = sensai_graph::trim(&stepper::<Norm, Swiglu>(&c), 0).unwrap();
    assert_eq!(trimmed.entries[0].inputs[3].ty.shape.0[0], Picks.into());
    let set = [2u32, 11, 17, 30];
    for s in runs() {
        let (g, weights, mut engine) = open_on(29, 1, true, Recipe::default(), &s);
        let rows = [row(0, 3, 0), row(0, 9, 1)];
        let got = run_kept(&g, &weights, &mut Store::default(), 0, &feeds(&rows)).unwrap();
        let logits = got[1].floats().unwrap();
        let score = |id: &&u32| logits[32 + **id as usize];
        let best = *set
            .iter()
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .unwrap();
        // The sparse head, and the full head without the rest.
        let except = Allowed::Except((0..32).filter(|i| !set.contains(i)).collect());
        for allowed in [Allowed::Few(set.to_vec()), except] {
            assert_eq!(engine.last_guided(&rows, &[allowed]).unwrap(), [best]);
        }
        // The host makes single choices. Lists pick among their ids.
        let seen = Arc::default();
        let request = |allowed| Request {
            tokens: vec![3, 9],
            max_tokens: 3,
            guide: Some(Box::new(Fixed(allowed, Arc::clone(&seen)))),
            ..Request::default()
        };
        assert_eq!(ids(&events(&mut engine, request(Allowed::One(5)))), [5; 3]);
        let got = ids(&events(&mut engine, request(Allowed::Few(set.to_vec()))));
        assert!(got[0] == best && got.iter().all(|i| set.contains(i)));
        assert_eq!(seen.lock().unwrap().len(), 6);
    }
}

#[test]
fn prefix_comes_first() {
    for s in runs() {
        let (g, weights, mut engine) = open(31, 1, &s);
        let want = reference(&g, &weights, &[3, 9, 5, 7], 3);
        let request = Request {
            tokens: vec![3, 9],
            max_tokens: 5,
            guide: Some(Box::new(Prefix(vec![5, 7]))),
            ..Request::default()
        };
        assert_eq!(
            ids(&events(&mut engine, request)),
            [&[5, 7], &want[..]].concat()
        );
    }
}

#[test]
fn budget_refuses_plan() {
    let g = || stepper::<Norm, Swiglu>(&Config::toy(1, 16));
    let open = |memory| {
        let settings = Settings {
            memory,
            ..sensai_run::testing::host()
        };
        Engine::open(g(), 16, &Recipe::default(), &settings)
    };
    let refused = open(sensai_engine::Memory::Bytes(1024));
    assert!(matches!(
        refused,
        Err(sensai_engine::Error::Run(
            sensai_run::Error::OverBudget { .. }
        ))
    ));
    assert!(open(sensai_engine::Memory::Off).is_ok());
}

/// Fails on its first pick.
struct Broken;

impl Guide for Broken {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(Allowed::Any)
    }

    fn accept(&mut self, _: u32) -> Result<(), Fault> {
        Err("broken".into())
    }
}

#[test]
fn failures_stay_local() {
    let (_, _, mut engine) = open(37, 2, &runs()[0]);
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    let mut taken = Vec::new();
    for guide in [Some(Box::new(Broken) as Box<dyn Guide>), None] {
        let (events, receiver) = std::sync::mpsc::channel();
        let request = Request {
            tokens: vec![3, 9],
            max_tokens: 3,
            guide,
            ..Request::default()
        };
        batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
        taken.push(receiver);
    }
    while batch.step().unwrap() {}
    let last = |r: &std::sync::mpsc::Receiver<Event>| r.try_iter().last();
    let ends: Vec<_> = taken.iter().map(last).collect();
    let want = [Reason::Guide, Reason::Length].map(|r| Some(Event::Done(r)));
    assert_eq!(ends, want);
}
