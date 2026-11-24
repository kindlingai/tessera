//! The engine's decode loop against the interpreter.

use std::sync::{Arc, Mutex};
use tessera_engine::{
    Allowed, Batch, Compute, Engine, Event, Fair, Fault, Guide, Prefix, Read, Reason, Request, Row,
    Site, Steps,
};
use tessera_graph::{Graph, Op, Role};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::{Pins, Recipe, Settings, Weight, kept, rules};
use tessera_models::decoder::{Config, Moe, Swiglu, stepper};
use tessera_models::{mla, small};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::DimName::Picks;
use tessera_trace::Norm;

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
    open_on(seed, slots, Steps::Plain, Recipe::default(), settings)
}

/// The graph stays plain. The engine's takes `steps`.
fn open_on(
    seed: u64,
    slots: u64,
    steps: Steps,
    recipe: Recipe,
    settings: &Settings,
) -> (Graph, Vec<Tensor>, Engine) {
    open_sized(seed, [slots, 16], steps, recipe, settings)
}

/// As `open_on`, with `[slots, context]`.
fn open_sized(
    seed: u64,
    [slots, context]: [u64; 2],
    steps: Steps,
    recipe: Recipe,
    settings: &Settings,
) -> (Graph, Vec<Tensor>, Engine) {
    let mut c = Config::toy(slots, context);
    let g = stepper::<Norm, Swiglu>(&c);
    let weights = weights(&g, &mut Noise(seed));
    c.stack.steps = steps;
    let graph = stepper::<Norm, Swiglu>(&c);
    let mut engine = Engine::open(graph, context as usize, &recipe, settings).unwrap();
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
        _ => None,
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
        ..Request::default()
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

/// Profiling names every step's kernel and every launch.
#[test]
fn profile_counts_launches() {
    for settings in runs() {
        let settings = Settings {
            profile: Some(1000),
            ..settings
        };
        let (g, weights, mut engine) = open(17, 1, &settings);
        let prompt = [3u32, 9, 4];
        let want = reference(&g, &weights, &prompt, 3);
        let request = Request {
            tokens: prompt.to_vec(),
            max_tokens: 3,
            ..Request::default()
        };
        assert_eq!(ids(&events(&mut engine, request)), want);
        let profile = engine.take_profile().unwrap();
        assert_eq!(profile.spent["engine: launch"].count, 3);
        let kernels: f64 = engine
            .kernels()
            .keys()
            .map(|k| profile.spent[*k].seconds)
            .sum();
        assert!(kernels > 0.0, "{}", engine.device());
    }
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
        let (g, weights, mut engine) = open_on(13, 4, Steps::Plain, recipe.clone(), settings);
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
            ..Request::default()
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

/// Multi-slot MLA stays exact when context bucket < max_context.
#[test]
fn mla_multislot_bucketed() {
    let mut cfg = small::glm53();
    cfg.stack.max_context = 512;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let weights = weights(&g, &mut Noise(7));
    let stored: Vec<Weight> = weights.iter().cloned().map(Weight::Floats).collect();
    let prompts: [&[u32]; 4] = [&[3, 9, 4, 11], &[5, 2, 8], &[12, 1, 6, 7, 10], &[2, 14]];
    for settings in runs() {
        let mut solo = Engine::open(g.clone(), 512, &Recipe::default(), &settings).unwrap();
        solo.load(&stored).unwrap();
        let want: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                let mut b = Batch::new(&mut solo, Box::new(Fair));
                asked(&mut b, &[p]).0.pop().unwrap()
            })
            .collect();
        for paged in [false, true] {
            let mut recipe = Recipe::default();
            if paged {
                recipe.kept = &kept::PAGED16;
            }
            let mut multi = Engine::open(g.clone(), 512, &recipe, &settings).unwrap();
            multi.load(&stored).unwrap();
            let mut batch = Batch::new(&mut multi, Box::new(Fair));
            if paged {
                batch.reuse();
            }
            let (got, _) = asked(&mut batch, &prompts);
            assert_eq!(got, want, "paged={paged}");
        }
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

/// Three ids after each prompt, and the steps they took.
fn asked(batch: &mut Batch, prompts: &[&[u32]]) -> (Vec<Vec<u32>>, usize) {
    let mut taken = Vec::new();
    for prompt in prompts {
        let (events, receiver) = std::sync::mpsc::channel();
        let request = Request {
            tokens: prompt.to_vec(),
            max_tokens: 3,
            ..Request::default()
        };
        batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
        taken.push(receiver);
    }
    let mut steps = 0;
    while batch.step().unwrap() {
        steps += 1;
    }
    let got = |r: &std::sync::mpsc::Receiver<Event>| ids(&r.try_iter().collect::<Vec<_>>());
    (taken.iter().map(got).collect(), steps)
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
        // Each prompt, and the step count for its new rows.
        let cases: [(&[u32], usize); 3] = [(&[3, 9, 4], 5), (&longer, 3), (&[3, 9, 7, 1], 4)];
        for (prompt, steps) in cases {
            let want = reference(&g, &weights, prompt, 3);
            assert_eq!(asked(&mut batch, &[prompt]), (vec![want], steps));
        }
        // The engine tells the sink the rows it kept.
        let (events, taken) = std::sync::mpsc::channel();
        let request = Request {
            tokens: longer,
            max_tokens: 1,
            ..Request::default()
        };
        batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
        batch.step().unwrap();
        assert_eq!(taken.try_iter().next(), Some(Event::Kept(2)));
        let totals = batch.totals().unwrap();
        assert_eq!((totals.hits, totals.misses), (3, 1));
        assert_eq!((totals.pool.size, totals.pool.active), (32, 16));
    }
}

#[test]
fn paged_slots_share() {
    let recipe = Recipe {
        kept: &kept::PAGED16,
        ..Recipe::default()
    };
    for s in runs() {
        let (g, weights, mut engine) = open_sized(37, [2, 64], Steps::Plain, recipe.clone(), &s);
        let mut batch = Batch::new(&mut engine, Box::new(Fair));
        batch.reuse();
        batch.budget(1);
        let base: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let (a, b) = (
            [&base[..18], &[1, 2]].concat(),
            [&base[..17], &[5]].concat(),
        );
        // Both later prompts start at the kept answer's first page.
        let rounds: [(&[&[u32]], usize); 2] = [(&[&base], 22), (&[&a, &b], 10)];
        for (prompts, steps) in rounds {
            let want = prompts.iter().map(|p| reference(&g, &weights, p, 3));
            assert_eq!(asked(&mut batch, prompts), (want.collect(), steps));
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
        let trimmed = tessera_graph::trim(&g, 0).unwrap();
        let picked =
            |n: &tessera_graph::Node| n.op == Op::Matmul && n.ty.shape.0[0] == Picks.into();
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
fn reads_score() {
    for s in runs() {
        let (g, weights, mut engine) = open_on(23, 1, Steps::Scored, Recipe::default(), &s);
        let rows = [row(0, 3, 0), row(0, 9, 1), row(0, 4, 2)];
        let full = run_kept(&g, &weights, &mut Store::default(), 0, &feeds(&rows)).unwrap();
        let logits = full[1].floats().unwrap();
        let read = |row, ids: &[u32]| Read {
            row,
            ids: ids.to_vec(),
        };
        let reads = vec![read(0, &[5, 2]), read(2, &[30, 1, 7])];
        let request = Request {
            tokens: vec![3, 9, 4],
            reads: reads.clone(),
            ..Request::default()
        };
        let got = events(&mut engine, request);
        let [Event::Scores(scores), Event::Done(Reason::Stop)] = got.as_slice() else {
            panic!("{got:?}");
        };
        for (read, scored) in reads.iter().zip(scores) {
            for (id, s) in read.ids.iter().zip(scored) {
                let want = logits[32 * read.row as usize + *id as usize];
                assert!((want - s).abs() < 1e-3 * want.abs().max(1.0), "{want} {s}");
            }
        }
    }
}

#[test]
fn heads_agree() {
    let mut c = Config::toy(1, 16);
    c.stack.steps = Steps::Masked;
    let trimmed = tessera_graph::trim(&stepper::<Norm, Swiglu>(&c), 0).unwrap();
    assert_eq!(trimmed.entries[0].inputs[3].ty.shape.0[0], Picks.into());
    let set = [2u32, 11, 17, 30];
    for s in runs() {
        let (g, weights, mut engine) = open_on(29, 1, Steps::Masked, Recipe::default(), &s);
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
            ..tessera_run::testing::host()
        };
        Engine::open(g(), 16, &Recipe::default(), &settings)
    };
    let refused = open(tessera_engine::Memory::Bytes(1024));
    assert!(matches!(
        refused,
        Err(tessera_engine::Error::Run(
            tessera_run::Error::OverBudget { .. }
        ))
    ));
    assert!(open(tessera_engine::Memory::Off).is_ok());
}

#[test]
fn unfilled_port_refuses() {
    let mut g = stepper::<Norm, Swiglu>(&Config::toy(1, 16));
    g.entries[0].inputs[2].role = Role::Keys;
    let refused = Engine::open(g, 16, &Recipe::default(), &tessera_run::testing::host());
    assert!(matches!(
        refused,
        Err(tessera_engine::Error::Port {
            entry: 0,
            role: Role::Keys
        })
    ));
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

/// Pinned bf16 activations and bf16/fp8 KV match f32 logits.
#[test]
fn pinned_decoder_matches() {
    use tessera_testkit::cosine_f32;
    let prompt = [3u32, 11, 5, 2];
    let configs = [
        (
            Pins::all(BF16_FORM)
                .activations(BF16_FORM)
                .kept(BF16_FORM)
                .site(Site::Norm, F32_FORM)
                .site(Site::Logits, F32_FORM),
            Compute::Bf16,
            0.999f32,
        ),
        (
            Pins::all(F32_FORM)
                .activations(BF16_FORM)
                .kept(FP8_FORM)
                .site(Site::Norm, F32_FORM)
                .site(Site::Logits, F32_FORM),
            Compute::Exact,
            0.98f32,
        ),
    ];
    for (pins, compute, min_cos) in configs {
        let recipe = Recipe {
            pins: pins.clone(),
            compute,
            ..Recipe::default()
        };
        for settings in runs() {
            let (g, weights, mut engine) = open_on(41, 1, Steps::Scored, recipe.clone(), &settings);
            let prefill: Vec<Row> = prompt
                .iter()
                .enumerate()
                .map(|(i, &t)| row(0, t, i as u32))
                .collect();
            let mut store = Store::default();
            let want = run_kept(&g, &weights, &mut store, 0, &feeds(&prefill)).unwrap();
            let want_logits = want[1].floats().unwrap();
            let vocab: Vec<u32> = (0..32).collect();
            let reads: Vec<Read> = (0..prompt.len() as u32)
                .map(|row| Read {
                    row,
                    ids: vocab.clone(),
                })
                .collect();
            let score_req = Request {
                tokens: prompt.to_vec(),
                reads,
                ..Request::default()
            };
            let got = events(&mut engine, score_req);
            let Event::Scores(scores) = &got[0] else {
                panic!("{got:?}");
            };
            let flat: Vec<f32> = scores.iter().flatten().copied().collect();
            let cos = cosine_f32(&flat, want_logits);
            assert!(
                cos > min_cos,
                "{} on {:?}: cos {cos}",
                pins.kept.name(),
                settings.backend
            );
            // Multi-step decode through the narrow KV cache completes cleanly.
            let gen_req = Request {
                tokens: prompt.to_vec(),
                max_tokens: 4,
                ..Request::default()
            };
            let decoded = ids(&events(&mut engine, gen_req));
            assert_eq!(decoded.len(), 4, "{decoded:?}");
            let want_ids = reference(&g, &weights, &prompt, 4);
            if pins.kept == BF16_FORM {
                assert_eq!(
                    decoded, want_ids,
                    "bf16 greedy decode on {:?}",
                    settings.backend
                );
            }
        }
    }
}
