//! The engine's decode loop against the interpreter.

mod common;

use common::{
    collect_ids as ids, gen_events as events, open_engine, open_target, req, row,
    step_feeds as feeds,
};
use std::sync::{Arc, Mutex};
use tessera_engine::{
    Allowed, Asks, Batch, Compute, Engine, Event, Fair, Fault, Guide, Prefix, Read, Reason,
    Request, Row, Site, Steps,
};
use tessera_graph::{
    DimName::{Context, Rows, Seqs},
    Graph, Role,
};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::{Device, Pins, Recipe, Settings, kept, lower, rules};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::mla;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::DimName::Picks;
use tessera_trace::Norm;

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
        let rows: Vec<Row> = next
            .iter()
            .enumerate()
            .map(|(i, &t)| row(0, t, (past + i) as u32))
            .collect();
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
    let engine = open_engine(&graph, &weights, context as usize, &recipe, settings);
    (g, weights, engine)
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
    for mut settings in runs() {
        let asks = Asks::default();
        settings.asks = Some(asks.clone());
        let (g, weights, mut engine) = open(17, 1, &settings);
        let prompt = [3u32, 9, 4];
        let want = reference(&g, &weights, &prompt, 3);
        let request = Request {
            tokens: prompt.to_vec(),
            max_tokens: 3,
            ..Request::default()
        };
        let ask = asks.hold();
        assert_eq!(ids(&events(&mut engine, request)), want);
        let profile = engine.take_profile().unwrap();
        drop(ask);
        engine.settle().unwrap();
        assert!(engine.take_profile().is_none());
        assert_eq!(profile.spent["engine: launch"].count, 3);
        let names = engine.kernels();
        let kernel = |row: &str| names.contains_key(row.split(' ').next().unwrap_or(row));
        let rows = profile.spent.iter().filter(|(row, _)| kernel(row));
        let kernels: f64 = rows.map(|(_, s)| s.seconds).sum();
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

fn solo_wants(
    g: &Graph,
    w: &[Tensor],
    ctx: usize,
    s: &Settings,
    prompts: &[&[u32]],
) -> Vec<Vec<u32>> {
    let mut solo = open_target(g, w, ctx, s);
    prompts
        .iter()
        .map(|p| {
            let mut b = Batch::new(&mut solo, Box::new(Fair));
            asked(&mut b, &[p]).0.pop().unwrap()
        })
        .collect()
}

/// Multi-slot MLA stays exact when context bucket < max_context.
#[test]
fn mla_multislot_bucketed() {
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.max_context = 512;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let w = weights(&g, &mut Noise(7));
    let prompts: [&[u32]; 4] = [&[3, 9, 4, 11], &[5, 2, 8], &[12, 1, 6, 7, 10], &[2, 14]];
    for settings in runs() {
        let want = solo_wants(&g, &w, 512, &settings, &prompts);
        for paged in [false, true] {
            let mut recipe = Recipe::default();
            if paged {
                recipe.kept = &kept::PAGED16;
            }
            let mut multi = open_engine(&g, &w, 512, &recipe, &settings);
            let mut batch = Batch::new(&mut multi, Box::new(Fair));
            if paged {
                batch.reuse();
            }
            assert_eq!(asked(&mut batch, &prompts).0, want, "paged={paged}");
        }
    }
}

/// Low-rank attention computes its keys at each launch.
#[test]
fn mla_matches_interpreter() {
    let graphs = [
        mla::stepper::<mla::Moe>(&mla::Config::deepseek_small()),
        mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
    ];
    let launches = [
        vec![row(1, 5, 0), row(1, 7, 1), row(0, 9, 0)],
        vec![row(0, 4, 1), row(1, 3, 2)],
        vec![row(1, 8, 3), row(0, 2, 2), row(0, 6, 3), row(0, 1, 4)],
    ];
    for settings in runs() {
        for g in &graphs {
            let w = weights(g, &mut Noise(3));
            let mut engine = open_target(g, &w, 16, &settings);
            let mut store = Store::default();
            for rows in &launches {
                assert_eq!(engine.step(rows).unwrap(), step(g, &w, &mut store, rows));
            }
        }
    }
}

/// Large `INDEX_KEEP` MLA uses `radix_top_k` and matches interpreter.
#[test]
fn mla_radix_index() {
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.max_seqs = 1;
    cfg.stack.max_context = 128;
    cfg.index.as_mut().unwrap().keep = 64;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let w = weights(&g, &mut Noise(13));
    let prefill: Vec<Row> = (0..72u32).map(|p| row(0, (p * 7 + 3) % 64, p)).collect();
    let decode = [row(0, 11, 72)];
    for settings in runs() {
        let mut engine = open_target(&g, &w, 128, &settings);
        let mut store = Store::default();
        for rows in [&prefill[..], &decode[..]] {
            assert_eq!(engine.step(rows).unwrap(), step(&g, &w, &mut store, rows));
        }
    }
}

/// Three ids after each prompt, and the steps they took.
fn asked(batch: &mut Batch, prompts: &[&[u32]]) -> (Vec<Vec<u32>>, usize) {
    let mut taken = Vec::new();
    for prompt in prompts {
        let (events, receiver) = std::sync::mpsc::channel();
        batch.submit(req(prompt, 3), Box::new(move |e| events.send(e).is_ok()));
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

/// MLA with bf16/fp8 KV shrinks buffers and matches f32.
#[test]
fn mla_narrow_kept() {
    use tessera_testkit::cosine_f32;
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.taps = vec![1];
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let w = weights(&g, &mut Noise(53));
    let launches = [
        vec![row(1, 5, 0), row(1, 7, 1), row(0, 9, 0), row(0, 3, 1)],
        vec![row(0, 4, 2), row(1, 3, 2)],
        vec![row(1, 8, 3), row(0, 2, 3)],
    ];
    let sizes = [(Seqs, 4), (Rows, 4), (Context, 16)];
    let kept_bytes = |plan: &tessera_lower::Plan| -> usize {
        plan.kept
            .iter()
            .map(|(_, s)| plan.buffers[s.buffer as usize])
            .sum()
    };
    let f32_plan = lower(&g, 0, &sizes, &Device::default()).unwrap();
    let base_bytes = kept_bytes(&f32_plan);
    let mut store = Store::default();
    let want: Vec<Vec<f32>> = launches
        .iter()
        .map(|rows| {
            let out = run_kept(&g, &w, &mut store, 0, &feeds(rows)).unwrap();
            out[1].floats().unwrap().to_vec()
        })
        .collect();
    let layouts: [&'static dyn kept::Layout; 2] = [kept::DENSE, &kept::PAGED16];
    for (form, factor, min_cos) in [(BF16_FORM, 2usize, 0.999f32), (FP8_FORM, 4, 0.95)] {
        let pins = Pins::default().kept(form);
        for &layout in &layouts {
            let mut dev = Device::default().pinned(pins.clone());
            dev.kept = layout;
            let plan = lower(&g, 0, &sizes, &dev).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert!(names.contains(&"listed_attend"), "{names:?}");
            assert!(!names.contains(&"widen"), "{names:?}");
            assert_eq!(kept_bytes(&plan) * factor, base_bytes);
            let recipe = Recipe {
                pins: pins.clone(),
                kept: layout,
                ..Recipe::default()
            };
            for settings in runs() {
                let mut engine = open_engine(&g, &w, 16, &recipe, &settings);
                for (step_idx, (rows, ref_logits)) in launches.iter().zip(&want).enumerate() {
                    let (logits, _) = engine.step_logits(rows).unwrap();
                    let cos = cosine_f32(&logits, ref_logits);
                    assert!(
                        cos > min_cos,
                        "{} {layout:?} step {step_idx} on {:?}: cos {cos}",
                        form.name(),
                        settings.backend
                    );
                }
            }
        }
    }
}

/// `Engine::lowering_report` lowers and sets the row label.
#[test]
fn engine_lowering_report() {
    let g = stepper::<Norm, Swiglu>(&Config::toy(1, 16));
    let settings = Settings {
        backend: tessera_engine::Pick::Cpu,
        ..Settings::default()
    };
    let mut engine = Engine::open(g, 16, &Recipe::default(), &settings).unwrap();
    let r = engine.lowering_report(1).unwrap();
    assert_eq!(r.label, "rows=1");
    assert!(r.total_ops > 0);
}

#[test]
fn shared_pool_preempts() {
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.max_context = 32;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let w = weights(&g, &mut Noise(11));
    let prompts: [&[u32]; 4] = [
        &[3, 9, 4, 11, 2, 7, 5, 1, 8, 6],
        &[5, 2, 8, 1, 9, 3, 4, 6, 7, 10],
        &[12, 1, 6, 7, 10, 2, 3, 8, 5, 4],
        &[2, 14, 1, 5, 9, 4, 7, 3, 6, 8],
    ];
    let settings = Settings::default();
    let want = solo_wants(&g, &w, 32, &settings, &prompts);
    let recipe = Recipe {
        kept: &kept::PAGED16,
        kept_rows: Some(32),
        ..Recipe::default()
    };
    let mut shared = open_engine(&g, &w, 32, &recipe, &settings);
    let mut batch = Batch::new(&mut shared, Box::new(Fair));
    assert_eq!(asked(&mut batch, &prompts).0, want);
    assert_eq!(batch.totals().unwrap().pool.size, 32);
}
