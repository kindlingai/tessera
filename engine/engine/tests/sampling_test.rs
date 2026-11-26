//! Sampling, guides and logit edits.

mod common;

use common::{gen_ids, host_settings, open_target, req, zero_last};
use tessera_engine::{
    Allowed, DEPTH, Edit, Engine, Fault, Guide, Move, Penalties, Prefix, Program, Request, Runs,
    Sampling, Scoped, Stacked, Steps, Tally, Term,
};
use tessera_lower::Settings;
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

fn request(sampling: Sampling, guide: Option<Box<dyn Guide>>) -> Request {
    Request {
        sampling,
        guide,
        ..req(&[3, 9, 4], 6)
    }
}

fn engine() -> Engine {
    engine_on(Steps::Plain, &Settings::default())
}

fn engine_on(steps: Steps, settings: &Settings) -> Engine {
    let mut c = Config::toy(2, 16);
    c.stack.steps = steps;
    let g = stepper::<Norm, Swiglu>(&c);
    let w = weights(&g, &mut Noise(41));
    open_target(&g, &w, 16, settings)
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
    let a = gen_ids(&mut engine, request(sampling(42), None));
    let b = gen_ids(&mut engine, request(sampling(42), None));
    assert_eq!(a.len(), 6);
    assert_eq!(a, b);
}

#[test]
fn edits_steer_greedy() {
    let mut engine = engine();
    let greedy = Sampling::default();
    let plain = gen_ids(&mut engine, request(greedy, None));
    let none = Some(Box::new(Edits(Vec::new())) as Box<dyn Guide>);
    assert_eq!(gen_ids(&mut engine, request(greedy, none)), plain);
    let ban = Edit {
        id: plain[0],
        add: -1e9,
        rep: 1.0,
    };
    let banned = gen_ids(
        &mut engine,
        request(greedy, Some(Box::new(Edits(vec![ban])))),
    );
    assert!(!banned.contains(&plain[0]), "{banned:?}");
    let once = Penalties {
        presence: 1e9,
        ..Penalties::default()
    };
    let mut fresh = gen_ids(&mut engine, request(greedy, once.guide(&[])));
    fresh.sort_unstable();
    fresh.dedup();
    assert_eq!(fresh.len(), 6);
}

/// Allows a fixed set and edits fixed ids.
struct Fixed(Allowed, Vec<Edit>);

impl Guide for Fixed {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        Ok(self.0.clone())
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        match self.0.allows(id) {
            true => Ok(()),
            false => Err("refused".into()),
        }
    }

    fn edits(&mut self) -> Vec<Edit> {
        self.1.clone()
    }
}

fn add(id: u32, add: f32, rep: f32) -> Edit {
    Edit { id, add, rep }
}

fn stack(members: Vec<Box<dyn Guide>>) -> Box<dyn Guide> {
    Stacked::of(members).expect("a guide")
}

#[test]
fn stack_meets_and_combines() {
    let bits = Allowed::Bits(vec![0b1111_0000, u32::MAX]);
    let few = Allowed::Few(vec![2, 5, 7, 40]);
    let mut g = stack(vec![
        Box::new(Fixed(bits, vec![add(5, 1.0, 2.0)])),
        Box::new(Fixed(few, vec![add(5, -3.0, 1.5), add(7, 0.5, 1.0)])),
        Box::new(Fixed(Allowed::Except(vec![7]), Vec::new())),
    ]);
    assert_eq!(g.allowed().unwrap(), Allowed::Few(vec![5, 40]));
    assert_eq!(g.edits(), [add(5, -2.0, 3.0), add(7, 0.5, 1.0)]);
    assert!(g.accept(7).is_err());
    let mut none = stack(vec![
        Box::new(Fixed(Allowed::One(3), Vec::new())),
        Box::new(Fixed(Allowed::Except(vec![3]), Vec::new())),
    ]);
    assert!(none.allowed().is_err());
}

#[test]
fn stack_shares_forced() {
    let mut g = stack(vec![
        Box::new(Prefix(vec![4, 6])),
        Box::new(Fixed(Allowed::Few(vec![4, 6, 9]), Vec::new())),
    ]);
    assert_eq!(g.forced(), [4, 6]);
    assert!(g.allowed().is_ok());
    let mut bad = stack(vec![
        Box::new(Prefix(vec![4])),
        Box::new(Fixed(Allowed::Few(vec![6, 9]), Vec::new())),
    ]);
    assert_eq!(bad.forced(), [4]);
    assert!(!bad.done() && bad.allowed().is_err());
    let mut split = stack(vec![Box::new(Prefix(vec![4])), Box::new(Prefix(vec![5]))]);
    split.forced();
    assert!(split.allowed().is_err());
}

fn go(from: u8, ids: &[u32], to: u8) -> Move {
    Move {
        from,
        ids: ids.to_vec(),
        to,
    }
}

fn moves() -> Vec<Move> {
    vec![go(0, &[9], 1), go(1, &[8], 0)]
}

#[test]
fn scopes_move() {
    let program = Program {
        moves: moves(),
        terms: Vec::new(),
    };
    let mut tally = Tally::new(&program, &[9], 1);
    let mut seen = Vec::new();
    for id in [9, 8, 5, 9] {
        tally.step(&program, id);
        seen.push((tally.scope, tally.since));
    }
    assert_eq!(seen, [(1, 1), (0, 0), (0, 1), (1, 0)]);
}

#[test]
fn sequences_move() {
    let moves = vec![
        go(0, &[7, 9], 1),
        go(0, &[5, 7, 9], 2),
        go(1, &[6, 6], 0),
        go(2, &[6], 0),
    ];
    let program = Program {
        moves: moves.clone(),
        terms: Vec::new(),
    };
    let mut tally = Tally::new(&program, &[], 0);
    let mut seen = Vec::new();
    for id in [7, 9, 6, 6, 5, 7, 9, 6] {
        tally.step(&program, id);
        seen.push(tally.scope);
    }
    assert_eq!(seen, [0, 1, 1, 0, 0, 0, 2, 0]);
    let terms = vec![Term::Count {
        presence: vec![1.0, 2.0, 3.0],
        frequency: Vec::new(),
    }];
    let mut g = Scoped::new(Program { moves, terms }, &[], 0);
    g.accept(5).unwrap();
    g.accept(7).unwrap();
    // A fork taken inside a marker finishes it.
    let mut fork = g.fork().unwrap();
    for (id, presence) in [(9, -3.0), (6, -1.0)] {
        g.accept(id).unwrap();
        fork.accept(id).unwrap();
        assert_eq!(fork.edits(), g.edits());
        assert_eq!(g.edits()[0].add, presence);
    }
}

#[test]
fn counts_follow_scope() {
    let terms = vec![
        Term::Count {
            presence: vec![1.0, 0.0],
            frequency: vec![0.5, 2.0],
        },
        Term::Repeat {
            repetition: vec![1.25, 1.0],
        },
    ];
    let mut g = Scoped::new(
        Program {
            moves: moves(),
            terms,
        },
        &[3],
        0,
    );
    g.accept(4).unwrap();
    g.accept(4).unwrap();
    assert_eq!(g.edits(), [add(3, 0.0, 1.25), add(4, -2.0, 1.25)]);
    g.accept(9).unwrap();
    assert_eq!(g.edits(), [add(4, -4.0, 1.0), add(9, -2.0, 1.0)]);
}

#[test]
fn ramp_is_bounded() {
    let ramp = Term::Ramp {
        id: 2,
        scope: 1,
        start: 2,
        slope: 0.5,
        max: 1.0,
    };
    let program = Program {
        moves: moves(),
        terms: vec![ramp],
    };
    let mut tally = Tally::new(&program, &[], 1);
    let mut adds = Vec::new();
    for id in [5, 5, 5, 5, 5, 8] {
        tally.step(&program, id);
        adds.push(program.edits(&tally).first().map(|e| e.add));
    }
    // The pick of token `start + 1` gets the first raise.
    let want = [None, Some(0.5), Some(1.0), Some(1.0), Some(1.0), None];
    assert_eq!(adds, want);
}

#[test]
fn runs_find_copies() {
    let runs = Runs::new(&[1, 2, 3, 9, 0, 1, 2, 3, 7, 0, 1, 2, 3]);
    assert_eq!(runs.next(1), [(7, 4), (9, 3)]);
    // One id many times holds `DEPTH` runs, the newest longest.
    let mut runs = Runs::default();
    (0..3000).for_each(|_| runs.push(4));
    assert_eq!((runs.ends().len(), runs.next(1)), (DEPTH, vec![(4, 2999)]));
}

#[test]
fn runs_fork() {
    let mut noise = Noise(5);
    let prompt: Vec<u32> = (0..2000)
        .map(|_| (noise.draw() * 4.0 + 4.0) as u32)
        .collect();
    let mut runs = Runs::new(&prompt);
    let mut fork = runs.clone();
    let more = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
    for &id in &more {
        fork.push(id);
    }
    for &id in &more {
        runs.push(id);
    }
    let whole = Runs::new(&[&prompt[..], &more].concat());
    assert_eq!((runs.ends(), runs.next(1)), (fork.ends(), fork.next(1)));
    assert_eq!(runs.next(1), whole.next(1));
}

#[test]
fn sequence_edits() {
    let program = Program {
        moves: Vec::new(),
        terms: vec![Term::Sequence {
            mult: vec![1.0],
            base: vec![2.0],
            min: vec![3],
            cap: vec![5.0],
        }],
    };
    let mut g = Scoped::new(program, &[8, 1, 2, 3, 4, 1, 2, 3], 0);
    let mut adds = vec![g.edits()];
    for id in [4, 1, 2] {
        g.accept(id).unwrap();
        adds.push(g.edits());
    }
    let want = [
        add(4, -1.0, 1.0),
        add(1, -2.0, 1.0),
        add(2, -4.0, 1.0),
        add(3, -5.0, 1.0),
    ];
    assert_eq!(adds, want.map(|e| vec![e]));
}

#[test]
fn device_edits_match_host() {
    let greedy = Sampling::default();
    for s in runs() {
        let (mut host, mut device) = (engine_on(Steps::Plain, &s), engine_on(Steps::Masked, &s));
        let plain = gen_ids(&mut host, request(greedy, None));
        let mut ids = plain.clone();
        ids.sort_unstable();
        ids.dedup();
        // Unsorted, with each id twice.
        let each = ids
            .iter()
            .rev()
            .flat_map(|&id| [add(id, -0.5, 1.5), add(id, 0.25, 1.2)]);
        let edits: Vec<Edit> = each.collect();
        let guide = || Some(Box::new(Edits(edits.clone())) as Box<dyn Guide>);
        let want = gen_ids(&mut host, request(greedy, guide()));
        assert_ne!(want, plain);
        assert_eq!(gen_ids(&mut device, request(greedy, guide())), want);
    }
}

/// A prefill report covers one step, within the context.
#[test]
fn prefill_report_fits_context() {
    let mut e = engine();
    let most = e.chunk(None);
    assert_eq!(e.chunk(Some(2)), 2);
    assert_eq!(e.chunk(Some(usize::MAX)), most);
    let r = e.prefill_report(usize::MAX).unwrap();
    assert!(r.label.starts_with("prefill rows="), "{}", r.label);
    assert!(r.total_ops > 0);
}

#[test]
fn min_tokens_holds() {
    let settings = host_settings();
    for steps in [Steps::Plain, Steps::Masked] {
        let mut cfg = Config::toy(1, 16);
        cfg.stack.steps = steps;
        let g = stepper::<Norm, Swiglu>(&cfg);
        let mut w = weights(&g, &mut Noise(19));
        // Zero output head so greedy argmax picks stop token 0.
        zero_last(&mut w);
        let mut engine = open_target(&g, &w, 16, &settings);
        let mut run = |min_tokens| {
            gen_ids(
                &mut engine,
                Request {
                    min_tokens,
                    stop: vec![0],
                    ..req(&[2, 3], 4)
                },
            )
        };
        assert!(run(0).is_empty(), "steps={steps:?}");
        let held = run(3);
        assert_eq!(held.len(), 3, "steps={steps:?}");
        assert!(held.iter().all(|&id| id != 0), "steps={steps:?}");
    }
}
