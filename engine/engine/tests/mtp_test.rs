//! Native MTP speculative decoding tests (single-rank and TP).

mod common;

use common::{collect_ids, float_weights, loopback_links, open_ranked_target, open_target};
use tessera_engine::{
    Allowed, BatchMtp, Drafter, Event, Fault, Guide, MtpDrafter, MtpSpeculate, Penalties, Place,
    Prefix, Program, Reason, Recipe, Request, Scoped, SpecStats, Stacked, Steps, Term,
};
use tessera_graph::DimName::{Context, Hidden, Rows, Segments, Seqs, Vocab};
use tessera_graph::{Graph, Parts, PathSeg, Role};
use tessera_interp::{Tensor, run, run_ranks};
use tessera_lower::{Cache, Cluster, Device, Pick, Settings, Wants, Weight};
use tessera_models::{mla, mtp};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

fn mla_cfg(indexed: bool, ffn: u64) -> mla::Config {
    let mut cfg = mla::Config::deepseek_small();
    cfg.stack.max_seqs = 1;
    cfg.stack.max_context = 24;
    cfg.stack.ffn = ffn;
    cfg.index = indexed.then_some(mla::Index {
        heads: 2,
        dim: 16,
        keep: 8,
        every: 1,
    });
    cfg.stack.taps = vec![mtp::tap_layer(&cfg)];
    cfg
}

fn run_single(
    t_graph: &Graph,
    t_weights: &[Tensor],
    m_graph: &Graph,
    m_weights: &[Tensor],
    depth: usize,
) -> SpecStats {
    let mut last = SpecStats::default();
    let prompt = vec![3u32, 9, 4, 11];
    let m_stored: Vec<Weight> = m_weights.iter().cloned().map(Weight::Floats).collect();
    for settings in runs() {
        let mut baseline = open_target(t_graph, t_weights, 24, &settings);
        let mut target = open_target(t_graph, t_weights, 24, &settings);
        let mut drafter =
            MtpDrafter::open(m_graph.clone(), 24, &Recipe::default(), &settings).unwrap();
        drafter.load(&m_stored).unwrap();
        let mut spec = MtpSpeculate::new(&mut target, &mut drafter, depth).unwrap();
        let req = |max_tokens, stop| Request {
            tokens: prompt.clone(),
            max_tokens,
            stop,
            ..Request::default()
        };
        let (mut want, mut got) = (Vec::new(), Vec::new());
        baseline
            .generate(req(10, Vec::new()), &mut |e| want.push(e))
            .unwrap();
        last = spec
            .generate(req(10, Vec::new()), &mut |e| {
                got.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got), collect_ids(&want));
        assert_eq!(got.last(), Some(&Event::Done(Reason::Length)));
        assert!(last.steps > 0);
        let stop_id = collect_ids(&want)[4];
        let (mut sw, mut sg) = (Vec::new(), Vec::new());
        baseline
            .generate(req(10, vec![stop_id]), &mut |e| sw.push(e))
            .unwrap();
        spec.generate(req(10, vec![stop_id]), &mut |e| {
            sg.push(e);
            true
        })
        .unwrap();
        assert_eq!(collect_ids(&sg), collect_ids(&sw));
        assert_eq!(sg.last(), Some(&Event::Done(Reason::Stop)));
        let (mut cw, mut cg) = (Vec::new(), Vec::new());
        baseline
            .generate(req(30, Vec::new()), &mut |e| cw.push(e))
            .unwrap();
        spec.generate(req(30, Vec::new()), &mut |e| {
            cg.push(e);
            true
        })
        .unwrap();
        assert_eq!(collect_ids(&cg), collect_ids(&cw));
        assert_eq!(cg.last(), Some(&Event::Done(Reason::Context)));
        let mut chunked = spec.with_prefill_chunk(2);
        let (mut mw, mut mg) = (Vec::new(), Vec::new());
        baseline
            .generate(req(8, Vec::new()), &mut |e| mw.push(e))
            .unwrap();
        chunked
            .generate(req(8, Vec::new()), &mut |e| {
                mg.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&mg), collect_ids(&mw));
    }
    last
}

fn open_mtp_rank<T: tessera_engine::comm::Transport + 'static>(
    (t_graph, t_seed): (Graph, u64),
    (m_graph, m_seed): (Graph, u64),
    moe_tp: bool,
    settings: &Settings,
    (sum, ctl): (Box<dyn tessera_engine::comm::Transport>, T),
) -> (tessera_engine::Engine, MtpDrafter) {
    let [ts, ms] = Drafter::share_transport(sum);
    let place = || Place::new(Cluster::two("a", "b"), Parts::even(2)).with_moe_tp(moe_tp);
    let (tw, mw) = (
        float_weights(&t_graph, t_seed),
        float_weights(&m_graph, m_seed),
    );
    let target = open_ranked_target(t_graph, 24, settings, place(), ts, ctl, &tw);
    let mut drafter =
        MtpDrafter::open_ranked(m_graph, 24, (&Recipe::default(), settings), place(), ms).unwrap();
    drafter.load(&mw).unwrap();
    (target, drafter)
}

fn run_tp_pair(
    t_graph: &Graph,
    t_seed: u64,
    m_graph: &Graph,
    m_seed: u64,
    depth: usize,
    moe_tp: bool,
) {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f) = (t_graph.clone(), m_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_mtp_rank(
            (tg_f, t_seed),
            (mg_f, m_seed),
            moe_tp,
            &s_f,
            (Box::new(s1), c1),
        );
        MtpSpeculate::new(&mut te, &mut dr, depth)
            .unwrap()
            .follow()
            .unwrap();
    });
    let t_weights = weights(t_graph, &mut Noise(t_seed));
    let (mut target, mut drafter) = open_mtp_rank(
        (t_graph.clone(), t_seed),
        (m_graph.clone(), m_seed),
        moe_tp,
        &settings,
        (Box::new(s0), c0),
    );
    assert!(drafter.has_kernel("all_reduce"), "missing all_reduce");
    assert!(drafter.has_kernel("all_gather"), "missing all_gather");
    for name in ["eh_proj", "q_b", "kv_b", "o", "head"] {
        let at = m_graph
            .weights
            .iter()
            .position(|w| matches!(w.path.last(), Some(PathSeg::Field(f)) if f == name))
            .unwrap() as u32;
        assert!(drafter.share(at).is_some(), "{name} should shard");
    }
    let mut baseline = open_target(t_graph, &t_weights, 24, &settings);
    let mut spec = MtpSpeculate::new(&mut target, &mut drafter, depth).unwrap();
    for prompt in [vec![2u32, 6, 1], vec![4u32, 1, 5, 3]] {
        let req = || Request {
            tokens: prompt.clone(),
            max_tokens: 8,
            ..Request::default()
        };
        let (mut want, mut got) = (Vec::new(), Vec::new());
        baseline.generate(req(), &mut |e| want.push(e)).unwrap();
        let stats = spec
            .generate(req(), &mut |e| {
                got.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got), collect_ids(&want));
        assert!(stats.steps > 0);
    }
    target.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn mtp_matches_greedy() {
    for indexed in [false, true] {
        let t_cfg = mla_cfg(indexed, 16);
        let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
        let m_cfg = mtp::config(&t_cfg, indexed);
        let m_graph = mtp::stepper::<mla::Moe>(&m_cfg);
        let t_weights = weights(&t_graph, &mut Noise(11 + indexed as u64));
        let m_weights = weights(&m_graph, &mut Noise(29 + indexed as u64));
        for depth in [1, 2, 3] {
            run_single(&t_graph, &t_weights, &m_graph, &m_weights, depth);
        }
    }
    let t_cfg = mla_cfg(true, 16);
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let mut t_weights = weights(&t_graph, &mut Noise(41));
    let mut m_weights = weights(&m_graph, &mut Noise(43));
    let zero = |t: &mut Tensor| *t = Tensor::f32(t.shape.clone(), vec![0.0; t.len()]);
    zero(t_weights.last_mut().unwrap());
    zero(m_weights.last_mut().unwrap());
    let stats = run_single(&t_graph, &t_weights, &m_graph, &m_weights, 3);
    assert!(stats.accepted > 0 && stats.steps < stats.emitted);
}

fn plan_gathers(g: &Graph, parts: &Parts, rows: u64, wants: Wants) -> Vec<usize> {
    let mut dev = Device::guess(&Settings {
        backend: Pick::Host,
        ..Settings::default()
    });
    dev = dev
        .zoned(Cluster::two("a", "b"))
        .ranked(vec![0, 1])
        .uneven(parts.0.clone())
        .as_rank(0);
    let mut cache = Cache::default();
    let dims = [(Seqs, 1), (Rows, rows), (Context, rows), (Segments, 1)];
    let (plan, _) = cache
        .wanting(g, 0, wants, &dims, &[Context, Segments], &dev)
        .unwrap();
    let mut out = Vec::new();
    for step in &plan.steps {
        if step.kernel.name() == "all_gather" {
            out.push(step.write.len);
        }
    }
    out
}

fn verify_local_argmax(g: &Graph, seed: u64, moe_tp: bool) {
    let parts = Parts::even(2);
    let sharded = mtp::shard(g, &parts, &|_, _, _, _| true, moe_tp)
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let dim = |name| g.dims.iter().find(|d| d.name == name).unwrap().size.max() as usize;
    let (vocab, hidden) = (dim(Vocab), dim(Hidden));
    let w = weights(g, &mut Noise(seed));
    let mut tied = w.clone();
    let head = g
        .weights
        .iter()
        .position(|wt| matches!(wt.path.last(), Some(PathSeg::Field(f)) if f == "head"))
        .unwrap();
    tied[head] = Tensor::f32(tied[head].shape.clone(), vec![0.0; tied[head].len()]);
    for rows in [1usize, 3] {
        let feeds = vec![
            Tensor::u32(vec![rows], (0..rows as u32).map(|i| i + 1).collect()),
            Tensor::u32(vec![rows], (0..rows as u32).collect()),
            Tensor::u32(vec![rows], vec![0; rows]),
            Tensor::f32(vec![rows, hidden], vec![0.25; rows * hidden]),
            Tensor::f32(vec![rows, 1], vec![1.0; rows]),
        ];
        let want = run(g, &w, 0, &feeds).unwrap();
        let got = run_ranks(&sharded, &w, 0, &feeds, 2).unwrap();
        assert_eq!(got[0].indices().unwrap(), want[0].indices().unwrap());
        let want_tie = run(g, &tied, 0, &feeds).unwrap();
        let got_tie = run_ranks(&sharded, &tied, 0, &feeds, 2).unwrap();
        assert_eq!(
            got_tie[0].indices().unwrap(),
            want_tie[0].indices().unwrap()
        );
    }
    for rows in [1u64, 4] {
        let expect = vec![rows as usize * hidden, 2 * rows as usize * 2];
        for wants in [
            Wants::role(&sharded, 0, Role::Tokens),
            Wants::roles(&sharded, 0, &[Role::Tokens, Role::Taps]),
        ] {
            let gathers = plan_gathers(&sharded, &parts, rows, wants);
            assert_eq!(gathers, expect, "rows={rows}");
            assert!(!gathers.contains(&(rows as usize * vocab)), "rows={rows}");
        }
        let logits_gathers = plan_gathers(
            &sharded,
            &parts,
            rows,
            Wants::role(&sharded, 0, Role::Logits),
        );
        assert!(logits_gathers.contains(&(rows as usize * vocab)));
    }
}

#[test]
fn mtp_tp_matches_baseline() {
    let mut t_cfg = mla_cfg(true, 512);
    t_cfg.stack.vocab = 128;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    verify_local_argmax(&m_graph, 67, false);
    verify_local_argmax(&m_graph, 73, true);
    run_tp_pair(&t_graph, 61, &m_graph, 67, 2, false);
    run_tp_pair(&t_graph, 71, &m_graph, 73, 3, true);
}

#[test]
fn batch_mtp_matches_target() {
    let mut t_cfg = mla_cfg(true, 16);
    t_cfg.stack.max_seqs = 3;
    t_cfg.stack.max_context = 24;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let t_weights = weights(&t_graph, &mut Noise(83));
    let m_weights = weights(&m_graph, &mut Noise(89));
    let m_stored: Vec<Weight> = m_weights.iter().cloned().map(Weight::Floats).collect();
    let prompts = [
        (vec![3u32, 9, 4, 11, 2], 7usize),
        (vec![7u32, 1, 5], 6),
        (vec![2u32, 8, 12, 6], 5),
        (vec![5u32, 4], 6),
    ];
    for settings in runs() {
        let mut baseline = open_target(&t_graph, &t_weights, 24, &settings);
        let mut expected = Vec::new();
        for (toks, max_tokens) in &prompts {
            let mut got = Vec::new();
            let req = Request {
                tokens: toks.clone(),
                max_tokens: *max_tokens,
                ..Request::default()
            };
            baseline.generate(req, &mut |e| got.push(e)).unwrap();
            expected.push(collect_ids(&got));
        }
        let mut target = open_target(&t_graph, &t_weights, 24, &settings);
        let mut drafter =
            MtpDrafter::open(m_graph.clone(), 24, &Recipe::default(), &settings).unwrap();
        drafter.load(&m_stored).unwrap();
        let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
            .unwrap()
            .with_prefill_chunk(3);
        let mut rxs = Vec::new();
        for (toks, max_tokens) in &prompts {
            let (tx, rx) = std::sync::mpsc::channel();
            let req = Request {
                tokens: toks.clone(),
                max_tokens: *max_tokens,
                ..Request::default()
            };
            batch.submit(req, Box::new(move |e| tx.send(e).is_ok()));
            rxs.push(rx);
        }
        while batch.step().unwrap() {}
        for (rx, want) in rxs.into_iter().zip(&expected) {
            let evs: Vec<_> = rx.try_iter().collect();
            assert_eq!(&collect_ids(&evs), want);
            assert_eq!(evs.last(), Some(&Event::Done(Reason::Length)));
        }
    }
}

#[test]
fn batch_mtp_ranked_matches() {
    let mut t_cfg = mla_cfg(true, 512);
    t_cfg.stack.vocab = 128;
    t_cfg.stack.max_seqs = 2;
    t_cfg.stack.max_context = 24;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f) = (t_graph.clone(), m_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) =
            open_mtp_rank((tg_f, 97), (mg_f, 101), true, &s_f, (Box::new(s1), c1));
        BatchMtp::new(&mut te, &mut dr, 2)
            .unwrap()
            .with_prefill_chunk(3)
            .follow()
            .unwrap();
    });
    let t_weights = weights(&t_graph, &mut Noise(97));
    let (mut target, mut drafter) = open_mtp_rank(
        (t_graph.clone(), 97),
        (m_graph.clone(), 101),
        true,
        &settings,
        (Box::new(s0), c0),
    );
    let mut baseline = open_target(&t_graph, &t_weights, 24, &settings);
    let prompts = [
        (vec![2u32, 6, 1, 9], 7usize, false),
        (vec![4u32, 1, 5, 3], 7, true),
        (vec![8u32, 3, 7], 6, false),
    ];
    let mut expected = Vec::new();
    for (toks, max_tokens, _) in &prompts {
        let mut got = Vec::new();
        let req = Request {
            tokens: toks.clone(),
            max_tokens: *max_tokens,
            ..Request::default()
        };
        baseline.generate(req, &mut |e| got.push(e)).unwrap();
        expected.push(collect_ids(&got));
    }
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(3);
    let mut rxs = Vec::new();
    for (toks, max_tokens, cancel_early) in &prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut seen = 0usize;
        let cancel = *cancel_early;
        let req = Request {
            tokens: toks.clone(),
            max_tokens: *max_tokens,
            ..Request::default()
        };
        batch.submit(
            req,
            Box::new(move |e| {
                if matches!(e, Event::Token(_)) {
                    seen += 1;
                }
                let _ = tx.send(e);
                !(cancel && seen >= 2)
            }),
        );
        rxs.push(rx);
    }
    while batch.step().unwrap() {}
    drop(batch);
    target.dismiss().unwrap();
    follower.join().unwrap();
    for ((rx, want), (_, _, cancel)) in rxs.into_iter().zip(&expected).zip(&prompts) {
        let evs: Vec<_> = rx.try_iter().collect();
        let ids = collect_ids(&evs);
        if *cancel {
            assert_eq!(&ids[..2], &want[..2]);
            assert_eq!(evs.last(), Some(&Event::Done(Reason::Stop)));
        } else {
            assert_eq!(&ids, want);
            assert_eq!(evs.last(), Some(&Event::Done(Reason::Length)));
        }
    }
}

/// Active decode steps interleave with a new prompt's prefill chunks.
#[test]
fn chunked_prefill_interleaves() {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut t_cfg = mla_cfg(false, 16);
    t_cfg.stack.max_seqs = 2;
    t_cfg.stack.max_context = 32;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, false));
    let t_weights = weights(&t_graph, &mut Noise(131));
    let m_weights = float_weights(&m_graph, 137);
    let mut baseline = open_target(&t_graph, &t_weights, 32, &settings);
    let short = vec![2u32, 5];
    let long = vec![1u32, 3, 4, 6, 7, 8, 9, 10, 11];
    let mut want_short = Vec::new();
    let mut want_long = Vec::new();
    baseline
        .generate(
            Request {
                tokens: short.clone(),
                max_tokens: 10,
                ..Request::default()
            },
            &mut |e| want_short.push(e),
        )
        .unwrap();
    baseline
        .generate(
            Request {
                tokens: long.clone(),
                max_tokens: 5,
                ..Request::default()
            },
            &mut |e| want_long.push(e),
        )
        .unwrap();
    let mut target = open_target(&t_graph, &t_weights, 32, &settings);
    let mut drafter = MtpDrafter::open(m_graph, 32, &Recipe::default(), &settings).unwrap();
    drafter.load(&m_weights).unwrap();
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(3);
    let (tx0, rx0) = std::sync::mpsc::channel();
    batch.submit(
        Request {
            tokens: short,
            max_tokens: 10,
            ..Request::default()
        },
        Box::new(move |e| tx0.send(e).is_ok()),
    );
    assert!(batch.step().unwrap());
    let before = rx0.try_iter().collect::<Vec<_>>();
    let (tx1, rx1) = std::sync::mpsc::channel();
    batch.submit(
        Request {
            tokens: long,
            max_tokens: 5,
            ..Request::default()
        },
        Box::new(move |e| tx1.send(e).is_ok()),
    );
    // One step runs one chunk plus active decode.
    assert!(batch.step().unwrap());
    let mid0: Vec<_> = rx0.try_iter().collect();
    let mid1: Vec<_> = rx1.try_iter().collect();
    assert!(!mid0.is_empty(), "active decode stalled during prefill");
    assert!(mid1.is_empty(), "long prompt finished in one chunk");
    while batch.step().unwrap() {}
    let mut all0 = before;
    all0.extend(mid0);
    all0.extend(rx0.try_iter());
    let all1: Vec<_> = rx1.try_iter().collect();
    assert_eq!(collect_ids(&all0), collect_ids(&want_short));
    assert_eq!(collect_ids(&all1), collect_ids(&want_long));
}

/// A guide for a prompt, or none.
type MakeGuide = fn(&[u32]) -> Option<Box<dyn Guide>>;

/// Bans 0 every third token and forces 1 every fifth.
#[derive(Clone)]
struct Thirds(usize);

impl Guide for Thirds {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        match self.0 % 3 {
            2 => Ok(Allowed::Except(vec![0])),
            _ => Ok(Allowed::Any),
        }
    }

    fn forced(&mut self) -> Vec<u32> {
        if self.0 % 5 != 4 {
            return Vec::new();
        }
        self.0 += 1;
        vec![1]
    }

    fn accept(&mut self, _: u32) -> Result<(), Fault> {
        self.0 += 1;
        Ok(())
    }

    fn fork(&self) -> Option<Box<dyn Guide>> {
        Some(Box::new(self.clone()))
    }
}

/// `Thirds` with no fork.
struct Unforked(Thirds);

impl Guide for Unforked {
    fn allowed(&mut self) -> Result<Allowed, Fault> {
        self.0.allowed()
    }

    fn forced(&mut self) -> Vec<u32> {
        self.0.forced()
    }

    fn accept(&mut self, id: u32) -> Result<(), Fault> {
        self.0.accept(id)
    }
}

#[test]
fn guided_mtp_matches() {
    let mut t_cfg = mla_cfg(false, 16);
    t_cfg.stack.steps = Steps::Masked;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, false));
    let prompt = vec![3u32, 9, 4, 11];
    let guides: [MakeGuide; 5] = [
        |_| None,
        |_| Some(Box::new(Thirds(0))),
        |p| penalties_guide(p),
        |p| {
            Stacked::of(vec![
                Box::new(Prefix(vec![5, 0])),
                penalties_guide(p).unwrap(),
                Box::new(Scoped::new(sequence(), p, 0)),
            ])
        },
        |_| Some(Box::new(Unforked(Thirds(0)))),
    ];
    // Zero heads draft 0 always, and the target holds most.
    for zeroed in [false, true] {
        let mut t_weights = weights(&t_graph, &mut Noise(11));
        let mut m_weights: Vec<Weight> = float_weights(&m_graph, 29);
        if zeroed {
            let t = t_weights.last_mut().unwrap();
            *t = Tensor::f32(t.shape.clone(), vec![0.0; t.len()]);
            let Some(Weight::Floats(m)) = m_weights.last_mut() else {
                unreachable!()
            };
            *m = Tensor::f32(m.shape.clone(), vec![0.0; m.len()]);
        }
        for settings in runs() {
            let mut baseline = open_target(&t_graph, &t_weights, 24, &settings);
            let mut target = open_target(&t_graph, &t_weights, 24, &settings);
            let mut drafter =
                MtpDrafter::open(m_graph.clone(), 24, &Recipe::default(), &settings).unwrap();
            drafter.load(&m_weights).unwrap();
            let mut spec = MtpSpeculate::new(&mut target, &mut drafter, 3).unwrap();
            let mut unguided = Vec::new();
            for guide in guides {
                let req = || Request {
                    tokens: prompt.clone(),
                    max_tokens: 12,
                    guide: guide(&prompt),
                    ..Request::default()
                };
                let (mut want, mut got) = (Vec::new(), Vec::new());
                baseline.generate(req(), &mut |e| want.push(e)).unwrap();
                spec.generate(req(), &mut |e| {
                    got.push(e);
                    true
                })
                .unwrap();
                assert_eq!(got, want);
                // With tied logits every guide changes the answer.
                assert!(!zeroed || unguided.is_empty() || want != unguided);
                if unguided.is_empty() {
                    unguided = want;
                }
            }
        }
    }
}

/// A sequence penalty from runs of one id.
fn sequence() -> Program {
    let terms = vec![Term::Sequence {
        mult: vec![2.0],
        base: vec![2.0],
        min: vec![1],
        cap: vec![8.0],
    }];
    Program {
        moves: Vec::new(),
        terms,
    }
}

fn penalties_guide(prompt: &[u32]) -> Option<Box<dyn Guide>> {
    let penalties = Penalties {
        presence: 4.0,
        ..Penalties::default()
    };
    penalties.guide(prompt)
}
