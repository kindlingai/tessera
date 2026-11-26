//! Native MTP speculative decoding tests (single-rank and TP).

mod common;

use common::{
    check_sharded_head, check_spec, collect_ids, dim_max, gen_events, gen_ids, host_settings,
    kept_count, loopback_links, open_mtp, open_target, open_tp_mtp, plan_gathers, req, spec_events,
    zero_last,
};
use tessera_engine::{
    Allowed, BatchMtp, Event, Fault, Guide, MtpSpeculate, Penalties, Prefix, Program, Reason,
    Request, Scoped, SpecStats, Stacked, Steps, Term,
};
use tessera_graph::DimName::{Hidden, Vocab};
use tessera_graph::{Graph, Parts, PathSeg, Role};
use tessera_interp::Tensor;
use tessera_lower::Wants;
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
    let prompt = [3u32, 9, 4, 11];
    for settings in runs() {
        let mut baseline = open_target(t_graph, t_weights, 24, &settings);
        let mut target = open_target(t_graph, t_weights, 24, &settings);
        let mut drafter = open_mtp(m_graph, m_weights, 24, &settings);
        let mut spec = MtpSpeculate::new(&mut target, &mut drafter, depth).unwrap();
        last = check_spec(
            &mut baseline,
            |r| spec_events(|e| spec.generate(r, e)),
            &prompt,
            [10, 4, 30],
        );
        let mut chunked = spec.with_prefill_chunk(2);
        let (mg, _) = spec_events(|e| chunked.generate(req(&prompt, 8), e));
        assert_eq!(collect_ids(&mg), gen_ids(&mut baseline, req(&prompt, 8)));
    }
    last
}

fn run_tp_pair(
    t_graph: &Graph,
    t_seed: u64,
    m_graph: &Graph,
    m_seed: u64,
    depth: usize,
    moe_tp: bool,
) {
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f) = (t_graph.clone(), m_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_mtp(
            (&tg_f, t_seed),
            (&mg_f, m_seed),
            24,
            moe_tp,
            &s_f,
            (Box::new(s1), c1),
        );
        MtpSpeculate::new(&mut te, &mut dr, depth)
            .unwrap()
            .follow()
            .unwrap();
    });
    let (mut target, mut drafter) = open_tp_mtp(
        (t_graph, t_seed),
        (m_graph, m_seed),
        24,
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
    let mut baseline = open_target(
        t_graph,
        &weights(t_graph, &mut Noise(t_seed)),
        24,
        &settings,
    );
    let mut spec = MtpSpeculate::new(&mut target, &mut drafter, depth).unwrap();
    for prompt in [&[2u32, 6, 1][..], &[4, 1, 5, 3][..]] {
        let (got, stats) = spec_events(|e| spec.generate(req(prompt, 8), e));
        assert_eq!(collect_ids(&got), gen_ids(&mut baseline, req(prompt, 8)));
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
        let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, indexed));
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
    zero_last(&mut t_weights);
    zero_last(&mut m_weights);
    let stats = run_single(&t_graph, &t_weights, &m_graph, &m_weights, 3);
    assert!(stats.accepted > 0 && stats.steps < stats.emitted);
}

fn verify_local_argmax(g: &Graph, seed: u64, moe_tp: bool) {
    let parts = Parts::even(2);
    let sharded = mtp::shard(g, &parts, &|_, _, _, _| true, moe_tp)
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let (vocab, hidden) = (dim_max(g, Vocab), dim_max(g, Hidden));
    let w = weights(g, &mut Noise(seed));
    for rows in [1usize, 3] {
        let feeds = vec![
            Tensor::u32(vec![rows], (0..rows as u32).map(|i| i + 1).collect()),
            Tensor::u32(vec![rows], (0..rows as u32).collect()),
            Tensor::u32(vec![rows], vec![0; rows]),
            Tensor::f32(vec![rows, hidden], vec![0.25; rows * hidden]),
            Tensor::f32(vec![rows, 1], vec![1.0; rows]),
        ];
        check_sharded_head(g, &sharded, &w, &feeds);
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
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let t_weights = weights(&t_graph, &mut Noise(83));
    let m_weights = weights(&m_graph, &mut Noise(89));
    let prompts: &[(&[u32], usize)] = &[
        (&[3, 9, 4, 11, 2], 7),
        (&[7, 1, 5], 6),
        (&[2, 8, 12, 6], 5),
        (&[5, 4], 6),
    ];
    for settings in runs() {
        let mut baseline = open_target(&t_graph, &t_weights, 24, &settings);
        let expected: Vec<Vec<u32>> = prompts
            .iter()
            .map(|&(p, max)| gen_ids(&mut baseline, req(p, max)))
            .collect();
        let mut target = open_target(&t_graph, &t_weights, 24, &settings);
        let mut drafter = open_mtp(&m_graph, &m_weights, 24, &settings);
        let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
            .unwrap()
            .with_prefill_chunk(3);
        let mut rxs = Vec::new();
        for &(p, max) in prompts {
            let (tx, rx) = std::sync::mpsc::channel();
            batch.submit(req(p, max), Box::new(move |e| tx.send(e).is_ok()));
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
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f) = (t_graph.clone(), m_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_mtp(
            (&tg_f, 97),
            (&mg_f, 101),
            24,
            true,
            &s_f,
            (Box::new(s1), c1),
        );
        BatchMtp::new(&mut te, &mut dr, 2)
            .unwrap()
            .with_prefill_chunk(3)
            .follow()
            .unwrap();
    });
    let (mut target, mut drafter) = open_tp_mtp(
        (&t_graph, 97),
        (&m_graph, 101),
        24,
        true,
        &settings,
        (Box::new(s0), c0),
    );
    let mut baseline = open_target(&t_graph, &weights(&t_graph, &mut Noise(97)), 24, &settings);
    let prompts: &[(&[u32], usize, bool)] = &[
        (&[2, 6, 1, 9], 7, false),
        (&[4, 1, 5, 3], 7, true),
        (&[8, 3, 7], 6, false),
    ];
    let expected: Vec<Vec<u32>> = prompts
        .iter()
        .map(|&(p, max, _)| gen_ids(&mut baseline, req(p, max)))
        .collect();
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(3);
    let mut rxs = Vec::new();
    for &(p, max, cancel) in prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut seen = 0usize;
        batch.submit(
            req(p, max),
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
    for ((rx, want), &(_, _, cancel)) in rxs.into_iter().zip(&expected).zip(prompts) {
        let evs: Vec<_> = rx.try_iter().collect();
        let ids = collect_ids(&evs);
        if cancel {
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
    let settings = host_settings();
    let mut t_cfg = mla_cfg(false, 16);
    t_cfg.stack.max_seqs = 2;
    t_cfg.stack.max_context = 32;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, false));
    let t_weights = weights(&t_graph, &mut Noise(131));
    let m_weights = weights(&m_graph, &mut Noise(137));
    let mut baseline = open_target(&t_graph, &t_weights, 32, &settings);
    let short = [2u32, 5];
    let long = [1u32, 3, 4, 6, 7, 8, 9, 10, 11];
    let want_short = gen_ids(&mut baseline, req(&short, 10));
    let want_long = gen_ids(&mut baseline, req(&long, 5));
    let mut target = open_target(&t_graph, &t_weights, 32, &settings);
    let mut drafter = open_mtp(&m_graph, &m_weights, 32, &settings);
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(3);
    batch.reuse();
    let (tx0, rx0) = std::sync::mpsc::channel();
    batch.submit(req(&short, 10), Box::new(move |e| tx0.send(e).is_ok()));
    assert!(batch.step().unwrap());
    let before: Vec<_> = rx0.try_iter().collect();
    let (tx1, rx1) = std::sync::mpsc::channel();
    batch.submit(req(&long, 5), Box::new(move |e| tx1.send(e).is_ok()));
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
    assert_eq!(collect_ids(&all0), want_short);
    assert_eq!(collect_ids(&rx1.try_iter().collect::<Vec<_>>()), want_long);

    // Follow-up prompts reuse full and partial prefixes identically to baseline.
    for (prompt, min_kept) in [
        (&[1u32, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13][..], 8usize),
        (&[1u32, 3, 4, 6, 14, 15][..], 3usize),
    ] {
        let want = gen_ids(&mut baseline, req(prompt, 5));
        let (tx, rx) = std::sync::mpsc::channel();
        batch.submit(req(prompt, 5), Box::new(move |e| tx.send(e).is_ok()));
        while batch.step().unwrap() {}
        let got: Vec<_> = rx.try_iter().collect();
        assert!(kept_count(&got) >= min_kept, "got={got:?}");
        assert_eq!(collect_ids(&got), want);
    }
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
    let prompt = [3u32, 9, 4, 11];
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
        let mut m_weights = weights(&m_graph, &mut Noise(29));
        if zeroed {
            zero_last(&mut t_weights);
            zero_last(&mut m_weights);
        }
        for settings in runs() {
            let mut baseline = open_target(&t_graph, &t_weights, 24, &settings);
            let mut target = open_target(&t_graph, &t_weights, 24, &settings);
            let mut drafter = open_mtp(&m_graph, &m_weights, 24, &settings);
            let mut spec = MtpSpeculate::new(&mut target, &mut drafter, 3).unwrap();
            let mut unguided = Vec::new();
            for guide in guides {
                let make_req = || Request {
                    guide: guide(&prompt),
                    ..req(&prompt, 12)
                };
                let want = gen_events(&mut baseline, make_req());
                let (got, _) = spec_events(|e| spec.generate(make_req(), e));
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
