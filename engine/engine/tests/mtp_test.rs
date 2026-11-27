//! Native MTP speculative decoding tests (single-rank and TP).

mod common;

use common::{
    check_sharded_head, check_spec, collect_ids, dim_max, float_weights, gen_events, gen_ids,
    host_settings, kept_count, loopback_links, open_mtp, open_target, open_tp_mtp, plan_gathers,
    req, spec_events, two_place, zero_last,
};
use tessera_engine::{
    Allowed, BatchMtp, Drafter, Engine, Event, Fault, Guide, MtpDrafter, MtpSpeculate, Penalties,
    Prefix, Program, Reason, Recipe, Request, Scoped, SpecStats, Stacked, Steps, Term,
};
use tessera_graph::DimName::{Context, Hidden, Picks, Rows, Segments, Seqs, Vocab};
use tessera_graph::{DType, Graph, Parts, PathSeg, Role, shard_mode, trim};
use tessera_interp::Tensor;
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Cache, Cluster, Device, Limits, Pins, Platform, Wants, ZoneKind};
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
        rope_interleave: true,
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

fn greedy_answers(base: &mut Engine, prompts: &[(Vec<u32>, usize)]) -> Vec<Vec<u32>> {
    prompts
        .iter()
        .map(|(toks, max_tokens)| gen_ids(base, req(toks, *max_tokens)))
        .collect()
}

fn run_batch_seq(
    batch: &mut BatchMtp<'_>,
    prompts: &[(Vec<u32>, usize)],
    want: &[Vec<u32>],
) -> bool {
    let mut kept_hit = false;
    for ((toks, max_tokens), expected) in prompts.iter().zip(want) {
        let (tx, rx) = std::sync::mpsc::channel();
        batch.submit(
            req(toks, *max_tokens),
            Box::new(move |e| tx.send(e).is_ok()),
        );
        while batch.step().unwrap() {}
        let evs: Vec<_> = rx.try_iter().collect();
        kept_hit |= kept_count(&evs) > 0;
        assert_eq!(&collect_ids(&evs), expected);
    }
    kept_hit
}

#[test]
fn batch_mtp_paged() {
    let mut paged = Recipe::default();
    paged.share_kept(1);
    let page = paged.kept.page().expect("paged layout");
    let ctx = page * 3;
    paged.share_kept(page * 4);
    let mut t_cfg = mla_cfg(true, 16);
    t_cfg.stack.max_seqs = 3;
    t_cfg.stack.max_context = ctx as u64;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let t_weights = weights(&t_graph, &mut Noise(131));
    let (t_stored, m_stored) = (float_weights(&t_graph, 131), float_weights(&m_graph, 137));
    let settings = host_settings();
    let mut baseline = open_target(&t_graph, &t_weights, ctx, &settings);
    let base: Vec<u32> = (0..(page * 2 + 5) as u32)
        .map(|i| (i * 7 + 3) % 16)
        .collect();
    let mut p1 = base.clone();
    *p1.last_mut().unwrap() = (base[base.len() - 1] + 1) % 16;
    let prompts = [(base, 8usize), (vec![7u32, 1, 5, 2, 4], 7), (p1, 8)];
    let expected = greedy_answers(&mut baseline, &prompts);
    for reuse in [false, true] {
        let mut target = Engine::open(t_graph.clone(), ctx, &paged, &settings).unwrap();
        target.load(&t_stored).unwrap();
        let mut drafter = MtpDrafter::open(m_graph.clone(), ctx, &paged, &settings).unwrap();
        drafter.load(&m_stored).unwrap();
        let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
            .unwrap()
            .with_prefill_chunk(page + 3);
        if reuse {
            batch.reuse();
        }
        assert_eq!(run_batch_seq(&mut batch, &prompts, &expected), reuse);
    }
}

#[test]
fn batch_mtp_paged_ranked() {
    let mut paged = Recipe::default();
    paged.share_kept(1);
    let page = paged.kept.page().expect("paged layout");
    let ctx = page * 3;
    paged.share_kept(page * 3);
    let mut t_cfg = mla_cfg(true, 512);
    t_cfg.stack.vocab = 128;
    t_cfg.stack.max_seqs = 2;
    t_cfg.stack.max_context = ctx as u64;
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, true));
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f, r_f) = (
        t_graph.clone(),
        m_graph.clone(),
        settings.clone(),
        paged.clone(),
    );
    let open_pair =
        move |tg: Graph, mg: Graph, r: &Recipe, s: &tessera_lower::Settings, sum, ctl| {
            let [ts, ms] = Drafter::share_transport(sum);
            let (tw, mw) = (float_weights(&tg, 149), float_weights(&mg, 151));
            let mut te = Engine::open_ranked(tg, ctx, (r, s), two_place(true), ts, ctl).unwrap();
            te.load(&tw).unwrap();
            let mut dr = MtpDrafter::open_ranked(mg, ctx, (r, s), two_place(true), ms).unwrap();
            dr.load(&mw).unwrap();
            (te, dr)
        };
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_pair(tg_f, mg_f, &r_f, &s_f, Box::new(s1), Box::new(c1));
        let mut batch = BatchMtp::new(&mut te, &mut dr, 2)
            .unwrap()
            .with_prefill_chunk(page + 3);
        batch.reuse();
        batch.follow().unwrap();
    });
    let (mut target, mut drafter) = open_pair(
        t_graph.clone(),
        m_graph,
        &paged,
        &settings,
        Box::new(s0),
        Box::new(c0),
    );
    let t_weights = weights(&t_graph, &mut Noise(149));
    let mut baseline = open_target(&t_graph, &t_weights, ctx, &settings);
    let base: Vec<u32> = (0..(page * 2 + 5) as u32)
        .map(|i| (i * 5 + 2) % 16)
        .collect();
    let mut p1 = base.clone();
    *p1.last_mut().unwrap() = (base[base.len() - 1] + 1) % 16;
    let prompts = [(base, 7usize), (p1, 7)];
    let expected = greedy_answers(&mut baseline, &prompts);
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(page + 3);
    batch.reuse();
    assert!(run_batch_seq(&mut batch, &prompts, &expected));
    drop(batch);
    target.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn solo_mtp_paged() {
    let mut paged = Recipe::default();
    paged.share_kept(1);
    let page = paged.kept.page().expect("paged layout");
    let ctx = page * 3;
    paged.share_kept(page * 4);
    let mut cfg = mla_cfg(true, 16);
    (cfg.stack.max_seqs, cfg.stack.max_context) = (2, ctx as u64);
    let tg = mla::stepper::<mla::Moe>(&cfg);
    let mg = mtp::stepper::<mla::Moe>(&mtp::config(&cfg, true));
    let (tw, mw) = (weights(&tg, &mut Noise(157)), weights(&mg, &mut Noise(163)));
    let (ts, s) = (float_weights(&tg, 157), host_settings());
    let p: Vec<u32> = (0..(page + 5) as u32).map(|i| (i * 5 + 1) % 16).collect();
    let want = gen_ids(&mut open_target(&tg, &tw, ctx, &s), req(&p, 8));
    let mut target = Engine::open(tg, ctx, &paged, &s).unwrap();
    target.load(&ts).unwrap();
    let mut drafter = open_mtp(&mg, &mw, ctx, &s);
    let mut sp = MtpSpeculate::new(&mut target, &mut drafter, 2)
        .unwrap()
        .with_prefill_chunk(page);
    let (got, _) = spec_events(|e| sp.generate(req(&p, 8), e));
    assert_eq!(collect_ids(&got), want);
    let long: Vec<u32> = (0..=ctx as u32).map(|i| i % 16).collect();
    let (cap, _) = spec_events(|e| sp.generate(req(&long, 4), e));
    assert_eq!(cap, vec![Event::Done(Reason::Context)]);
}

#[test]
fn taps_only_skips_head() {
    let mut cfg = mla_cfg(true, 512);
    cfg.stack.vocab = 600;
    let tg = mla::stepper::<mla::Moe>(&cfg);
    let mg = mtp::stepper::<mla::Moe>(&mtp::config(&cfg, true));
    let parts = Parts::even(2);
    let sh = mtp::shard(&mg, &parts, &|_, _, _, _| true, true)
        .unwrap()
        .graph;
    let sharded = sh.wired(DType::F32);
    let rows = 4u64;
    let g = plan_gathers(&sharded, &parts, rows, Wants::role(&sharded, 0, Role::Taps));
    assert!(!g.contains(&(2 * rows as usize * 2)) && !g.contains(&(rows as usize * 600)));
    let dev = Device::guess(&host_settings());
    let dims = [(Seqs, 1), (Rows, rows), (Context, rows), (Segments, 1)];
    let mut cache = Cache::default();
    let mut count = |w| {
        let p = cache
            .wanting(&tg, 0, w, &dims, &[Context, Segments], &dev)
            .unwrap()
            .0;
        (
            p.steps.len(),
            p.steps.iter().any(|s| s.write.len == rows as usize * 600),
        )
    };
    let (ts, tv) = count(Wants::role(&tg, 0, Role::Taps));
    let (bs, bv) = count(Wants::roles(&tg, 0, &[Role::Tokens, Role::Taps]));
    assert!(ts < bs && bv && !tv);
    // Multi-chunk prefill with max_tokens=1 and max_tokens=6 matches baseline.
    let mut c1 = mla_cfg(false, 16);
    c1.stack.max_seqs = 2;
    let g1 = mla::stepper::<mla::Moe>(&c1);
    let m1 = mtp::stepper::<mla::Moe>(&mtp::config(&c1, false));
    let (tw, mw, s) = (
        weights(&g1, &mut Noise(173)),
        weights(&m1, &mut Noise(179)),
        host_settings(),
    );
    let mut base = open_target(&g1, &tw, 24, &s);
    let mut target = open_target(&g1, &tw, 24, &s);
    let mut drafter = open_mtp(&m1, &mw, 24, &s);
    let prompt = [2u32, 5, 1, 8, 3, 6, 4];
    let mut sp = MtpSpeculate::new(&mut target, &mut drafter, 2).unwrap();
    sp = sp.with_prefill_chunk(2);
    for max_tokens in [1, 6] {
        let (got, _) = spec_events(|e| sp.generate(req(&prompt, max_tokens), e));
        assert_eq!(
            collect_ids(&got),
            gen_ids(&mut base, req(&prompt, max_tokens))
        );
    }
}

#[test]
fn glm53_tp4_skips_head() {
    // Full 78-layer GLM-5.3 TP=4 32K prefill on GB10 CUDA platform.
    let mut cfg = mla::Config::glm53(4, 32_768);
    cfg.stack.taps = vec![mtp::tap_layer(&cfg)];
    let vocab = cfg.stack.vocab as usize;
    let tg = mla::stepper::<mla::RotatedMoe>(&cfg);
    let mut gpu = Platform::assumed(Limits::assumed());
    (gpu.backend, gpu.name, gpu.units, gpu.bandwidth) =
        ("cuda".into(), "NVIDIA GB10".into(), 48, 273e9);
    gpu.row.as_mut().unwrap().bf16_tile = Some([16, 8, 16]);
    let bf16 = "embed head *.norm *.q_norm *.kv_norm *.k_norm *.k_bias *.mlp_norm.* norm *.router *.router_bias *.weights";
    let mxfp8 = named("mxfp8").unwrap();
    let pins = bf16
        .split_whitespace()
        .fold(Pins::all(mxfp8), |p, k| p.pin(k, BF16_FORM));
    let (parts, trimmed) = (Parts::even(4), trim(&tg, 0).unwrap());
    let has_picks = trimmed.entries[0]
        .inputs
        .iter()
        .any(|i| i.role == Role::Picks);
    assert!(has_picks);
    let sh = shard_mode(&trimmed, &parts, &pins.shares(&trimmed), false)
        .unwrap()
        .graph;
    let sharded = sh.wired(DType::F32);
    let dev = Device::default()
        .on(gpu)
        .pinned(pins)
        .zoned(Cluster::mesh(4, ZoneKind::Gpu));
    let dev = dev.ranked(vec![0, 1, 2, 3]).uneven(parts.0).as_rank(0);
    let dims = [
        (Seqs, 4),
        (Rows, 1024),
        (Context, 32_768),
        (Segments, 1),
        (Picks, 1),
    ];
    let mut cache = Cache::default();
    let mut stats = |w| {
        let p = cache
            .wanting(&sharded, 0, w, &dims, &[Context, Segments], &dev)
            .unwrap()
            .0;
        assert_eq!(p.ranks, 4);
        let h = p
            .steps
            .iter()
            .filter(|s| s.kernel.name() != "all_gather" && s.write.len == vocab / 4)
            .count();
        let g: Vec<_> = p
            .steps
            .iter()
            .filter(|s| s.kernel.name() == "all_gather")
            .map(|s| s.write.len)
            .collect();
        (p.steps.len(), h, g)
    };
    let (ts, th, tg_g) = stats(Wants::role(&sharded, 0, Role::Taps));
    let (bs, bh, bg_g) = stats(Wants::roles(&sharded, 0, &[Role::Tokens, Role::Taps]));
    // Proves spec_loop whole-step fallback (taps > both) never triggers.
    assert!(ts < bs && th == 0 && bh >= 1 && !tg_g.contains(&vocab) && bg_g.contains(&vocab));
    // BatchMtp multi-chunk prefill invokes head once (final chunk only).
    let mut c1 = mla_cfg(false, 16);
    (c1.stack.vocab, c1.stack.max_seqs) = (80, 2);
    let g1 = mla::stepper::<mla::Moe>(&c1);
    let m1 = mtp::stepper::<mla::Moe>(&mtp::config(&c1, false));
    let (tw, mw, hs) = (
        weights(&g1, &mut Noise(181)),
        weights(&m1, &mut Noise(191)),
        host_settings(),
    );
    let mut ps = host_settings();
    ps.profile = Some(u64::MAX);
    let mut base = open_target(&g1, &tw, 24, &hs);
    let mut target = open_target(&g1, &tw, 24, &ps);
    let mut drafter = open_mtp(&m1, &mw, 24, &hs);
    let prompt = vec![1u32, 3, 5, 7, 2, 4, 6, 8];
    let expected = greedy_answers(&mut base, &[(prompt.clone(), 1)]);
    let mut batch = BatchMtp::new(&mut target, &mut drafter, 2).unwrap();
    batch = batch.with_prefill_chunk(2);
    run_batch_seq(&mut batch, &[(prompt, 1)], &expected);
    drop(batch);
    let prof = target.take_profile().expect("profile active");
    let cnt = |f: fn(&str) -> bool| -> u64 {
        prof.spent
            .iter()
            .filter(|(k, _)| f(k))
            .map(|(_, v)| v.count)
            .sum()
    };
    assert_eq!(prof.spent.get("engine: launch").map_or(0, |v| v.count), 4);
    assert_eq!(
        (
            cnt(|k| k.contains("x80x64")),
            cnt(|k| k.starts_with("argmax "))
        ),
        (1, 1)
    );
}
