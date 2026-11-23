//! Native MTP speculative decoding tests (single-rank and TP).

mod common;

use common::{collect_ids, float_weights, loopback_links, open_ranked_target, open_target};
use tessera_engine::{
    Drafter, Event, MtpDrafter, MtpSpeculate, Place, Reason, Recipe, Request, SpecStats,
};
use tessera_graph::DimName::{Context, Hidden, Rows, Segments, Seqs, Vocab};
use tessera_graph::{Graph, Parts, PathSeg, Role};
use tessera_interp::{Tensor, run, run_ranks};
use tessera_lower::{Cache, Cluster, Device, Pick, Settings, Wants, Weight};
use tessera_models::{mla, mtp, small};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

fn mla_cfg(indexed: bool, ffn: u64) -> mla::Config {
    let mut cfg = small::deepseek();
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
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, mg_f, s_f) = (t_graph.clone(), m_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let [ts1, ms1] = Drafter::share_transport(Box::new(s1));
        let place = || Place::new(Cluster::two("a", "b"), Parts::even(2)).with_moe_tp(moe_tp);
        let (tw_f, mw_f) = (float_weights(&tg_f, t_seed), float_weights(&mg_f, m_seed));
        let mut te = open_ranked_target(tg_f, 24, &s_f, place(), ts1, c1, &tw_f);
        let mut dr =
            MtpDrafter::open_ranked(mg_f, 24, (&Recipe::default(), &s_f), place(), ms1).unwrap();
        dr.load(&mw_f).unwrap();
        MtpSpeculate::new(&mut te, &mut dr, depth)
            .unwrap()
            .follow()
            .unwrap();
    });
    let [ts0, ms0] = Drafter::share_transport(Box::new(s0));
    let place = || Place::new(Cluster::two("a", "b"), Parts::even(2)).with_moe_tp(moe_tp);
    let t_weights = weights(t_graph, &mut Noise(t_seed));
    let tw = float_weights(t_graph, t_seed);
    let mw = float_weights(m_graph, m_seed);
    let mut target = open_ranked_target(t_graph.clone(), 24, &settings, place(), ts0, c0, &tw);
    let mut drafter = MtpDrafter::open_ranked(
        m_graph.clone(),
        24,
        (&Recipe::default(), &settings),
        place(),
        ms0,
    )
    .unwrap();
    drafter.load(&mw).unwrap();
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
        .graph;
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
