//! Tests for deterministic SpecLen and dual-drafter decoding.

mod common;

use common::{
    collect_ids, gen_ids, host_settings, loopback_links, open_drafter, open_mtp, open_target,
    open_tp_dual, req, spec_events, zero_last,
};
use tessera_engine::{
    BatchMtp, CostModel, DualSelector, DualSpeculate, Event, MtpSpeculate, Reason, SpecLen,
    Speculate,
};
use tessera_graph::Graph;
use tessera_models::dflash::{self, TargetSpec};
use tessera_models::{mla, mtp};
use tessera_testkit::{Noise, weights};

fn toy_dual_cfgs(seqs: u64, ctx: u64) -> (mla::Config, dflash::Config) {
    let mut t_cfg = mla::Config::deepseek_small();
    t_cfg.stack.max_seqs = seqs;
    t_cfg.stack.max_context = ctx;
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: false,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, ctx);
    (t_cfg, d_cfg)
}

fn toy_dual_graphs(seqs: u64, ctx: u64) -> (Graph, Graph, Graph) {
    let (mut t_cfg, d_cfg) = toy_dual_cfgs(seqs, ctx);
    t_cfg.stack.taps = d_cfg
        .tap_layers()
        .into_iter()
        .chain([mtp::tap_layer(&t_cfg)])
        .collect();
    (
        mla::stepper::<mla::Moe>(&t_cfg),
        dflash::stepper::<()>(&d_cfg),
        mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, false)),
    )
}

fn dual_sel() -> DualSelector {
    DualSelector::new(
        SpecLen::adaptive(3, CostModel::dual_dflash()),
        SpecLen::adaptive(2, CostModel::dual_mtp()).with_mtp_seeds(),
    )
    .with_dwell(2)
    .with_probe(2)
}

#[test]
fn spec_len_policy() {
    assert_eq!(SpecLen::bucket(f32::NAN), 0);
    assert_eq!(SpecLen::bucket(f32::NEG_INFINITY), 0);
    assert_eq!(SpecLen::bucket(0.0), 7);
    assert_eq!(SpecLen::bucket((0.5f64).ln() as f32), 2);

    let mut high = SpecLen::adaptive(7, CostModel::dflash());
    assert_eq!(high.levels(), &[1, 2, 3, 5, 7]);
    assert_eq!(
        SpecLen::adaptive(15, CostModel::dflash()).levels(),
        &[1, 2, 3, 5, 7, 11, 15]
    );
    let mut low = SpecLen::adaptive(7, CostModel::dflash());
    let mut rank_b = SpecLen::adaptive(7, CostModel::dflash());
    let lps = [-0.01f32; 7];
    for _ in 0..40 {
        high.observe(7, 7, Some(&lps));
        rank_b.observe(7, 7, Some(&lps));
        low.observe(7, 0, None);
    }
    let kh = high.choose(Some(&lps), 7, 7);
    assert_eq!(kh, rank_b.choose(Some(&lps), 7, 7));
    assert_eq!(kh, 7);
    assert_eq!(low.choose(None, 7, 7), 1);

    let trunc = SpecLen::fixed(3);
    assert_eq!(trunc.choose(None, 7, 7), 3);
    assert_eq!(trunc.choose(None, 2, 7), 2);

    let mut mtp_low = SpecLen::adaptive(3, CostModel::mtp()).with_mtp_seeds();
    for _ in 0..30 {
        mtp_low.observe(2, 0, None);
    }
    assert!(!mtp_low.more(None, 1));
}

#[test]
fn dual_selector_switching() {
    let d = SpecLen::adaptive(7, CostModel::dual_dflash());
    let m = SpecLen::adaptive(3, CostModel::dual_mtp()).with_mtp_seeds();
    let mut sel = DualSelector::new(d, m).with_dwell(3).with_probe(2);
    // Prose regime: DFlash rejects, MTP accepts pos 1.
    for _ in 0..8 {
        if !sel.decide_next(true).0 {
            sel.observe_dflash(7, 0, None, Some(9), 9);
        } else {
            sel.observe_mtp(3, 2, &[9, 9, 9]);
        }
    }
    assert!(sel.use_mtp(), "expected switch to MTP on prose");
    // Code regime: MTP drops while DFlash probes match all 7.
    let mut probed = false;
    for _ in 0..24 {
        if sel.decide_next(true).0 {
            if sel.should_probe(true) {
                sel.start_probe(vec![7; 7], None);
                probed = true;
            }
            sel.observe_mtp(3, 0, &[7]);
        } else {
            sel.observe_dflash(7, 7, None, Some(1), 7);
        }
    }
    assert!(probed);
    assert!(!sel.use_mtp(), "expected switch back to DFlash on code");
}

#[test]
fn adaptive_and_dual_greedy() {
    let settings = host_settings();
    let (t_cfg, d_cfg) = toy_dual_cfgs(1, 24);
    let m_cfg = mtp::config(&t_cfg, false);
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&m_cfg);
    let prompt = [3u32, 9, 4, 11];

    // Truncated and adaptive DFlash plus adaptive MTP.
    let mut d_tcfg = t_cfg.clone();
    d_tcfg.stack.taps = d_cfg.tap_layers();
    let dt_graph = mla::stepper::<mla::Moe>(&d_tcfg);
    let dt_w = weights(&dt_graph, &mut Noise(11));
    let d_w = weights(&d_graph, &mut Noise(13));
    let want_d = gen_ids(
        &mut open_target(&dt_graph, &dt_w, 24, &settings),
        req(&prompt, 10),
    );
    for sl in [SpecLen::fixed(2), SpecLen::adaptive(3, CostModel::dflash())] {
        let mut target = open_target(&dt_graph, &dt_w, 24, &settings);
        let mut dr = open_drafter(&d_graph, &d_w, 24, 4, 0, &settings);
        let (got, stats) = spec_events(|e| {
            Speculate::new(&mut target, &mut dr)
                .unwrap()
                .with_spec_len(sl)
                .generate(req(&prompt, 10), e)
        });
        assert_eq!(collect_ids(&got), want_d);
        assert!(stats.dflash_steps > 0);
    }

    let mut m_tcfg = t_cfg.clone();
    m_tcfg.stack.taps = vec![mtp::tap_layer(&m_tcfg)];
    let mt_graph = mla::stepper::<mla::Moe>(&m_tcfg);
    let mt_w = weights(&mt_graph, &mut Noise(17));
    let m_w = weights(&m_graph, &mut Noise(19));
    let want_m = gen_ids(
        &mut open_target(&mt_graph, &mt_w, 24, &settings),
        req(&prompt, 10),
    );
    {
        let mut target = open_target(&mt_graph, &mt_w, 24, &settings);
        let mut dr = open_mtp(&m_graph, &m_w, 24, &settings);
        let sl = SpecLen::adaptive(3, CostModel::mtp()).with_mtp_seeds();
        let (got, stats) = spec_events(|e| {
            MtpSpeculate::new(&mut target, &mut dr, 3)
                .unwrap()
                .with_spec_len(sl)
                .generate(req(&prompt, 10), e)
        });
        assert_eq!(collect_ids(&got), want_m);
        assert!(stats.mtp_steps > 0 && dr.plan_hits() > 0);
    }

    // Combined-tap target for single-rank DualSpeculate.
    let (dual_graph, _, _) = toy_dual_graphs(1, 24);
    let mut dual_tw = weights(&dual_graph, &mut Noise(23));
    let mut dual_mw = weights(&m_graph, &mut Noise(29));
    zero_last(&mut dual_tw);
    zero_last(&mut dual_mw);
    let mut base = open_target(&dual_graph, &dual_tw, 24, &settings);
    let mut target = open_target(&dual_graph, &dual_tw, 24, &settings);
    let mut dr = open_drafter(&d_graph, &d_w, 24, 4, 0, &settings);
    let mut mtp_dr = open_mtp(&m_graph, &dual_mw, 24, &settings);
    let mut dual = DualSpeculate::new(&mut target, &mut dr, &mut mtp_dr, 2)
        .unwrap()
        .with_selector(dual_sel());
    let (got, stats) = spec_events(|e| dual.generate(req(&prompt, 12), e));
    assert_eq!(collect_ids(&got), gen_ids(&mut base, req(&prompt, 12)));
    assert_eq!(got.last(), Some(&Event::Done(Reason::Length)));
    assert!(stats.dflash_steps > 0 && stats.mtp_steps > 0 && stats.switches > 0);
    let (cg, _) = spec_events(|e| dual.generate(req(&prompt, 30), e));
    assert_eq!(collect_ids(&cg), gen_ids(&mut base, req(&prompt, 30)));
    assert_eq!(cg.last(), Some(&Event::Done(Reason::Context)));
}

#[test]
fn dual_tp_matches_baseline() {
    let settings = host_settings();
    let (t_graph, d_graph, m_graph) = toy_dual_graphs(1, 24);
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, mg_f, s_f) = (
        t_graph.clone(),
        d_graph.clone(),
        m_graph.clone(),
        settings.clone(),
    );
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr, mut mtp) = open_tp_dual(
            (&tg_f, &dg_f, &mg_f),
            (31, 37, 41),
            24,
            &s_f,
            (Box::new(s1), c1),
        );
        DualSpeculate::new(&mut te, &mut dr, &mut mtp, 2)
            .unwrap()
            .follow()
            .unwrap();
    });
    let (mut target, mut dr, mut mtp) = open_tp_dual(
        (&t_graph, &d_graph, &m_graph),
        (31, 37, 41),
        24,
        &settings,
        (Box::new(s0), c0),
    );
    let mut baseline = open_target(&t_graph, &weights(&t_graph, &mut Noise(31)), 24, &settings);
    let mut dual = DualSpeculate::new(&mut target, &mut dr, &mut mtp, 2)
        .unwrap()
        .with_selector(dual_sel())
        .with_prefill_chunk(3);
    for prompt in [&[2u32, 6, 1][..], &[4, 1, 5, 3, 2, 7, 1][..]] {
        let (got, stats) = spec_events(|e| dual.generate(req(prompt, 10), e));
        assert_eq!(collect_ids(&got), gen_ids(&mut baseline, req(prompt, 10)));
        assert!(stats.steps > 0);
    }
    target.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn batch_dual_matches_target() {
    let settings = host_settings();
    let (t_graph, d_graph, m_graph) = toy_dual_graphs(2, 32);
    let t_w = weights(&t_graph, &mut Noise(53));
    let d_w = weights(&d_graph, &mut Noise(59));
    let m_w = weights(&m_graph, &mut Noise(61));
    let mut base = open_target(&t_graph, &t_w, 32, &settings);
    let reqs: [(&[u32], usize); 2] = [(&[2, 6, 1, 4], 6), (&[7, 3, 5], 16)];
    let wants: Vec<Vec<u32>> = reqs
        .iter()
        .map(|&(p, max)| gen_ids(&mut base, req(p, max)))
        .collect();
    let mut target = open_target(&t_graph, &t_w, 32, &settings);
    let mut dr = open_drafter(&d_graph, &d_w, 32, 4, 0, &settings);
    let mut mtp_dr = open_mtp(&m_graph, &m_w, 32, &settings);
    let mut batch = BatchMtp::new_dual(&mut target, &mut dr, &mut mtp_dr, 2, dual_sel())
        .unwrap()
        .with_prefill_chunk(3);
    let (tx0, rx0) = std::sync::mpsc::channel();
    batch.submit(
        req(reqs[0].0, reqs[0].1),
        Box::new(move |e| tx0.send(e).is_ok()),
    );
    // Run solo DFlash rounds, then demote and promote tail.
    assert!(batch.step().unwrap());
    assert!(batch.step().unwrap());
    let (tx1, rx1) = std::sync::mpsc::channel();
    batch.submit(
        req(reqs[1].0, reqs[1].1),
        Box::new(move |e| tx1.send(e).is_ok()),
    );
    while batch.step().unwrap() {}
    for (rx, want) in [rx0, rx1].into_iter().zip(&wants) {
        let evs: Vec<_> = rx.try_iter().collect();
        assert_eq!(&collect_ids(&evs), want);
        assert_eq!(evs.last(), Some(&Event::Done(Reason::Length)));
    }
}
