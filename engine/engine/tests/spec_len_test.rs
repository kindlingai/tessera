//! Tests for deterministic SpecLen and dual-drafter decoding.

mod common;

use common::{collect_ids, float_weights, loopback_links, open_ranked_target, open_target};
use tessera_engine::{
    CostModel, Drafter, DualSelector, DualSpeculate, Event, MtpDrafter, MtpSpeculate, Place,
    Reason, Recipe, Request, SpecLen, Speculate,
};
use tessera_graph::Parts;
use tessera_interp::Tensor;
use tessera_lower::{Cluster, Pick, Settings, Weight};
use tessera_models::dflash::{self, TargetSpec};
use tessera_models::{mla, mtp, small};
use tessera_testkit::{Noise, weights};

fn zero_last(w: &mut [Tensor]) {
    let last = w.last_mut().unwrap();
    *last = Tensor::f32(last.shape.clone(), vec![0.0; last.len()]);
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
        high.observe(7, 7, Some(&lps), 8.0);
        rank_b.observe(7, 7, Some(&lps), 8.0);
        low.observe(7, 0, None, 1.0);
    }
    let (kh, eh) = high.choose(Some(&lps), 7, 7);
    let (kb, eb) = rank_b.choose(Some(&lps), 7, 7);
    assert_eq!((kh, eh.to_bits()), (kb, eb.to_bits()));
    assert_eq!(
        SpecLen::decision_hash(Some(&lps), kh),
        SpecLen::decision_hash(Some(&lps), kb)
    );
    assert_eq!(kh, 7);
    assert_eq!(low.choose(None, 7, 7).0, 1);

    let trunc = SpecLen::fixed(3);
    assert_eq!(trunc.choose(None, 7, 7).0, 3);
    assert_eq!(trunc.choose(None, 2, 7).0, 2);

    let mut mtp_low = SpecLen::adaptive(3, CostModel::mtp()).with_mtp_seeds();
    for _ in 0..30 {
        mtp_low.observe(2, 0, None, 1.0);
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
        let (use_mtp, _) = sel.decide_next(true);
        if !use_mtp {
            sel.observe_dflash(7, 0, 1.0, Some(9), 9);
        } else {
            sel.observe_mtp(3, 2, 3.0, &[9, 9, 9]);
        }
    }
    assert!(sel.use_mtp(), "expected switch to MTP on prose");
    // Code regime: MTP drops while DFlash probes match all 7.
    let mut probed = false;
    for _ in 0..24 {
        let (use_mtp, _) = sel.decide_next(true);
        if use_mtp {
            if sel.should_probe(true) {
                sel.start_probe(vec![7; 7]);
                probed = true;
            }
            sel.observe_mtp(3, 0, 1.0, &[7]);
        } else {
            sel.observe_dflash(7, 7, 8.0, Some(1), 7);
        }
    }
    assert!(probed);
    assert!(!sel.use_mtp(), "expected switch back to DFlash on code");
}

#[test]
fn adaptive_and_dual_greedy() {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut t_cfg = small::deepseek();
    t_cfg.stack.max_seqs = 1;
    t_cfg.stack.max_context = 24;
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: false,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    let m_cfg = mtp::config(&t_cfg, false);
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&m_cfg);

    // Truncated and adaptive DFlash plus adaptive MTP.
    let mut d_tcfg = t_cfg.clone();
    d_tcfg.stack.taps = d_cfg.tap_layers();
    let dt_graph = mla::stepper::<mla::Moe>(&d_tcfg);
    let dt_w = weights(&dt_graph, &mut Noise(11));
    let d_w = weights(&d_graph, &mut Noise(13));
    let d_stored: Vec<Weight> = d_w.iter().cloned().map(Weight::Floats).collect();
    let prompt = vec![3u32, 9, 4, 11];
    let req = |max_tokens, stop| Request {
        tokens: prompt.clone(),
        max_tokens,
        stop,
        ..Request::default()
    };
    let mut base_d = open_target(&dt_graph, &dt_w, 24, &settings);
    let mut want_d = Vec::new();
    base_d
        .generate(req(10, Vec::new()), &mut |e| want_d.push(e))
        .unwrap();
    for sl in [SpecLen::fixed(2), SpecLen::adaptive(3, CostModel::dflash())] {
        let mut target = open_target(&dt_graph, &dt_w, 24, &settings);
        let mut dr =
            Drafter::open(d_graph.clone(), 24, 4, 0, &Recipe::default(), &settings).unwrap();
        dr.load(&d_stored).unwrap();
        let mut got = Vec::new();
        let stats = Speculate::new(&mut target, &mut dr)
            .unwrap()
            .with_spec_len(sl)
            .generate(req(10, Vec::new()), &mut |e| {
                got.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got), collect_ids(&want_d));
        assert!(stats.dflash_steps > 0);
    }

    let mut m_tcfg = t_cfg.clone();
    m_tcfg.stack.taps = vec![mtp::tap_layer(&m_tcfg)];
    let mt_graph = mla::stepper::<mla::Moe>(&m_tcfg);
    let mt_w = weights(&mt_graph, &mut Noise(17));
    let m_w = weights(&m_graph, &mut Noise(19));
    let m_stored: Vec<Weight> = m_w.iter().cloned().map(Weight::Floats).collect();
    let mut base_m = open_target(&mt_graph, &mt_w, 24, &settings);
    let mut want_m = Vec::new();
    base_m
        .generate(req(10, Vec::new()), &mut |e| want_m.push(e))
        .unwrap();
    {
        let mut target = open_target(&mt_graph, &mt_w, 24, &settings);
        let mut dr = MtpDrafter::open(m_graph.clone(), 24, &Recipe::default(), &settings).unwrap();
        dr.load(&m_stored).unwrap();
        let mut got = Vec::new();
        let sl = SpecLen::adaptive(3, CostModel::mtp()).with_mtp_seeds();
        let stats = MtpSpeculate::new(&mut target, &mut dr, 3)
            .unwrap()
            .with_spec_len(sl)
            .generate(req(10, Vec::new()), &mut |e| {
                got.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got), collect_ids(&want_m));
        assert!(stats.mtp_steps > 0);
        assert!(dr.plan_hits() > 0);
    }

    // Combined-tap target for single-rank DualSpeculate.
    let mut dual_tcfg = t_cfg.clone();
    dual_tcfg.stack.taps = d_cfg
        .tap_layers()
        .into_iter()
        .chain([mtp::tap_layer(&t_cfg)])
        .collect();
    let dual_graph = mla::stepper::<mla::Moe>(&dual_tcfg);
    let mut dual_tw = weights(&dual_graph, &mut Noise(23));
    let mut dual_mw = weights(&m_graph, &mut Noise(29));
    zero_last(&mut dual_tw);
    zero_last(&mut dual_mw);
    let dual_ms: Vec<Weight> = dual_mw.iter().cloned().map(Weight::Floats).collect();
    let mut base = open_target(&dual_graph, &dual_tw, 24, &settings);
    let mut target = open_target(&dual_graph, &dual_tw, 24, &settings);
    let mut dr = Drafter::open(d_graph.clone(), 24, 4, 0, &Recipe::default(), &settings).unwrap();
    dr.load(&d_stored).unwrap();
    let mut mtp_dr = MtpDrafter::open(m_graph.clone(), 24, &Recipe::default(), &settings).unwrap();
    mtp_dr.load(&dual_ms).unwrap();
    let sel = DualSelector::new(
        SpecLen::adaptive(3, CostModel::dual_dflash()),
        SpecLen::adaptive(2, CostModel::dual_mtp()).with_mtp_seeds(),
    )
    .with_dwell(2)
    .with_probe(2);
    let mut dual = DualSpeculate::new(&mut target, &mut dr, &mut mtp_dr, 2)
        .unwrap()
        .with_selector(sel);
    let (mut want, mut got) = (Vec::new(), Vec::new());
    base.generate(req(12, Vec::new()), &mut |e| want.push(e))
        .unwrap();
    let stats = dual
        .generate(req(12, Vec::new()), &mut |e| {
            got.push(e);
            true
        })
        .unwrap();
    assert_eq!(collect_ids(&got), collect_ids(&want));
    assert_eq!(got.last(), Some(&Event::Done(Reason::Length)));
    assert!(stats.dflash_steps > 0 && stats.mtp_steps > 0 && stats.switches > 0);
    // Context tail also matches baseline.
    let (mut cw, mut cg) = (Vec::new(), Vec::new());
    base.generate(req(30, Vec::new()), &mut |e| cw.push(e))
        .unwrap();
    dual.generate(req(30, Vec::new()), &mut |e| {
        cg.push(e);
        true
    })
    .unwrap();
    assert_eq!(collect_ids(&cg), collect_ids(&cw));
    assert_eq!(cg.last(), Some(&Event::Done(Reason::Context)));
}

#[test]
fn dual_tp_matches_baseline() {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut t_cfg = small::deepseek();
    t_cfg.stack.max_seqs = 1;
    t_cfg.stack.max_context = 24;
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: false,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    t_cfg.stack.taps = d_cfg
        .tap_layers()
        .into_iter()
        .chain([mtp::tap_layer(&t_cfg)])
        .collect();
    let t_graph = mla::stepper::<mla::Moe>(&t_cfg);
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let m_graph = mtp::stepper::<mla::Moe>(&mtp::config(&t_cfg, false));
    let open_rank_triple = |sums: tessera_engine::comm::Sockets,
                            ctrl: tessera_engine::comm::Sockets,
                            tg: &tessera_graph::Graph,
                            dg: &tessera_graph::Graph,
                            mg: &tessera_graph::Graph,
                            st: &Settings| {
        let [ts, ds, ms] = Drafter::share_transports(Box::new(sums));
        let place = || Place::new(Cluster::two("a", "b"), Parts::even(2));
        let te = open_ranked_target(
            tg.clone(),
            24,
            st,
            place(),
            ts,
            ctrl,
            &float_weights(tg, 31),
        );
        let mut dr = Drafter::open_ranked(
            dg.clone(),
            24,
            4,
            0,
            (&Recipe::default(), st),
            place(),
            ds,
            None,
        )
        .unwrap();
        dr.load(&float_weights(dg, 37)).unwrap();
        let mut mtp_dr =
            MtpDrafter::open_ranked(mg.clone(), 24, (&Recipe::default(), st), place(), ms).unwrap();
        mtp_dr.load(&float_weights(mg, 41)).unwrap();
        (te, dr, mtp_dr)
    };
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, mg_f, s_f) = (
        t_graph.clone(),
        d_graph.clone(),
        m_graph.clone(),
        settings.clone(),
    );
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr, mut mtp_dr) = open_rank_triple(s1, c1, &tg_f, &dg_f, &mg_f, &s_f);
        DualSpeculate::new(&mut te, &mut dr, &mut mtp_dr, 2)
            .unwrap()
            .follow()
            .unwrap();
    });
    let (mut target, mut dr, mut mtp_dr) =
        open_rank_triple(s0, c0, &t_graph, &d_graph, &m_graph, &settings);
    let mut baseline = open_target(&t_graph, &weights(&t_graph, &mut Noise(31)), 24, &settings);
    let sel = DualSelector::new(
        SpecLen::adaptive(3, CostModel::dual_dflash()),
        SpecLen::adaptive(2, CostModel::dual_mtp()).with_mtp_seeds(),
    )
    .with_dwell(2)
    .with_probe(2);
    let mut dual = DualSpeculate::new(&mut target, &mut dr, &mut mtp_dr, 2)
        .unwrap()
        .with_selector(sel);
    let check_prompt =
        |tokens: Vec<u32>, base: &mut tessera_engine::Engine, d: &mut DualSpeculate| {
            let mut want = Vec::new();
            let mut got = Vec::new();
            let r0 = Request {
                tokens: tokens.clone(),
                max_tokens: 10,
                ..Request::default()
            };
            let r1 = Request {
                tokens,
                max_tokens: 10,
                ..Request::default()
            };
            base.generate(r0, &mut |e| want.push(e)).unwrap();
            let stats = d
                .generate(r1, &mut |e| {
                    got.push(e);
                    true
                })
                .unwrap();
            assert_eq!(collect_ids(&got), collect_ids(&want));
            assert!(stats.steps > 0);
        };
    check_prompt(vec![2, 6, 1], &mut baseline, &mut dual);
    check_prompt(vec![4, 1, 5, 3], &mut baseline, &mut dual);
    target.dismiss().unwrap();
    follower.join().unwrap();
}
