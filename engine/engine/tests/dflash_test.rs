//! DFlash speculative decoding and tapped step tests.

use tessera_engine::{
    Drafter, Engine, Event, Reason, Recipe, Request, Row, SpecStats, Speculate, accept_greedy,
    accept_sample,
};
use tessera_graph::Graph;
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::Weight;
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::dflash::{self, TargetSpec};
use tessera_models::{mla, small};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

fn step_feeds(rows: &[Row]) -> Vec<Tensor> {
    let col = |of: fn(&Row) -> u32| Tensor::u32(vec![rows.len()], rows.iter().map(of).collect());
    vec![col(|r| r.token), col(|r| r.position), col(|r| r.slot)]
}

fn collect_ids(events: &[Event]) -> Vec<u32> {
    let mut out = Vec::new();
    for e in events {
        if let Event::Token(id) = e {
            out.push(*id);
        }
    }
    out
}

fn open_target(g: &Graph, w: &[Tensor], ctx: usize, s: &tessera_lower::Settings) -> Engine {
    let mut engine = Engine::open(g.clone(), ctx, &Recipe::default(), s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    engine.load(&stored).unwrap();
    engine
}

fn open_drafter(
    g: &Graph,
    w: &[Tensor],
    ctx: usize,
    block: usize,
    mask: u32,
    s: &tessera_lower::Settings,
) -> Drafter {
    let mut drafter = Drafter::open(g.clone(), ctx, block, mask, &Recipe::default(), s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    drafter.load(&stored).unwrap();
    drafter
}

#[test]
fn tapped_step_matches() {
    let mut dec_cfg = Config::toy(1, 16);
    dec_cfg.stack.taps = vec![1, 2];
    let mut mla_cfg = small::deepseek();
    mla_cfg.stack.vocab = 600;
    mla_cfg.stack.max_seqs = 1;
    mla_cfg.stack.max_context = 16;
    mla_cfg.stack.taps = vec![1, 2];
    let graphs = [
        stepper::<Norm, Swiglu>(&dec_cfg),
        mla::stepper::<mla::Moe>(&mla_cfg),
    ];
    let launches = [
        vec![row(0, 3, 0), row(0, 9, 1), row(0, 4, 2)],
        vec![row(0, 7, 3), row(0, 2, 4)],
        // Overwrites position 4 to verify KV rewind.
        vec![row(0, 5, 4), row(0, 1, 5)],
    ];
    for settings in runs() {
        for g in &graphs {
            let w = weights(g, &mut Noise(7));
            let mut engine = open_target(g, &w, 16, &settings);
            let mut store = Store::default();
            for rows in &launches {
                let want = run_kept(g, &w, &mut store, 0, &step_feeds(rows)).unwrap();
                let want_ids: Vec<u32> = want[0]
                    .indices()
                    .unwrap()
                    .iter()
                    .map(|&i| i as u32)
                    .collect();
                let want_taps = want[2].floats().unwrap();
                let (got_ids, got_taps) = engine.step_taps(rows).unwrap();
                assert_eq!(got_ids, want_ids);
                assert_eq!(got_taps.len(), want_taps.len());
                for (a, b) in got_taps.iter().zip(want_taps) {
                    assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{a} vs {b}");
                }
            }
        }
    }
}

#[test]
fn dflash_matches_greedy() {
    let mut t_cfg = Config::toy(1, 32);
    t_cfg.stack.taps = vec![1, 2];
    let t_graph = stepper::<Norm, Swiglu>(&t_cfg);
    let t_weights = weights(&t_graph, &mut Noise(19));
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: t_cfg.tied,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 32);
    let d_graph = dflash::stepper::<Norm>(&d_cfg);
    let d_weights = weights(&d_graph, &mut Noise(23));
    let prompt = vec![3u32, 9, 4, 11];
    for settings in runs() {
        let mut baseline = open_target(&t_graph, &t_weights, 32, &settings);
        let mut target = open_target(&t_graph, &t_weights, 32, &settings);
        let mut drafter = open_drafter(&d_graph, &d_weights, 32, 4, 0, &settings);
        let req = |max_tokens, stop| Request {
            tokens: prompt.clone(),
            max_tokens,
            stop,
            ..Request::default()
        };
        let mut want_ev = Vec::new();
        baseline
            .generate(req(12, Vec::new()), &mut |e| want_ev.push(e))
            .unwrap();
        let mut got_ev = Vec::new();
        let mut spec_run = Speculate::new(&mut target, &mut drafter).unwrap();
        let stats = spec_run
            .generate(req(12, Vec::new()), &mut |e| {
                got_ev.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got_ev), collect_ids(&want_ev));
        assert_eq!(got_ev.last(), Some(&Event::Done(Reason::Length)));
        assert!(stats.steps > 0);
        // Stop token ends at the exact same offset.
        let stop_id = collect_ids(&want_ev)[5];
        let mut stop_want = Vec::new();
        baseline
            .generate(req(12, vec![stop_id]), &mut |e| stop_want.push(e))
            .unwrap();
        let mut stop_ev = Vec::new();
        spec_run
            .generate(req(12, vec![stop_id]), &mut |e| {
                stop_ev.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&stop_ev), collect_ids(&stop_want));
        assert_eq!(stop_ev.last(), Some(&Event::Done(Reason::Stop)));
        // Context tail falls back cleanly past block_size.
        let mut full_want = Vec::new();
        baseline
            .generate(req(40, Vec::new()), &mut |e| full_want.push(e))
            .unwrap();
        let mut full_got = Vec::new();
        spec_run
            .generate(req(40, Vec::new()), &mut |e| {
                full_got.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&full_got), collect_ids(&full_want));
        assert_eq!(full_got.last(), Some(&Event::Done(Reason::Context)));
    }
}

fn verify_pair(
    t_graph: &Graph,
    t_weights: &[Tensor],
    d_graph: &Graph,
    d_weights: &[Tensor],
    max_tokens: usize,
    causal: bool,
) -> SpecStats {
    let mut last = SpecStats::default();
    for settings in runs() {
        let mut baseline = open_target(t_graph, t_weights, 24, &settings);
        let mut target = open_target(t_graph, t_weights, 24, &settings);
        let mut drafter = open_drafter(d_graph, d_weights, 24, 4, 0, &settings);
        drafter.set_causal(causal);
        let req = || Request {
            tokens: vec![2, 6, 1],
            max_tokens,
            ..Request::default()
        };
        let mut want_ev = Vec::new();
        baseline.generate(req(), &mut |e| want_ev.push(e)).unwrap();
        let mut got_ev = Vec::new();
        last = Speculate::new(&mut target, &mut drafter)
            .unwrap()
            .generate(req(), &mut |e| {
                got_ev.push(e);
                true
            })
            .unwrap();
        assert_eq!(collect_ids(&got_ev), collect_ids(&want_ev));
    }
    last
}

#[test]
fn dflash_accepts_and_mla() {
    let mut mla_cfg = small::deepseek();
    mla_cfg.stack.max_seqs = 1;
    mla_cfg.stack.max_context = 24;
    let spec = TargetSpec {
        vocab: mla_cfg.stack.vocab,
        hidden: mla_cfg.stack.hidden,
        layers: mla_cfg.stack.layers,
        tied: false,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    mla_cfg.stack.taps = d_cfg.tap_layers();
    let t_graph = mla::stepper::<mla::Moe>(&mla_cfg);
    let mut t_weights = weights(&t_graph, &mut Noise(31));
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let mut d_weights = weights(&d_graph, &mut Noise(37));
    // Zero both output heads so both predict token 0.
    let zero_head = |t: &mut Tensor| {
        *t = Tensor::f32(t.shape.clone(), vec![0.0; t.len()]);
    };
    let last_t = t_weights.len() - 1;
    let last_d = d_weights.len() - 1;
    zero_head(&mut t_weights[last_t]);
    zero_head(&mut d_weights[last_d]);
    let stats = verify_pair(&t_graph, &t_weights, &d_graph, &d_weights, 10, false);
    assert!(
        stats.accepted > 0,
        "expected accepted drafts, got {stats:?}"
    );
    assert!(stats.steps < stats.emitted);
}

#[test]
fn dflash2_selector() {
    let mut mla_cfg = small::deepseek();
    mla_cfg.stack.vocab = 600;
    mla_cfg.stack.max_seqs = 1;
    mla_cfg.stack.max_context = 24;
    let spec = TargetSpec {
        vocab: mla_cfg.stack.vocab,
        hidden: mla_cfg.stack.hidden,
        layers: mla_cfg.stack.layers,
        tied: false,
    };
    let mut d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    d_cfg.sliding_window = Some(16);
    d_cfg.conv = Some(dflash::ConvConfig {
        taps: 2,
        group_size: 4,
    });
    d_cfg.selector = Some(dflash::SelectorConfig { rank: 8, top_k: 4 });
    mla_cfg.stack.taps = d_cfg.tap_layers();
    let t_graph = mla::stepper::<mla::Moe>(&mla_cfg);
    let t_weights = weights(&t_graph, &mut Noise(41));
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let d_weights = weights(&d_graph, &mut Noise(43));
    let stats = verify_pair(&t_graph, &t_weights, &d_graph, &d_weights, 8, false);
    assert!(stats.emitted == 8 && stats.steps > 0);
}

#[test]
fn dspine_bridge() {
    let mut t_cfg = Config::toy(1, 24);
    t_cfg.stack.taps = vec![1, 2];
    let t_graph = stepper::<Norm, Swiglu>(&t_cfg);
    let t_weights = weights(&t_graph, &mut Noise(47));
    let spec = TargetSpec {
        vocab: t_cfg.stack.vocab,
        hidden: t_cfg.stack.hidden,
        layers: t_cfg.stack.layers,
        tied: t_cfg.tied,
    };
    let mut d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    d_cfg.stack.layers = 2;
    d_cfg.causal = true;
    d_cfg.conv = Some(dflash::ConvConfig {
        taps: 2,
        group_size: 4,
    });
    d_cfg.bridge = Some(dflash::BridgeConfig {
        code_size: 16,
        rank: 8,
        top_k: 4,
        eps: 1e-6,
    });
    let d_graph = dflash::stepper::<Norm>(&d_cfg);
    let d_weights = weights(&d_graph, &mut Noise(53));
    let stats = verify_pair(&t_graph, &t_weights, &d_graph, &d_weights, 8, true);
    assert!(stats.emitted == 8 && stats.steps > 0);
}

#[test]
fn accept_rules() {
    assert_eq!(accept_greedy(&[1, 2, 3], &[1, 2, 3, 4]), (3, 4));
    assert_eq!(accept_greedy(&[1, 2, 3], &[1, 9, 3, 4]), (1, 9));
    assert_eq!(accept_greedy(&[1, 2, 3], &[8, 2, 3, 4]), (0, 8));
    let q0 = [0.2f32, 0.8, 0.0];
    let p0 = [0.5f32, 0.4, 0.1];
    let p1 = [0.1f32, 0.2, 0.7];
    // Ratio p/q = 0.5 rejects at u = 0.7.
    let (acc, bonus) = accept_sample(&[1], &[&q0], &[&p0, &p1], &[0.7, 0.5]);
    assert_eq!((acc, bonus), (0, 0));
    // Accepts at u = 0.3 and draws bonus from p1.
    let (acc, bonus) = accept_sample(&[1], &[&q0], &[&p0, &p1], &[0.3, 0.5]);
    assert_eq!((acc, bonus), (1, 2));
}
