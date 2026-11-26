//! DFlash speculative decoding and tapped step tests.

mod common;
mod dflash_batch;

use common::{
    check_sharded_head, check_spec, collect_ids, dim_max, gen_ids, host_settings, kept_count,
    loopback_links, open_drafter, open_engine, open_target, open_tp_dflash, plan_gathers, req, row,
    spec_events, step_feeds, zero_last,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tessera_engine::comm::{Error as CommError, Sockets, Transport};
use tessera_engine::{
    BatchDFlash, Drafter, Engine, Event, Reason, Recipe, Request, SpecStats, Speculate,
};
use tessera_graph::DimName::{Embedding, Vocab};
use tessera_graph::{Graph, Parts, PathSeg, Role};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::{Settings, Wants};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::dflash::{self, TargetSpec};
use tessera_models::mla;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

fn target_spec(vocab: u64, hidden: u64, layers: u64, tied: bool) -> TargetSpec {
    TargetSpec {
        vocab,
        hidden,
        layers,
        tied,
    }
}

fn dflash2_cfg(spec: &TargetSpec, ctx: u64) -> dflash::Config {
    let mut cfg = dflash::Config::toy(spec, 4, 0, 1, ctx);
    cfg.sliding_window = Some(16);
    cfg.conv = Some(dflash::ConvConfig {
        taps: 2,
        group_size: 4,
    });
    cfg.selector = Some(dflash::SelectorConfig {
        rank: 8,
        top_k: 4,
        svd_rank: None,
    });
    cfg
}

fn dspine_cfg(spec: &TargetSpec, ctx: u64) -> dflash::Config {
    let mut cfg = dflash::Config::toy(spec, 4, 0, 1, ctx);
    cfg.stack.layers = 2;
    cfg.causal = true;
    cfg.conv = Some(dflash::ConvConfig {
        taps: 2,
        group_size: 4,
    });
    cfg.bridge = Some(dflash::BridgeConfig {
        code_size: 16,
        rank: 8,
        top_k: 4,
        eps: 1e-6,
    });
    cfg
}

#[test]
fn tapped_step_matches() {
    let mut dec_cfg = Config::toy(1, 16);
    dec_cfg.stack.taps = vec![1, 2];
    let mut mla_cfg = mla::Config::deepseek_small();
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
    let spec = target_spec(
        t_cfg.stack.vocab,
        t_cfg.stack.hidden,
        t_cfg.stack.layers,
        t_cfg.tied,
    );
    let d_graph = dflash::stepper::<Norm>(&dflash::Config::toy(&spec, 4, 0, 1, 32));
    let d_weights = weights(&d_graph, &mut Noise(23));
    let prompt = [3u32, 9, 4, 11];
    let long_prompt = [3u32, 9, 4, 11, 2, 7, 5, 1, 8, 6, 3, 10, 4];
    for settings in runs() {
        let mut baseline = open_target(&t_graph, &t_weights, 32, &settings);
        let mut target = open_target(&t_graph, &t_weights, 32, &settings);
        let mut drafter = open_drafter(&d_graph, &d_weights, 32, 4, 0, &settings);
        let mut spec_run = Speculate::new(&mut target, &mut drafter).unwrap();
        check_spec(
            &mut baseline,
            |r| spec_events(|e| spec_run.generate(r, e)),
            &prompt,
            [12, 5, 40],
        );
        // Chunked prefill matches baseline for single and multi-token runs.
        let mut chunked = spec_run.with_prefill_chunk(3);
        for max_tokens in [1, 8] {
            let (cg, _) = spec_events(|e| chunked.generate(req(&long_prompt, max_tokens), e));
            assert_eq!(
                collect_ids(&cg),
                gen_ids(&mut baseline, req(&long_prompt, max_tokens))
            );
        }
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
        let (got, stats) = spec_events(|e| {
            Speculate::new(&mut target, &mut drafter)
                .unwrap()
                .generate(req(&[2, 6, 1], max_tokens), e)
        });
        assert_eq!(
            collect_ids(&got),
            gen_ids(&mut baseline, req(&[2, 6, 1], max_tokens))
        );
        last = stats;
    }
    last
}

#[test]
fn dflash_accepts_and_mla() {
    let mut mla_cfg = mla::Config::deepseek_small();
    mla_cfg.stack.max_seqs = 1;
    mla_cfg.stack.max_context = 24;
    let spec = target_spec(
        mla_cfg.stack.vocab,
        mla_cfg.stack.hidden,
        mla_cfg.stack.layers,
        false,
    );
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 24);
    mla_cfg.stack.taps = d_cfg.tap_layers();
    let t_graph = mla::stepper::<mla::Moe>(&mla_cfg);
    let mut t_weights = weights(&t_graph, &mut Noise(31));
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let mut d_weights = weights(&d_graph, &mut Noise(37));
    // Zero both output heads so both predict token 0.
    zero_last(&mut t_weights);
    zero_last(&mut d_weights);
    let stats = verify_pair(&t_graph, &t_weights, &d_graph, &d_weights, 10, false);
    assert!(
        stats.accepted > 0,
        "expected accepted drafts, got {stats:?}"
    );
    assert!(stats.steps < stats.emitted);
}

#[test]
fn dflash2_selector() {
    let mut mla_cfg = mla::Config::deepseek_small();
    mla_cfg.stack.vocab = 600;
    mla_cfg.stack.max_seqs = 1;
    mla_cfg.stack.max_context = 24;
    let spec = target_spec(
        mla_cfg.stack.vocab,
        mla_cfg.stack.hidden,
        mla_cfg.stack.layers,
        false,
    );
    let d_cfg = dflash2_cfg(&spec, 24);
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
    let spec = target_spec(
        t_cfg.stack.vocab,
        t_cfg.stack.hidden,
        t_cfg.stack.layers,
        t_cfg.tied,
    );
    let d_graph = dflash::stepper::<Norm>(&dspine_cfg(&spec, 24));
    let d_weights = weights(&d_graph, &mut Noise(53));
    let stats = verify_pair(&t_graph, &t_weights, &d_graph, &d_weights, 8, true);
    assert!(stats.emitted == 8 && stats.steps > 0);
}

fn verify_tp_pair(
    t_graph: &Graph,
    t_seed: u64,
    d_graph: &Graph,
    d_seed: u64,
    causal: bool,
    expect_head_split: bool,
) {
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(
            (&tg_f, t_seed),
            (&dg_f, d_seed),
            24,
            causal,
            &s_f,
            (Box::new(s1), c1),
        );
        Speculate::new(&mut te, &mut dr).unwrap().follow().unwrap();
    });
    let (mut target, mut drafter) = open_tp_dflash(
        (t_graph, t_seed),
        (d_graph, d_seed),
        24,
        causal,
        &settings,
        (Box::new(s0), c0),
    );
    assert!(drafter.has_kernel("all_reduce"), "missing all_reduce");
    assert!(drafter.has_kernel("all_gather"), "missing all_gather");
    let at = |tail: &str| {
        d_graph
            .weights
            .iter()
            .position(|w| matches!(w.path.last(), Some(PathSeg::Field(f)) if f == tail))
            .unwrap() as u32
    };
    for name in ["fc", "q", "k", "v", "o", "gate", "up", "down"] {
        assert!(drafter.share(at(name)).is_some(), "{name} should shard");
    }
    assert_eq!(drafter.share(at("head")).is_some(), expect_head_split);
    let mut baseline = open_target(
        t_graph,
        &weights(t_graph, &mut Noise(t_seed)),
        24,
        &settings,
    );
    let mut spec = Speculate::new(&mut target, &mut drafter)
        .unwrap()
        .with_prefill_chunk(3);
    for prompt in [&[2u32, 6, 1][..], &[4, 1, 5, 3, 7, 2, 8][..]] {
        let (got, stats) = spec_events(|e| spec.generate(req(prompt, 7), e));
        assert_eq!(collect_ids(&got), gen_ids(&mut baseline, req(prompt, 7)));
        assert!(stats.steps > 0);
    }
    target.dismiss().unwrap();
    follower.join().unwrap();
}

fn verify_local_topk(g: &Graph, seed: u64, k: usize, selector: bool) {
    let parts = Parts::even(2);
    let sharded = dflash::shard(g, &parts, &|_, _, _, _| true)
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let (vocab, tap_w) = (dim_max(g, Vocab), dim_max(g, Embedding));
    let w = weights(g, &mut Noise(seed));
    let feeds = vec![
        Tensor::u32(vec![4], vec![2, 0, 0, 0]),
        Tensor::u32(vec![4], vec![2, 3, 4, 5]),
        Tensor::u32(vec![4], vec![0; 4]),
        Tensor::u32(vec![4], vec![5; 4]),
        Tensor::f32(vec![2, tap_w], vec![0.1; 2 * tap_w]),
        Tensor::u32(vec![6], vec![0, 1, 2, 3, 4, 5]),
        Tensor::u32(vec![6], vec![0; 6]),
    ];
    let (want, got) = check_sharded_head(g, &sharded, &w, &feeds);
    for (a, b) in got[1]
        .floats()
        .unwrap()
        .iter()
        .zip(want[1].floats().unwrap())
    {
        assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{a} vs {b}");
    }
    let wants = match selector {
        true => Wants::roles(&sharded, 0, &[Role::Candidates, Role::Taps]),
        false => Wants::role(&sharded, 0, Role::Tokens),
    };
    let gathers = plan_gathers(&sharded, &parts, 4, wants);
    assert!(
        gathers.contains(&(2 * 3 * 2 * k)),
        "missing top-k gather: {gathers:?}"
    );
    assert!(
        !gathers.contains(&(3 * vocab)),
        "full logits gather remained: {gathers:?}"
    );
    let logits_gathers = plan_gathers(&sharded, &parts, 4, Wants::role(&sharded, 0, Role::Logits));
    assert!(logits_gathers.contains(&(3 * vocab)));
}

#[test]
fn tp_drafter_matches_one() {
    let mut t_cfg = Config::toy(1, 24);
    t_cfg.tied = false;
    t_cfg.stack.taps = vec![1, 2];
    let t_graph = stepper::<Norm, Swiglu>(&t_cfg);
    let spec = target_spec(
        t_cfg.stack.vocab,
        t_cfg.stack.hidden,
        t_cfg.stack.layers,
        false,
    );

    let d1_g = dflash::stepper::<Norm>(&dflash::Config::toy(&spec, 4, 0, 1, 24));
    verify_local_topk(&d1_g, 67, 1, false);
    verify_tp_pair(&t_graph, 61, &d1_g, 67, false, true);

    let d2_g = dflash::stepper::<Norm>(&dflash2_cfg(&spec, 24));
    verify_local_topk(&d2_g, 71, 4, true);
    verify_tp_pair(&t_graph, 61, &d2_g, 71, false, true);

    let d3_g = dflash::stepper::<Norm>(&dspine_cfg(&spec, 24));
    verify_tp_pair(&t_graph, 61, &d3_g, 73, true, false);
}

struct Counting(Sockets, Arc<AtomicUsize>);

impl Transport for Counting {
    fn rank(&self) -> u32 {
        self.0.rank()
    }

    fn ranks(&self) -> u32 {
        self.0.ranks()
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        self.1.fetch_add(1, Ordering::Relaxed);
        self.0.exchange(mine)
    }
}

#[test]
fn tp_early_stop() {
    let mut t_cfg = Config::toy(1, 24);
    t_cfg.tied = false;
    t_cfg.stack.taps = vec![1, 2];
    let t_graph = stepper::<Norm, Swiglu>(&t_cfg);
    let spec = target_spec(
        t_cfg.stack.vocab,
        t_cfg.stack.hidden,
        t_cfg.stack.layers,
        false,
    );
    let d_graph = dflash::stepper::<Norm>(&dflash::Config::toy(&spec, 4, 0, 1, 24));
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let exchanges = Arc::new(AtomicUsize::new(0));
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(
            (&tg_f, 61),
            (&dg_f, 67),
            24,
            false,
            &s_f,
            (Box::new(s1), c1),
        );
        Speculate::new(&mut te, &mut dr).unwrap().follow().unwrap();
    });
    let (mut target, mut drafter) = open_tp_dflash(
        (&t_graph, 61),
        (&d_graph, 67),
        24,
        false,
        &settings,
        (Box::new(s0), Counting(c0, exchanges.clone())),
    );
    let mut spec = Speculate::new(&mut target, &mut drafter).unwrap();
    // Normal multi-step request uses only request-start and end sync.
    let (full, stats) = spec_events(|e| spec.generate(req(&[2, 6, 1], 7), e));
    assert!(stats.steps >= 2);
    assert_eq!(exchanges.load(Ordering::Relaxed), 2);
    // Early emit stop avoids hangs and allows later requests.
    let mut cut = Vec::new();
    let cut_stats = spec
        .generate(req(&[2, 6, 1], 7), &mut |e| {
            cut.push(e);
            collect_ids(&cut).len() < 2
        })
        .unwrap();
    assert_eq!(cut_stats.emitted, 2);
    assert_eq!(collect_ids(&cut), collect_ids(&full)[..2]);
    assert_eq!(cut.last(), Some(&Event::Done(Reason::Stop)));
    let (after, _) = spec_events(|e| spec.generate(req(&[4, 1, 5], 5), e));
    assert_eq!(collect_ids(&after).len(), 5);
    target.dismiss().unwrap();
    follower.join().unwrap();
}
