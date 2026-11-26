//! Multi-slot DFlash batching, prefix reuse, and sampling tests.

use super::*;

#[test]
fn batch_dflash_matches_single() {
    let (t_graph, d_graph) = toy_pair(32);
    let tw = weights(&t_graph, &mut Noise(83));
    let dw = weights(&d_graph, &mut Noise(89));
    let prompts: &[(&[u32], usize)] = &[
        (&[2, 5, 1, 7, 3], 6),
        (&[4, 2, 6], 5),
        (&[1, 3, 5, 2, 4, 6, 7], 4),
        (&[6, 1], 7),
    ];
    for settings in runs() {
        let want = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 32, &settings, prompts);
        for paged in [false, true] {
            let mut recipe = Recipe::default();
            if paged {
                recipe.kept = &tessera_lower::kept::PAGED16;
            }
            let mut te = Engine::open(t_graph.clone(), 32, &recipe, &settings).unwrap();
            let stored: Vec<Weight> = tw.iter().cloned().map(Weight::Floats).collect();
            te.load(&stored).unwrap();
            let mut dr = open_drafter(&d_graph, &dw, 32, 4, 0, &settings);
            let got = run_batch_reqs(&mut te, &mut dr, prompts);
            assert_eq!(got, want, "paged={paged}");
        }
    }
}

#[test]
fn batch_dflash_tp() {
    let (t_graph, d_graph) = toy_pair(24);
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let tw = weights(&t_graph, &mut Noise(97));
    let dw = weights(&d_graph, &mut Noise(101));
    let prompts: &[(&[u32], usize)] = &[(&[2, 5, 1, 4], 6), (&[6, 3, 2], 5), (&[1, 7, 3, 5, 2], 4)];
    let want = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 24, &settings, prompts);
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(&tg_f, &dg_f, 24, &s_f, Box::new(s1), c1, (97, 101));
        BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .with_prefill_chunk(3)
            .follow()
            .unwrap();
    });
    let (mut te, mut dr) = open_tp_dflash(
        &t_graph,
        &d_graph,
        24,
        &settings,
        Box::new(s0),
        c0,
        (97, 101),
    );
    let got = run_batch_reqs(&mut te, &mut dr, prompts);
    te.dismiss().unwrap();
    follower.join().unwrap();
    assert_eq!(got, want);
}

#[test]
fn batch_dflash_reuses_prefix() {
    let (t_graph, d_graph) = toy_pair(48);
    let tw = weights(&t_graph, &mut Noise(103));
    let dw = weights(&d_graph, &mut Noise(107));
    let p1: Vec<u32> = (0..20).map(|i| (i % 7) + 1).collect();
    let mut p2 = p1.clone();
    p2.extend_from_slice(&[6, 2, 5, 1]);
    let prompts: &[(&[u32], usize)] = &[(&p1, 6), (&p2, 6)];
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let want = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 48, &settings, prompts);
    for paged in [false, true] {
        let mut recipe = Recipe::default();
        if paged {
            recipe.kept = &tessera_lower::kept::PAGED16;
        }
        let mut te = Engine::open(t_graph.clone(), 48, &recipe, &settings).unwrap();
        let stored: Vec<Weight> = tw.iter().cloned().map(Weight::Floats).collect();
        te.load(&stored).unwrap();
        let mut dr = open_drafter(&d_graph, &dw, 48, 4, 0, &settings);
        let mut batch = BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .with_prefill_chunk(6);
        batch.reuse();
        let (got, kept) = run_reuse_seq(&mut batch, prompts);
        assert_eq!(got, want, "paged={paged}");
        assert_eq!(kept, vec![0, if paged { 16 } else { 20 }], "paged={paged}");
    }
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(&tg_f, &dg_f, 48, &s_f, Box::new(s1), c1, (103, 107));
        let mut batch = BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .with_prefill_chunk(6);
        batch.reuse();
        batch.follow().unwrap();
    });
    let (mut te, mut dr) = open_tp_dflash(
        &t_graph,
        &d_graph,
        48,
        &settings,
        Box::new(s0),
        c0,
        (103, 107),
    );
    let mut batch = BatchDFlash::new(&mut te, &mut dr)
        .unwrap()
        .with_prefill_chunk(6);
    batch.reuse();
    let (got, kept) = run_reuse_seq(&mut batch, prompts);
    te.dismiss().unwrap();
    follower.join().unwrap();
    assert_eq!(got, want);
    assert_eq!(kept, vec![0, 20]);
}

#[test]
fn dflash_sampling_seeded() {
    let (t_graph, d_graph) = toy_pair(24);
    let tw = weights(&t_graph, &mut Noise(113));
    let dw = weights(&d_graph, &mut Noise(127));
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let prompt = vec![2u32, 5, 1, 4, 3];
    let greedy = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 24, &settings, &[(&prompt, 8)])
        .pop()
        .unwrap();
    let run_solo = |sampling| {
        let mut te = open_target(&t_graph, &tw, 24, &settings);
        let mut dr = open_drafter(&d_graph, &dw, 24, 4, 0, &settings);
        let mut evs = Vec::new();
        Speculate::new(&mut te, &mut dr)
            .unwrap()
            .generate(
                Request {
                    tokens: prompt.clone(),
                    max_tokens: 8,
                    sampling,
                    ..Request::default()
                },
                &mut |e| {
                    evs.push(e);
                    true
                },
            )
            .unwrap();
        collect_ids(&evs)
    };
    let top1 = tessera_engine::Sampling {
        temperature: 0.9,
        top_k: 1,
        seed: Some(7),
        ..Default::default()
    };
    assert_eq!(run_solo(top1), greedy);
    let sampled = tessera_engine::Sampling {
        temperature: 1.5,
        top_p: 0.95,
        seed: Some(42),
        ..Default::default()
    };
    let a = run_solo(sampled);
    let b = run_solo(sampled);
    assert_eq!(a, b);
    assert_eq!(a.len(), 8);
    // BatchDFlash with 2-rank TP agrees with single-rank sampled output.
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(&tg_f, &dg_f, 24, &s_f, Box::new(s1), c1, (113, 127));
        BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .follow()
            .unwrap();
    });
    let (mut te, mut dr) = open_tp_dflash(
        &t_graph,
        &d_graph,
        24,
        &settings,
        Box::new(s0),
        c0,
        (113, 127),
    );
    let mut batch = BatchDFlash::new(&mut te, &mut dr).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    batch.submit(
        Request {
            tokens: prompt,
            max_tokens: 8,
            sampling: sampled,
            ..Request::default()
        },
        Box::new(move |e| tx.send(e).is_ok()),
    );
    while batch.step().unwrap() {}
    te.dismiss().unwrap();
    follower.join().unwrap();
    let got_tp = collect_ids(&rx.try_iter().collect::<Vec<_>>());
    assert_eq!(got_tp, a);
}
