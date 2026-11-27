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
            let mut te = open_engine(&t_graph, &tw, 32, &paged_recipe(paged), &settings);
            let mut dr = open_drafter(&d_graph, &dw, 32, 4, 0, &settings);
            assert_eq!(
                run_batch_reqs(&mut te, &mut dr, prompts),
                want,
                "paged={paged}"
            );
        }
    }
}

#[test]
fn batch_dflash_tp() {
    let (t_graph, d_graph) = toy_pair(24);
    let settings = host_settings();
    let tw = weights(&t_graph, &mut Noise(97));
    let dw = weights(&d_graph, &mut Noise(101));
    let prompts: &[(&[u32], usize)] = &[(&[2, 5, 1, 4], 6), (&[6, 3, 2], 5), (&[1, 7, 3, 5, 2], 4)];
    let want = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 24, &settings, prompts);
    let got = with_tp_batch(
        (&t_graph, &d_graph),
        24,
        (97, 101),
        Some(3),
        false,
        |batch| {
            let mut rxs = Vec::new();
            for &(p, max) in prompts {
                let (tx, rx) = std::sync::mpsc::channel();
                batch.submit(req(p, max), Box::new(move |e| tx.send(e).is_ok()));
                rxs.push(rx);
            }
            while batch.step().unwrap() {}
            rxs.into_iter()
                .map(|r| collect_ids(&r.try_iter().collect::<Vec<_>>()))
                .collect::<Vec<_>>()
        },
    );
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
    let settings = host_settings();
    let want = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 48, &settings, prompts);
    for paged in [false, true] {
        let mut te = open_engine(&t_graph, &tw, 48, &paged_recipe(paged), &settings);
        let mut dr = open_drafter(&d_graph, &dw, 48, 4, 0, &settings);
        let mut batch = BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .with_prefill_chunk(6);
        batch.reuse();
        let (got, kept) = run_reuse_seq(&mut batch, prompts);
        assert_eq!(got, want, "paged={paged}");
        assert_eq!(kept, vec![0, if paged { 16 } else { 20 }], "paged={paged}");
    }
    let (got, kept) = with_tp_batch(
        (&t_graph, &d_graph),
        48,
        (103, 107),
        Some(6),
        true,
        |batch| run_reuse_seq(batch, prompts),
    );
    assert_eq!(got, want);
    assert_eq!(kept, vec![0, 20]);
}

#[test]
fn dflash_sampling_seeded() {
    let (t_graph, d_graph) = toy_pair(24);
    let tw = weights(&t_graph, &mut Noise(113));
    let dw = weights(&d_graph, &mut Noise(127));
    let settings = host_settings();
    let prompt = [2u32, 5, 1, 4, 3];
    let greedy = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 24, &settings, &[(&prompt, 8)])
        .pop()
        .unwrap();
    let run_solo = |sampling| {
        let mut te = open_target(&t_graph, &tw, 24, &settings);
        let mut dr = open_drafter(&d_graph, &dw, 24, 4, 0, &settings);
        let r = Request {
            sampling,
            ..req(&prompt, 8)
        };
        collect_ids(&spec_events(|e| Speculate::new(&mut te, &mut dr).unwrap().generate(r, e)).0)
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
    assert_eq!(a, run_solo(sampled));
    assert_eq!(a.len(), 8);
    // BatchDFlash with 2-rank TP agrees with single-rank sampled output.
    let got_tp = with_tp_batch((&t_graph, &d_graph), 24, (113, 127), None, false, |batch| {
        let (tx, rx) = std::sync::mpsc::channel();
        batch.submit(
            Request {
                sampling: sampled,
                ..req(&prompt, 8)
            },
            Box::new(move |e| tx.send(e).is_ok()),
        );
        while batch.step().unwrap() {}
        collect_ids(&rx.try_iter().collect::<Vec<_>>())
    });
    assert_eq!(got_tp, a);
}

#[test]
fn batch_dflash_rolls_window() {
    let (t_graph, _) = toy_pair(48);
    let (_, d_graph) = toy_pair(16);
    let tw = weights(&t_graph, &mut Noise(109));
    let dw = weights(&d_graph, &mut Noise(113));
    let p1: Vec<u32> = (0..22).map(|i| (i % 7) + 1).collect();
    let p2: Vec<u32> = (0..18).map(|i| ((i * 3) % 7) + 1).collect();
    let prompts: &[(&[u32], usize)] = &[(&p1, 14), (&p2, 12)];
    let settings = host_settings();
    let (_, d_full) = toy_pair(48);
    let want = run_solo_refs(&t_graph, &d_full, &tw, &dw, 48, &settings, prompts);
    let mut te = open_target(&t_graph, &tw, 48, &settings);
    let mut dr = open_drafter(&d_graph, &dw, 16, 4, 0, &settings);
    assert_eq!(run_batch_reqs(&mut te, &mut dr, prompts), want);
}

#[test]
fn batch_dflash_packs_prefill() {
    let (t_graph, d_graph) = toy_pair(32);
    let tw = weights(&t_graph, &mut Noise(131));
    let dw = weights(&d_graph, &mut Noise(137));
    let settings = host_settings();
    let p_long: Vec<u32> = (0..18).map(|i| (i % 7) + 1).collect();
    let p_short: Vec<u32> = vec![3, 5, 2];
    let p_mid: Vec<u32> = vec![4, 1, 6, 2, 7];

    let step_once = |items: &[(&[u32], usize)]| -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut te = open_target(&t_graph, &tw, 32, &settings);
        let mut dr = open_drafter(&d_graph, &dw, 32, 4, 0, &settings);
        let mut batch = BatchDFlash::new(&mut te, &mut dr)
            .unwrap()
            .with_prefill_chunk(10);
        let mut rxs = Vec::new();
        for &(p, max) in items {
            let (tx, rx) = std::sync::mpsc::channel();
            batch.submit(req(p, max), Box::new(move |e| tx.send(e).is_ok()));
            rxs.push(rx);
        }
        assert!(batch.step().unwrap());
        let first: Vec<Vec<u32>> = rxs
            .iter()
            .map(|r| collect_ids(&r.try_iter().collect::<Vec<_>>()))
            .collect();
        let mut full = first.clone();
        while batch.step().unwrap() {}
        for (i, rx) in rxs.into_iter().enumerate() {
            full[i].extend(collect_ids(&rx.try_iter().collect::<Vec<_>>()));
        }
        (first, full)
    };

    // Two short prompts pack into one prefill step.
    let pair: &[(&[u32], usize)] = &[(&p_mid, 4), (&p_short, 4)];
    let want_pair = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 32, &settings, pair);
    let (first, got) = step_once(pair);
    assert!(!first[0].is_empty() && !first[1].is_empty());
    assert_eq!(got, want_pair);

    // Short prompt finishes prefill ahead of a queued long prompt.
    let mixed: &[(&[u32], usize)] = &[(&p_long, 4), (&p_short, 4)];
    let want_mixed = run_solo_refs(&t_graph, &d_graph, &tw, &dw, 32, &settings, mixed);
    let (first, got) = step_once(mixed);
    assert!(first[0].is_empty() && !first[1].is_empty());
    assert_eq!(got, want_mixed);

    // Sampled multi-lane prefill packs cleanly and matches 2-rank TP.
    let sampled = tessera_engine::Sampling {
        temperature: 1.2,
        top_p: 0.9,
        seed: Some(99),
        ..Default::default()
    };
    let run_sampled = |batch: &mut BatchDFlash<'_>| {
        let mut chans = Vec::new();
        for &(p, max) in pair {
            let (tx, rx) = std::sync::mpsc::channel();
            let r = Request {
                sampling: sampled,
                ..req(p, max)
            };
            batch.submit(r, Box::new(move |e| tx.send(e).is_ok()));
            chans.push(rx);
        }
        while batch.step().unwrap() {}
        chans
            .into_iter()
            .map(|r| collect_ids(&r.try_iter().collect::<Vec<_>>()))
            .collect::<Vec<_>>()
    };
    let mut te = open_target(&t_graph, &tw, 32, &settings);
    let mut dr = open_drafter(&d_graph, &dw, 32, 4, 0, &settings);
    let mut solo_batch = BatchDFlash::new(&mut te, &mut dr)
        .unwrap()
        .with_prefill_chunk(10);
    let want_sampled = run_sampled(&mut solo_batch);
    let got_tp = with_tp_batch(
        (&t_graph, &d_graph),
        32,
        (131, 137),
        Some(10),
        false,
        run_sampled,
    );
    assert_eq!(got_tp, want_sampled);
}

#[test]
fn taps_before_final_layer() {
    let mut cfg = Config::toy(2, 32);
    (cfg.tied, cfg.stack.layers, cfg.stack.taps) = (false, 4, vec![1, 2]);
    let spec = target_spec(cfg.stack.vocab, cfg.stack.hidden, 4, false);
    let mut d_cfg = dflash::Config::toy(&spec, 4, 0, 2, 32);
    d_cfg.target_layers = vec![0, 1];
    let (t_graph, d_graph) = (
        stepper::<Norm, Swiglu>(&cfg),
        dflash::stepper::<Norm>(&d_cfg),
    );
    let (tw, dw) = (
        weights(&t_graph, &mut Noise(149)),
        weights(&d_graph, &mut Noise(151)),
    );
    let prompt: Vec<u32> = (0..12).map(|i| (i % 7) + 1).collect();
    let s = host_settings();
    let want = gen_ids(&mut open_target(&t_graph, &tw, 32, &s), req(&prompt, 8));
    // Chunked prefill must keep untapped tail layers.
    assert_eq!(
        run_solo_refs(&t_graph, &d_graph, &tw, &dw, 32, &s, &[(&prompt, 8)])[0],
        want
    );
    let mut te = open_target(&t_graph, &tw, 32, &s);
    let mut dr = open_drafter(&d_graph, &dw, 32, 4, 0, &s);
    assert_eq!(run_batch_reqs(&mut te, &mut dr, &[(&prompt, 8)])[0], want);
}
