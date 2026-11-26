//! End-to-end parity between qwen4_exp forward and ragged stepper.

mod common;

use common::{
    collect_ids, float_weights, gen_ids, host_settings, loopback_links, open_drafter,
    open_ranked_target, open_target, req, row, spec_events, two_place,
};
use std::io::Write;
use tessera_engine::comm::Sockets;
use tessera_engine::qwen4_exp::{Config, Moe, program, stepper};
use tessera_engine::{
    Batch, Bf16, Engine, Event, Fair, Penalties, PleConfig, PleStore, Request, Row, Settings,
    Speculate, Steps, Tensor,
};
use tessera_models::dflash::{self, TargetSpec};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

fn make_ple_table() -> (std::path::PathBuf, PleConfig) {
    let dir = tessera_testkit::scratch(&format!("{}-ple-test", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ple.bin");
    let rows = 128usize;
    let row_dim = 1usize;
    let mut bytes = Vec::with_capacity(rows * row_dim * 2);
    for r in 0..rows {
        let v = ((r as f32) * 0.037).sin() * 0.2;
        bytes.extend_from_slice(&Bf16::from(v).0.to_le_bytes());
    }
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&bytes).unwrap();
    let cfg = PleConfig {
        path: path.clone(),
        eos_token: 15,
        multipliers: vec![100003, 200003, 300007],
        offsets: (0..16).map(|h| h * 8).collect(),
        vocab_sizes: vec![8; 16],
        row_dim,
        cache_rows: 64,
    };
    (path, cfg)
}

fn push_rows(out: &mut Vec<Row>, slot: u32, seq: &[u32], range: std::ops::Range<usize>) {
    for (t, &token) in seq[range.clone()].iter().enumerate() {
        out.push(row(slot, token, (range.start + t) as u32));
    }
}

#[test]
fn step_matches_forward() {
    let c = Config::tiny();
    let fwd = program::<Moe>(&c);
    let step = stepper::<Moe>(&c);
    let w = weights(&fwd, &mut Noise(7));
    let (ple_path, ple_cfg) = make_ple_table();

    let seq0 = [1u32, 3, 5, 2, 7, 4, 9, 6, 15, 2, 8];
    let seq1 = [4u32, 0, 11, 13, 2, 1, 6, 3, 5, 7, 10];
    let ctx = seq0.len();

    // Build exact PLE tensor for forward pass via PleStore.
    let mut ref_ple = PleStore::open(ple_cfg.clone()).unwrap();
    let mut all_rows = Vec::new();
    for (s, seq) in [seq0, seq1].iter().enumerate() {
        push_rows(&mut all_rows, s as u32, seq, 0..ctx);
    }
    let ple_flat = ref_ple.fill(&all_rows, all_rows.len()).unwrap();
    let ple_fwd = Tensor::f32(vec![2, ctx, 16], ple_flat.floats().unwrap().to_vec());

    let mut toks = Vec::new();
    toks.extend_from_slice(&seq0);
    toks.extend_from_slice(&seq1);
    let mut pos = Vec::new();
    for _ in 0..2 {
        pos.extend(0..ctx as u32);
    }
    let fwd_out = tessera_interp::run(
        &fwd,
        &w,
        0,
        &[
            Tensor::u32(vec![2, ctx], toks),
            Tensor::u32(vec![2, ctx], pos),
            ple_fwd,
        ],
    )
    .unwrap();
    let ref_logits = fwd_out[0].floats().unwrap();
    let vocab = 16usize;
    let argmax = |s: usize, t: usize| -> u32 {
        let row = &ref_logits[(s * ctx + t) * vocab..(s * ctx + t + 1) * vocab];
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };

    // Run stepper on host runner with chunked prefill and decode.
    let mut engine = open_target(&step, &w, 16, &host_settings());
    engine.set_ple(PleStore::open(ple_cfg).unwrap());

    // Chunk 1: 5 rows slot 0, 3 rows slot 1.
    let mut chunk1 = Vec::new();
    push_rows(&mut chunk1, 0, &seq0, 0..5);
    push_rows(&mut chunk1, 1, &seq1, 0..3);
    let out1 = engine.step(&chunk1).unwrap();
    for (t, &got) in out1[..5].iter().enumerate() {
        assert_eq!(got, argmax(0, t), "slot 0 chunk1 pos {t}");
    }
    for (t, &got) in out1[5..8].iter().enumerate() {
        assert_eq!(got, argmax(1, t), "slot 1 chunk1 pos {t}");
    }

    // Chunk 2: remaining tokens interleaved (non-power-of-two batch!).
    let mut chunk2 = Vec::new();
    push_rows(&mut chunk2, 0, &seq0, 5..9);
    push_rows(&mut chunk2, 1, &seq1, 3..ctx);
    let out2 = engine.step(&chunk2).unwrap();
    for (i, &got) in out2[..4].iter().enumerate() {
        assert_eq!(got, argmax(0, 5 + i), "slot 0 chunk2 pos {}", 5 + i);
    }
    for (i, &got) in out2[4..].iter().enumerate() {
        assert_eq!(got, argmax(1, 3 + i), "slot 1 chunk2 pos {}", 3 + i);
    }

    // Single-token decode exercising dilated conv and EOS reset.
    for (i, &token) in seq0[9..ctx].iter().enumerate() {
        let t = 9 + i;
        let got = engine.step(&[row(0, token, t as u32)]).unwrap();
        assert_eq!(got[0], argmax(0, t), "slot 0 decode pos {t}");
    }

    // Slot reset at position zero matches reference.
    let mut reset_rows = Vec::new();
    push_rows(&mut reset_rows, 0, &seq0, 0..ctx);
    let got_reset = engine.last(&reset_rows).unwrap();
    assert_eq!(got_reset[0], argmax(0, ctx - 1), "slot 0 reset parity");

    let _ = std::fs::remove_file(ple_path);
}

#[test]
fn ple_slot_reuse() {
    let (_, cfg) = make_ple_table();
    let fresh = [row(0, 5, 0), row(0, 7, 1)];
    let want = PleStore::open(cfg.clone())
        .unwrap()
        .fill(&fresh, 2)
        .unwrap();
    let mut reused = PleStore::open(cfg).unwrap();
    reused.fill(&[row(0, 3, 0)], 1).unwrap();
    let got = reused.fill(&fresh, 2).unwrap();
    assert_eq!(got.floats().unwrap(), want.floats().unwrap());
}

#[test]
fn ple_hash_orders() {
    let (_, base) = make_ple_table();
    let cfg = PleConfig {
        multipliers: vec![3, 5, 7, 11],
        offsets: vec![0, 10, 20, 30, 40, 50],
        vocab_sizes: vec![7, 9, 7, 9, 7, 9],
        ..base.clone()
    };
    let store = PleStore::open(cfg).unwrap();
    // The 4-gram reaches back to the EOS.
    let mut ids = [0u64; 6];
    store.hash_token(&[15, 4, 6, 8], &mut ids);
    let two = (8 * 3) ^ (6 * 5);
    let three = two ^ (4 * 7);
    let four = three ^ (15 * 11);
    let want = [two, two, three, three, four, four].map(|m: i64| m as u64);
    let got: Vec<u64> = (0..6)
        .map(|h| ids[h] - [0, 10, 20, 30, 40, 50][h])
        .collect();
    let sizes = [7, 9, 7, 9, 7, 9];
    assert_eq!(got, (0..6).map(|h| want[h] % sizes[h]).collect::<Vec<_>>());
    let uneven = PleConfig {
        offsets: vec![0; 5],
        vocab_sizes: vec![1; 5],
        ..base
    };
    assert!(PleStore::open(uneven).is_err());
}

/// A tiny model rank with its own PLE store.
fn ple_engine(links: Option<[Sockets; 2]>) -> Engine {
    let g = stepper::<Moe>(&Config::tiny());
    let mut engine = match links {
        Some([sums, control]) => {
            let weights = float_weights(&g, 3);
            open_ranked_target(
                g,
                16,
                &host_settings(),
                two_place(false),
                (Box::new(sums), control),
                &weights,
            )
        }
        None => open_target(&g, &weights(&g, &mut Noise(3)), 16, &host_settings()),
    };
    engine.set_ple(PleStore::open(make_ple_table().1).unwrap());
    engine
}

/// Two ranks with PLE step as one engine does.
#[test]
fn ranks_feed_ple() {
    let [[sums, others], [control, told]] = loopback_links();
    let follower = std::thread::spawn(move || ple_engine(Some([others, told])).follow().unwrap());
    let mut lead = ple_engine(Some([sums, control]));
    let mut alone = ple_engine(None);
    let mut rows = Vec::new();
    push_rows(&mut rows, 0, &[1, 3, 5, 2], 0..4);
    push_rows(&mut rows, 1, &[4, 15, 11], 0..3);
    for rows in [rows, vec![row(0, 7, 4)]] {
        assert_eq!(lead.step(&rows).unwrap(), alone.step(&rows).unwrap());
    }
    assert!(lead.kernels().contains_key("all_reduce"));
    lead.dismiss().unwrap();
    follower.join().unwrap();
}

/// The events of one answer.
fn answer(batch: &mut Batch, tokens: Vec<u32>) -> Vec<Event> {
    let (events, taken) = std::sync::mpsc::channel();
    batch.submit(req(&tokens, 3), Box::new(move |e| events.send(e).is_ok()));
    while batch.step().unwrap() {}
    taken.try_iter().collect()
}

/// Stateful prompts never resume from kept rows.
#[test]
fn stateful_skips_no_prefix() {
    let mut engine = ple_engine(None);
    assert!(engine.stateful());
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    batch.reuse();
    let base = vec![1u32, 3, 5, 2, 7];
    let first = collect_ids(&answer(&mut batch, base.clone()));
    let longer = [&base[..], &first[..]].concat();
    let events = answer(&mut batch, longer.clone());
    assert!(!events.iter().any(|e| matches!(e, Event::Kept(_))));
    assert_eq!(batch.totals().unwrap().hits, 0);
    let mut fresh = ple_engine(None);
    let want = answer(&mut Batch::new(&mut fresh, Box::new(Fair)), longer);
    assert_eq!(collect_ids(&events), collect_ids(&want));
}

fn bits(floats: &[f32]) -> Vec<u32> {
    floats.iter().map(|x| x.to_bits()).collect()
}

/// A tapped tiny model with its own PLE store.
fn tapped_engine(g: &tessera_engine::Graph, settings: &Settings) -> Engine {
    let mut engine = open_target(g, &weights(g, &mut Noise(3)), 16, settings);
    engine.set_ple(PleStore::open(make_ple_table().1).unwrap());
    engine
}

/// A verify then a commit store what plain steps store.
#[test]
fn commit_matches_steps() {
    let mut c = Config::tiny();
    c.stack.taps = vec![1, 2];
    let g = stepper::<Moe>(&c);
    let (mut prompt, mut drafts) = (Vec::new(), Vec::new());
    let seq = [1u32, 3, 5, 2, 7, 4, 9, 6];
    push_rows(&mut prompt, 0, &seq, 0..4);
    push_rows(&mut drafts, 0, &seq, 4..8);
    for settings in runs() {
        for kept in 1..=drafts.len() {
            let mut spec = tapped_engine(&g, &settings);
            let mut plain = tapped_engine(&g, &settings);
            spec.step_taps(&prompt).unwrap();
            plain.step_taps(&prompt).unwrap();
            let before = spec.recurrent_state().unwrap();
            let (ids, taps) = spec.verify(&drafts).unwrap();
            assert_eq!(bits(&spec.recurrent_state().unwrap()), bits(&before));
            spec.commit(kept).unwrap();
            let (want_ids, want_taps) = plain.step_taps(&drafts[..kept]).unwrap();
            // A commit with no verify before it stores nothing.
            let stored = bits(&plain.recurrent_state().unwrap());
            spec.commit(kept).unwrap();
            plain.commit(kept).unwrap();
            assert_eq!(bits(&plain.recurrent_state().unwrap()), stored);
            assert_eq!(ids[..kept], want_ids);
            assert_eq!(bits(&taps[..want_taps.len()]), bits(&want_taps));
            let (got, want) = (spec.recurrent_state(), plain.recurrent_state());
            assert_eq!(bits(&got.unwrap()), bits(&want.unwrap()));
        }
    }
}

/// Guided or not, stateful DFlash decodes as greedy does.
#[test]
fn stateful_dflash_greedy() {
    let mut c = Config::tiny();
    c.stack.steps = Steps::Masked;
    let spec = TargetSpec {
        vocab: c.stack.vocab,
        hidden: c.stack.hidden,
        layers: c.stack.layers,
        tied: c.tied,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 1, 16);
    c.stack.taps = d_cfg.tap_layers();
    let g = stepper::<Moe>(&c);
    let d_graph = dflash::stepper::<()>(&d_cfg);
    let penalties = Penalties {
        presence: 2.0,
        ..Penalties::default()
    };
    let request = |guided: bool| Request {
        tokens: vec![1, 3, 5, 2],
        max_tokens: 10,
        guide: penalties.guide(&[1, 3, 5, 2]).filter(|_| guided),
        ..Request::default()
    };
    for settings in runs() {
        for guided in [false, true] {
            let mut baseline = tapped_engine(&g, &settings);
            let mut target = tapped_engine(&g, &settings);
            let dw = weights(&d_graph, &mut Noise(23));
            let mut drafter = open_drafter(&d_graph, &dw, 16, 4, 0, &settings);
            let want = gen_ids(&mut baseline, request(guided));
            let mut run = Speculate::new(&mut target, &mut drafter).unwrap();
            let (got, stats) = spec_events(|emit| run.generate(request(guided), emit));
            assert_eq!(collect_ids(&got), want);
            assert!(stats.steps > 0);
        }
    }
}
