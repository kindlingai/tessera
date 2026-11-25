//! End-to-end parity between qwen4_exp forward and ragged stepper.

mod common;

use common::{collect_ids, float_weights, loopback_links, open_ranked_target, open_target};
use std::io::Write;
use tessera_engine::Place;
use tessera_engine::comm::Sockets;
use tessera_engine::qwen4_exp::{Config, Moe, Ple, RotatedMoe, program, stepper};
use tessera_engine::qwen35::{Mixing, Recurrent};
use tessera_engine::{
    Batch, Bf16, Drafter, Engine, Event, Fair, Pick, PleConfig, PleStore, Recipe, Request, Row,
    Settings, Speculate, Stack, Tensor, Weight,
};
use tessera_graph::Parts;
use tessera_lower::Cluster;
use tessera_models::dflash::{self, TargetSpec};
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

fn tiny_config() -> Config {
    Config {
        stack: Stack::sized([16, 16, 2, 2, 8, 8], 2, 16),
        kv_heads: 1,
        rotary: 4,
        rope_theta: 1e4,
        eps: 1e-6,
        hc_count: 4,
        hc_rank: 8,
        experts: 4,
        top_k: 2,
        shared_ffn: 8,
        linear: Recurrent {
            key_heads: 2,
            value_heads: 4,
            key_dim: 4,
            value_dim: 4,
            window: 4,
        },
        layers: vec![Mixing::Linear, Mixing::Full],
        ple: Some(Ple {
            layer: 1,
            dim: 16,
            window: 4,
            dilation: 3,
            heads: 16,
        }),
        eos_token: 15,
        tied: false,
    }
}

fn fill_weights(g: &tessera_engine::Graph) -> Vec<Tensor> {
    let mut seed = 0x1234_5678u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((seed >> 33) as f32) / (u32::MAX as f32 * 0.5) - 1.0;
        u * 0.15
    };
    g.weights
        .iter()
        .map(|w| {
            let shape: Vec<usize> =
                w.ty.shape
                    .0
                    .iter()
                    .map(|e| e.value(&g.dims, &|_| None).unwrap() as usize)
                    .collect();
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|_| next()).collect();
            Tensor::f32(shape, data)
        })
        .collect()
}

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
        out.push(Row {
            slot,
            token,
            position: (range.start + t) as u32,
        });
    }
}

#[test]
fn step_matches_forward() {
    let c = tiny_config();
    let fwd = program::<Moe>(&c);
    let step = stepper::<Moe>(&c);
    let weights = fill_weights(&fwd);
    let (ple_path, ple_cfg) = make_ple_table();

    let seq0 = [1u32, 3, 5, 2, 7, 4, 9, 6, 15, 2, 8];
    let seq1 = [4u32, 0, 11, 13, 2, 1, 6, 3, 5, 7, 10];
    let ctx = seq0.len();

    // Build exact PLE tensor for forward pass via PleStore.
    let mut ref_ple = PleStore::open(ple_cfg.clone()).unwrap();
    let mut all_rows = Vec::new();
    for (s, seq) in [seq0, seq1].iter().enumerate() {
        for (p, &tok) in seq.iter().enumerate() {
            all_rows.push(Row {
                slot: s as u32,
                token: tok,
                position: p as u32,
            });
        }
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
        &weights,
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
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let mut engine = Engine::open(step, 16, &Recipe::default(), &settings).unwrap();
    engine.set_ple(PleStore::open(ple_cfg).unwrap());
    let engine_w: Vec<Weight> = weights.into_iter().map(Weight::Floats).collect();
    engine.load(&engine_w).unwrap();

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
        let got = engine
            .step(&[Row {
                slot: 0,
                token,
                position: t as u32,
            }])
            .unwrap();
        assert_eq!(got[0], argmax(0, t), "slot 0 decode pos {t}");
    }

    // Slot reset at position zero matches reference.
    let reset_rows: Vec<Row> = seq0
        .iter()
        .enumerate()
        .map(|(t, &tok)| Row {
            slot: 0,
            token: tok,
            position: t as u32,
        })
        .collect();
    let got_reset = engine.last(&reset_rows).unwrap();
    assert_eq!(got_reset[0], argmax(0, ctx - 1), "slot 0 reset parity");

    let _ = std::fs::remove_file(ple_path);
}

#[test]
fn ple_slot_reuse() {
    let (_, cfg) = make_ple_table();
    let row = |token, position| Row {
        slot: 0,
        token,
        position,
    };
    let fresh = [row(5, 0), row(7, 1)];
    let want = PleStore::open(cfg.clone())
        .unwrap()
        .fill(&fresh, 2)
        .unwrap();
    let mut reused = PleStore::open(cfg).unwrap();
    reused.fill(&[row(3, 0)], 1).unwrap();
    let got = reused.fill(&fresh, 2).unwrap();
    assert_eq!(got.floats().unwrap(), want.floats().unwrap());
}

#[test]
fn rotated_trims() {
    let tiny = tiny_config();
    let c = Config {
        stack: Stack::sized([256, 128, 2, 2, 64, 128], 2, 16),
        rotary: 16,
        hc_rank: 32,
        shared_ffn: 128,
        linear: Recurrent {
            key_dim: 32,
            value_dim: 32,
            ..tiny.linear
        },
        ple: tiny.ple.map(|p| Ple { dim: 128, ..p }),
        ..tiny
    };
    let g = stepper::<RotatedMoe>(&c);
    let trimmed = tessera_graph::trim(&g, 0).unwrap();
    assert!(!trimmed.nodes.is_empty());
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

fn host() -> Settings {
    Settings {
        backend: Pick::Host,
        ..Settings::default()
    }
}

/// A tiny model rank with its own PLE store.
fn ple_engine(links: Option<[Sockets; 2]>) -> Engine {
    let g = stepper::<Moe>(&tiny_config());
    let mut engine = match links {
        Some([sums, control]) => {
            let place = Place::new(Cluster::two("a", "b"), Parts::even(2));
            let weights = float_weights(&g, 3);
            open_ranked_target(g, 16, &host(), place, Box::new(sums), control, &weights)
        }
        None => {
            let weights = tessera_testkit::weights(&g, &mut tessera_testkit::Noise(3));
            open_target(&g, &weights, 16, &host())
        }
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
    let next = vec![Row {
        slot: 0,
        token: 7,
        position: 4,
    }];
    for rows in [rows, next] {
        assert_eq!(lead.step(&rows).unwrap(), alone.step(&rows).unwrap());
    }
    assert!(lead.kernels().contains_key("all_reduce"));
    lead.dismiss().unwrap();
    follower.join().unwrap();
}

/// The events of one answer.
fn answer(batch: &mut Batch, tokens: Vec<u32>) -> Vec<Event> {
    let (events, taken) = std::sync::mpsc::channel();
    let request = Request {
        tokens,
        max_tokens: 3,
        ..Request::default()
    };
    batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
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
    let mut c = tiny_config();
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
            assert_eq!(ids[..kept], want_ids);
            assert_eq!(bits(&taps[..want_taps.len()]), bits(&want_taps));
            let (got, want) = (spec.recurrent_state(), plain.recurrent_state());
            assert_eq!(bits(&got.unwrap()), bits(&want.unwrap()));
        }
    }
}

/// DFlash over recurrent state decodes as greedy does.
#[test]
fn stateful_dflash_greedy() {
    let mut c = tiny_config();
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
    let request = || Request {
        tokens: vec![1, 3, 5, 2],
        max_tokens: 10,
        ..Request::default()
    };
    for settings in runs() {
        let mut baseline = tapped_engine(&g, &settings);
        let mut target = tapped_engine(&g, &settings);
        let recipe = Recipe::default();
        let mut drafter = Drafter::open(d_graph.clone(), 16, 4, 0, &recipe, &settings).unwrap();
        drafter.load(&float_weights(&d_graph, 23)).unwrap();
        let (mut want, mut got) = (Vec::new(), Vec::new());
        baseline.generate(request(), &mut |e| want.push(e)).unwrap();
        let mut run = Speculate::new(&mut target, &mut drafter).unwrap();
        let emit = &mut |e| {
            got.push(e);
            true
        };
        let stats = run.generate(request(), emit).unwrap();
        assert_eq!(collect_ids(&got), collect_ids(&want));
        assert!(stats.steps > 0);
    }
}
