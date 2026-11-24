//! End-to-end parity between qwen4_exp forward and ragged stepper.

use std::io::Write;
use tessera_engine::qwen4_exp::{Config, Moe, Ple, RotatedMoe, program, stepper};
use tessera_engine::qwen35::{Mixing, Recurrent};
use tessera_engine::{
    Bf16, Engine, Pick, PleConfig, PleStore, Recipe, Row, Settings, Stack, Tensor, Weight,
};

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
        multipliers: [100003, 200003, 300007],
        offsets: std::array::from_fn(|h| (h as i64) * 8),
        vocab_sizes: [8; 16],
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
