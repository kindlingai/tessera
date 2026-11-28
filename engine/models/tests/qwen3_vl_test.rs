//! End-to-end parity, isolation, and lowering checks for Qwen3-VL-Embedding.

use tessera_gpukit::Limits;
use tessera_gpukit::platform::{Asked, Platform};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, named};
use tessera_lower::{Device, Pins, Plan, Settings, View, lower};
use tessera_models::keys::{Keys, Offered};
use tessera_models::qwen3_vl::{Config, MERGED, VIS, VisionConfig, text_program, vision_program};
use tessera_rules as _;
use tessera_run::testing::runs;
use tessera_run::{Run, open};
use tessera_testkit::{Noise, cosine_f32, fill, max_abs_diff};
use tessera_trace::DimName::{Picks, Rows};
use tessera_trace::{Graph, Linear};

fn loaded(plan: Plan, weights: &[Tensor], settings: &Settings) -> Box<dyn Run> {
    let mut runner = open(plan, settings).unwrap();
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.view()).unwrap();
    }
    runner
}

fn faux_weights(g: &Graph, seed: u64) -> Vec<Tensor> {
    let mut noise = Noise(seed);
    g.weights
        .iter()
        .map(|w| fill(g, &w.ty, &[], &mut noise).unwrap())
        .collect()
}

fn check_close(label: &str, got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
    assert!(off < 2e-3 && cos > 0.9999, "{label}: off {off:e} cos {cos}");
}

/// Builds packed vision inputs for `(h_merged, w_merged)` images.
fn vision_inputs(c: &VisionConfig, images: &[(usize, usize)], seed: u64) -> (usize, Vec<Tensor>) {
    let mut noise = Noise(seed);
    let merged: usize = images.iter().map(|&(h, w)| h * w).sum();
    let patch_dim = c.patch_dim() as usize;
    let grid = c.pos_grid() as usize;
    let patches = Tensor::f32(
        vec![merged, 4, patch_dim],
        (0..merged * 4 * patch_dim).map(|_| noise.draw()).collect(),
    );
    let mut pos_ids = Vec::with_capacity(merged * 16);
    let mut pos_w = Vec::with_capacity(merged * 16);
    let mut rot_pos = Vec::with_capacity(merged * 8);
    let mut bounds = Vec::with_capacity(merged * 8);
    let mut offset = 0u32;
    for &(hm, wm) in images {
        let (vh, vw) = (hm * 2, wm * 2);
        let n = (vh * vw) as u32;
        for blk in 0..hm * wm {
            let (by, bx) = (blk / wm, blk % wm);
            for sub in 0..4 {
                let pr = by * 2 + sub / 2;
                let pc = bx * 2 + sub % 2;
                let rh = if vh > 1 {
                    pr as f32 * (grid - 1) as f32 / (vh - 1) as f32
                } else {
                    0.0
                };
                let rw = if vw > 1 {
                    pc as f32 * (grid - 1) as f32 / (vw - 1) as f32
                } else {
                    0.0
                };
                let (r0, c0) = (rh.floor() as usize, rw.floor() as usize);
                let (r1, c1) = ((r0 + 1).min(grid - 1), (c0 + 1).min(grid - 1));
                let (dr, dc) = (rh - r0 as f32, rw - c0 as f32);
                pos_ids.extend_from_slice(&[
                    (r0 * grid + c0) as u32,
                    (r0 * grid + c1) as u32,
                    (r1 * grid + c0) as u32,
                    (r1 * grid + c1) as u32,
                ]);
                pos_w.extend_from_slice(&[
                    (1.0 - dr) * (1.0 - dc),
                    (1.0 - dr) * dc,
                    dr * (1.0 - dc),
                    dr * dc,
                ]);
                rot_pos.extend_from_slice(&[pr as u32, pc as u32]);
                bounds.extend_from_slice(&[offset, offset + n]);
            }
        }
        offset += n;
    }
    let inputs = vec![
        patches,
        Tensor::u32(vec![merged, 4, 4], pos_ids),
        Tensor::f32(vec![merged, 4, 4], pos_w),
        Tensor::u32(vec![merged, 4, 2], rot_pos),
        Tensor::u32(vec![merged, 4, 2], bounds),
    ];
    (merged, inputs)
}

#[test]
fn vision_parity() {
    let c = Config::small().vision;
    let g = vision_program::<Linear>(&c);
    let weights = faux_weights(&g, 11);
    let (merged, inputs) = vision_inputs(&c, &[(2, 2), (1, 2)], 23);
    let want = run(&g, &weights, 0, &inputs).unwrap();
    assert_eq!(want.len(), 1 + c.deepstack_indexes.len());
    // Mutating image 1 must not change image 0.
    let mut alt = inputs.clone();
    let mut p = alt[0].floats().unwrap().to_vec();
    let split = 4 * 4 * c.patch_dim() as usize;
    for v in &mut p[split..] {
        *v += 1.0;
    }
    alt[0] = Tensor::f32(alt[0].shape.clone(), p);
    let want_alt = run(&g, &weights, 0, &alt).unwrap();
    let img0_len = 4 * c.out_hidden as usize;
    assert_eq!(
        &want[0].floats().unwrap()[..img0_len],
        &want_alt[0].floats().unwrap()[..img0_len]
    );
    check_backends(&g, &weights, &inputs, &[(MERGED, merged as u64)], "vision");
}

fn check_backends(
    g: &Graph,
    weights: &[Tensor],
    inputs: &[Tensor],
    sizes: &[(tessera_graph::DimName, u64)],
    tag: &str,
) {
    let views: Vec<View> = inputs.iter().map(Tensor::view).collect();
    for settings in runs() {
        for form in [F32_FORM, BF16_FORM] {
            let pins = Pins::all(form);
            let mut kept = Vec::with_capacity(weights.len());
            for (t, w) in weights.iter().zip(&g.weights) {
                let f = pins.form(&w.path);
                kept.push(Tensor::f32(
                    t.shape.clone(),
                    f.roundtrip(t.floats().unwrap()),
                ));
            }
            let want_f = run(g, &kept, 0, inputs).unwrap();
            let device = Device::guess(&settings).pinned(pins);
            let plan = lower(g, 0, sizes, &device).unwrap();
            let mut runner = loaded(plan, &kept, &settings);
            let got = runner.run(&views).unwrap();
            for (i, (a, b)) in got.iter().zip(&want_f).enumerate() {
                let label = format!("{tag} out[{i}] {} {:?}", form.name(), settings.backend);
                check_close(&label, a.floats().unwrap(), b.floats().unwrap());
            }
        }
    }
}

#[test]
fn text_parity() {
    let c = Config::small().text;
    let g = text_program::<Linear>(&c);
    let weights = faux_weights(&g, 31);
    let mut noise = Noise(37);
    // Two packed sequences: multimodal len 5, text-only len 6.
    let rows = 11usize;
    let vis = 2usize;
    let tokens = Tensor::u32(
        vec![rows],
        vec![1, u32::MAX, u32::MAX, 4, 5, 7, 8, 9, 10, 11, 12],
    );
    let positions = Tensor::u32(
        vec![rows, 3],
        vec![
            0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4,
            4, 5, 5, 5,
        ],
    );
    let bounds = Tensor::u32(
        vec![rows, 2],
        vec![
            0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 5, 11, 5, 11, 5, 11, 5, 11, 5, 11, 5, 11,
        ],
    );
    let picks = Tensor::u32(vec![2], vec![4, 10]);
    let vis_idx = Tensor::u32(
        vec![rows],
        vec![
            u32::MAX,
            0,
            1,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
            u32::MAX,
        ],
    );
    let mut inputs = vec![tokens, positions, bounds, picks, vis_idx];
    for _ in 0..=c.deepstack_layers {
        inputs.push(Tensor::f32(
            vec![vis, c.hidden as usize],
            (0..vis * c.hidden as usize).map(|_| noise.draw()).collect(),
        ));
    }
    let want = run(&g, &weights, 0, &inputs).unwrap();
    // Mutating seq 1 must not change seq 0.
    let mut alt = inputs.clone();
    alt[0] = Tensor::u32(
        vec![rows],
        vec![1, u32::MAX, u32::MAX, 4, 5, 20, 21, 22, 23, 24, 25],
    );
    let want_alt = run(&g, &weights, 0, &alt).unwrap();
    let h = c.hidden as usize;
    assert_eq!(
        &want[0].floats().unwrap()[..h],
        &want_alt[0].floats().unwrap()[..h]
    );

    let sizes = [(Rows, rows as u64), (Picks, 2), (VIS, vis as u64)];
    check_backends(&g, &weights, &inputs, &sizes, "text");
}

/// Checks 8B vision and text lower on CPU, CUDA, ROCm.
#[test]
fn eight_b_lowers() {
    let c = Config::qwen3_vl_8b(8, 2048, 256);
    let vg = vision_program::<Linear>(&c.vision);
    let tg = text_program::<Linear>(&c.text);
    assert_eq!(Graph::from_json(&vg.to_json()).unwrap(), vg);
    assert_eq!(Graph::from_json(&tg.to_json()).unwrap(), tg);
    let target = |backend: &'static str, arch: &str| {
        Device::default().on(Platform::new(Asked {
            backend,
            arch: arch.into(),
            name: arch.into(),
            driver: String::new(),
            limits: Limits::assumed(),
            units: 48,
            bandwidth: 256e9,
            integrated: true,
            memory: 64 << 30,
        }))
    };
    let platforms = [
        ("cpu", Device::guess(&Settings::default())),
        ("cuda", target("cuda", "sm_120")),
        ("rocm", target("rocm", "gfx1151")),
    ];
    for (name, base) in platforms {
        let dev = base.pinned(Pins::all(named("bf16").unwrap()));
        let vp = lower(&vg, 0, &[(MERGED, 16)], &dev).unwrap();
        let tp = lower(&tg, 0, &[(Rows, 32), (Picks, 2), (VIS, 16)], &dev).unwrap();
        if name != "cpu" {
            let has_flash = vp
                .steps
                .iter()
                .any(|s| s.kernel.name().starts_with("flash_packed"));
            assert!(has_flash, "{name} 8B vision did not pick flash_packed");
        }
        assert!(!tp.steps.is_empty());
    }
}

#[test]
fn config_reads_hf() {
    let json = r#"{
        "model_type": "qwen3_vl",
        "text_config": {
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "attention_bias": false,
            "rope_theta": 5000000.0,
            "rope_scaling": { "mrope_section": [24, 20, 20] }
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 4096,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [8, 16, 24]
        }
    }"#;
    let offered = [Offered::parse("config.json", json).unwrap()];
    let mut keys = Keys::new(&offered);
    let got = Config::read(&mut keys, 8, 2048, 256);
    keys.finish().unwrap();
    assert_eq!(got, Config::qwen3_vl_8b(8, 2048, 256));
}
