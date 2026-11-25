//! Fused `add_rms_norm` lowering and parity on MLA and decoder.

use tessera_engine::{Engine, Row};
use tessera_graph::DimName::{Context, Rows, Seqs};
use tessera_graph::{Parts, shard_mode};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Cluster, Device, Limits, Pins, Platform, Recipe, Weight, lower};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::mla;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

/// Bars every `add_rms_norm` kernel variant from `d`.
fn without_add_norm(d: Device) -> Device {
    [
        "block_add_norm_save",
        "block_add_rms_save",
        "block_add_norm",
        "block_add_rms",
        "block_add3_norm_save",
        "block_add3_rms_save",
        "block_add3_norm",
        "block_add3_rms",
        "lane_add_norm_save",
        "lane_add_rms_save",
        "lane_add_norm",
        "lane_add_rms",
        "lane_add3_norm_save",
        "lane_add3_rms_save",
        "lane_add3_norm",
        "lane_add3_rms",
    ]
    .into_iter()
    .fold(d, Device::without)
}

/// Fused `add_rms_norm` reduces steps and traffic on GLM-5.3 and Qwen3.
#[test]
fn glm53_and_qwen3_fuse() {
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    // Full 78-layer GLM-5.3 single-rank with MXFP8 linear weights.
    let mxfp8 = named("mxfp8").unwrap();
    let mxfp8_pins = Pins::all(mxfp8)
        .pin("embed", BF16_FORM)
        .pin("head", BF16_FORM)
        .pin("*.norm", BF16_FORM)
        .pin("*.q_norm", BF16_FORM)
        .pin("*.kv_norm", BF16_FORM)
        .pin("*.k_norm", BF16_FORM)
        .pin("*.k_bias", BF16_FORM)
        .pin("*.mlp_norm.*", BF16_FORM)
        .pin("norm", BF16_FORM)
        .pin("*.router", BF16_FORM)
        .pin("*.router_bias", BF16_FORM)
        .pin("*.weights", BF16_FORM);
    let mxfp8_device = Device::default().on(gpu.clone()).pinned(mxfp8_pins);
    let glm = mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53(1, 2048));
    let sizes = [(Seqs, 1), (Rows, 1), (Context, 2048)];
    let fused = lower(&glm, 0, &sizes, &mxfp8_device).unwrap();
    let unfused = lower(&glm, 0, &sizes, &without_add_norm(mxfp8_device.clone())).unwrap();
    let fused_norms = fused
        .steps
        .iter()
        .filter(|s| s.rule == "add_rms_norm")
        .count();
    let fused_add3 = fused
        .steps
        .iter()
        .filter(|s| s.kernel.name().contains("add3"))
        .count();
    let (fused_cost, unfused_cost) = (mxfp8_device.cost(&fused), mxfp8_device.cost(&unfused));
    assert_eq!(fused_norms, 156);
    assert_eq!(fused_add3, 75);
    assert_eq!(unfused.steps.len() - fused.steps.len(), 156);
    assert!(fused_cost.bytes < unfused_cost.bytes);
    // Six-layer GLM-5.3 on 2-rank TP bf16.
    let bf16_pins = Pins::all(BF16_FORM);
    let tp_device = Device::default()
        .on(gpu)
        .zoned(Cluster::two("gpu0", "gpu1"))
        .ranked(vec![0, 1])
        .pinned(bf16_pins.clone());
    let mut glm6_cfg = mla::Config::glm53(1, 512);
    glm6_cfg.stack.layers = 6;
    let glm6 = mla::stepper::<mla::RotatedMoe>(&glm6_cfg);
    let glm6_tp = shard_mode(&glm6, &Parts::even(2), &bf16_pins.shares(&glm6), false)
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let tp_sizes = [(Seqs, 1), (Rows, 1), (Context, 512)];
    let tp_fused = lower(&glm6_tp, 0, &tp_sizes, &tp_device).unwrap();
    let tp_unfused = lower(&glm6_tp, 0, &tp_sizes, &without_add_norm(tp_device.clone())).unwrap();
    let tp_norms = tp_fused
        .steps
        .iter()
        .filter(|s| s.rule == "add_rms_norm")
        .count();
    assert_eq!(tp_norms, 12);
    assert_eq!(tp_unfused.steps.len() - tp_fused.steps.len(), 12);
    assert!(tp_device.cost(&tp_fused).bytes < tp_device.cost(&tp_unfused).bytes);
    // Qwen3-0.6B decoder on 2-rank TP bf16.
    let qwen = stepper::<Norm, Swiglu>(&Config::qwen3(1, 256));
    let qwen_tp = shard_mode(&qwen, &Parts::even(2), &bf16_pins.shares(&qwen), false)
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let q_sizes = [(Seqs, 1), (Rows, 1), (Context, 256)];
    let q_fused = lower(&qwen_tp, 0, &q_sizes, &tp_device).unwrap();
    let q_unfused = lower(&qwen_tp, 0, &q_sizes, &without_add_norm(tp_device.clone())).unwrap();
    let q_norms = q_fused
        .steps
        .iter()
        .filter(|s| s.rule == "add_rms_norm")
        .count();
    assert_eq!(q_norms, 56);
    assert_eq!(q_unfused.steps.len() - q_fused.steps.len(), 56);
    assert!(tp_device.cost(&q_fused).bytes < tp_device.cost(&q_unfused).bytes);
}

/// Small GLM-5.3 parity with `add_rms_norm` required.
#[test]
fn glm53_small_parity() {
    let small = mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small());
    let w = weights(&small, &mut Noise(19));
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    let recipe = Recipe {
        favor: Some("add_rms_norm".into()),
        require_favored: true,
        ..Recipe::default()
    };
    let batch = [
        Row {
            slot: 0,
            token: 3,
            position: 0,
        },
        Row {
            slot: 1,
            token: 7,
            position: 0,
        },
    ];
    let inputs = vec![
        Tensor::u32(vec![2], vec![3, 7]),
        Tensor::u32(vec![2], vec![0, 0]),
        Tensor::u32(vec![2], vec![0, 1]),
    ];
    for settings in runs() {
        let mut engine = Engine::open(small.clone(), 16, &recipe, &settings).unwrap();
        engine.load(&stored).unwrap();
        let mut store = Store::default();
        let ref_out = run_kept(&small, &w, &mut store, 0, &inputs).unwrap();
        let want: Vec<u32> = ref_out[0]
            .indices()
            .unwrap()
            .iter()
            .map(|&i| i as u32)
            .collect();
        assert_eq!(engine.step(&batch).unwrap(), want);
    }
}
