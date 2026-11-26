//! Model graph trimming and rule-fusion lowering checks.

use tessera_graph::DimName::{Context, Rows, Seqs};
use tessera_graph::{DType, Op, Parts, shard_mode};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Cluster, Device, Limits, Pins, Plan, Platform, Recipe, Settings, lower};
use tessera_models::decoder::{Config, Moe, Swiglu, stepper};
use tessera_models::qwen4_exp::{self, Ple, RotatedMoe};
use tessera_models::qwen35::Recurrent;
use tessera_models::{Stack, mla};
use tessera_rules as _;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, weights};
use tessera_trace::DimName::Picks;
use tessera_trace::Norm;

fn step_feeds(rows: &[(u32, u32, u32)]) -> Vec<Tensor> {
    let n = rows.len();
    vec![
        Tensor::u32(vec![n], rows.iter().map(|r| r.1).collect()),
        Tensor::u32(vec![n], rows.iter().map(|r| r.2).collect()),
        Tensor::u32(vec![n], rows.iter().map(|r| r.0).collect()),
    ]
}

fn count(plan: &Plan, name: &str) -> usize {
    plan.steps
        .iter()
        .filter(|s| s.kernel.name() == name)
        .count()
}

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

/// Checks fused vs unfused launch counts on MLA.
fn check_launch_drop(cfg: &mla::Config, rows: u64, device: &Device) {
    let g = mla::stepper::<mla::RotatedMoe>(cfg);
    let sizes = [(Seqs, cfg.stack.max_seqs), (Rows, rows), (Context, 64)];
    let fused = lower(&g, 0, &sizes, device).unwrap();
    let bare = device
        .clone()
        .without("mla_kv_scatter")
        .without("mla_kv_scatter_plain")
        .without("join_rope");
    let unfused = lower(&g, 0, &sizes, &bare).unwrap();
    let layers = cfg.stack.layers as usize;
    let indexing = (0..cfg.stack.layers).filter(|&l| cfg.indexes(l)).count();
    let idx_joins = indexing
        * if cfg.index.is_some_and(|i| 64 > i.keep) {
            2
        } else {
            1
        };
    assert_eq!(count(&fused, "mla_kv_scatter"), layers);
    assert_eq!(count(&fused, "join_rope"), layers + idx_joins);
    assert_eq!(count(&fused, "rope_adjacent"), 0);
    assert_eq!(count(&fused, "join"), 0);
    // Saves 5 KV launches plus 2 or 3 Q launches.
    let min_drop = layers * if rows == 1 { 7 } else { 8 };
    assert!(
        unfused.steps.len() >= fused.steps.len() + min_drop,
        "rows={rows}: fused={} unfused={} (want >= {min_drop} drop)",
        fused.steps.len(),
        unfused.steps.len(),
    );
}

/// Trimmed steps return the whole graph's rows at their picks.
#[test]
fn trim_keeps_picks() {
    let graphs = [
        stepper::<Norm, Swiglu>(&Config::qwen3_small()),
        stepper::<(), Moe>(&Config::granite_small()),
        mla::stepper::<mla::Moe>(&mla::Config::deepseek_small()),
        mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
    ];
    let launches = [
        (vec![(1, 5, 0), (1, 7, 1), (0, 9, 0)], vec![1, 2]),
        (vec![(0, 4, 1), (1, 3, 2)], vec![0, 1]),
    ];
    for g in graphs {
        let trimmed = tessera_graph::trim(&g, 0).unwrap();
        let picked =
            |n: &tessera_graph::Node| n.op == Op::Matmul && n.ty.shape.0[0] == Picks.into();
        assert!(trimmed.nodes.iter().any(picked));
        let weights = weights(&g, &mut Noise(3));
        let (mut whole, mut cut) = (Store::default(), Store::default());
        for (rows, picks) in &launches {
            let mut feeds = step_feeds(rows);
            let want = run_kept(&g, &weights, &mut whole, 0, &feeds).unwrap();
            feeds.push(Tensor::u32(vec![picks.len()], picks.clone()));
            let got = run_kept(&trimmed, &weights, &mut cut, 0, &feeds).unwrap();
            let (want, got) = (want[1].floats().unwrap(), got[1].floats().unwrap());
            let width = got.len() / picks.len();
            for (at, p) in picks.iter().enumerate() {
                let p = *p as usize;
                assert_eq!(got[at * width..][..width], want[p * width..][..width]);
            }
        }
    }
}

#[test]
fn rotated_trims() {
    let tiny = qwen4_exp::Config::tiny();
    let c = qwen4_exp::Config {
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
    let g = qwen4_exp::stepper::<RotatedMoe>(&c);
    let trimmed = tessera_graph::trim(&g, 0).unwrap();
    assert!(!trimmed.nodes.is_empty());
}

/// GLM-5.3 lowers its indexer to one fused kernel without gather.
#[test]
fn mla_indexer_fused() {
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.max_context = 512;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let sizes = [(Seqs, 4), (Rows, 4), (Context, 512)];
    for settings in runs() {
        let device = Device::guess(&settings);
        let fused = lower(&g, 0, &sizes, &device).unwrap();
        let names: Vec<&str> = fused.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(names.contains(&"indexer_warp"), "{names:?}");
        assert!(!names.contains(&"kept_gather"), "{names:?}");
        let bare = device.without("indexer_warp").without("indexer");
        let unfused = lower(&g, 0, &sizes, &bare).unwrap();
        assert!(fused.steps.len() < unfused.steps.len());
        assert!(
            fused.scratch() < unfused.scratch(),
            "fused={} unfused={}",
            fused.scratch(),
            unfused.scratch()
        );
    }
    let whole = mla::program::<mla::RotatedMoe>(&cfg);
    let device = Device::guess(&Settings::default());
    let plan = lower(&whole, 0, &[(Seqs, 1), (Context, 8)], &device).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(names.contains(&"indexer_warp"), "{names:?}");
    let indexing = (0..cfg.stack.layers).filter(|&l| cfg.indexes(l)).count();
    let ropes: usize = g
        .nodes
        .iter()
        .map(|n| match &n.op {
            Op::Row(row) => row
                .0
                .iter()
                .filter(|s| matches!(s, tessera_graph::Stage::Rope { .. }))
                .count(),
            _ => 0,
        })
        .sum();
    assert_eq!(ropes, 2 * cfg.stack.layers as usize + 2 * indexing);
}

#[test]
fn mla_prep_launch_reduction() {
    let mut small = mla::Config::glm53_small();
    small.stack.max_context = 64;
    let fwd = mla::program::<mla::RotatedMoe>(&small);
    for settings in runs() {
        let device = Device::guess(&settings);
        for rows in [1u64, 8] {
            check_launch_drop(&small, rows, &device);
        }
        let plan = lower(&fwd, 0, &[(Seqs, 1), (Context, 8)], &device).unwrap();
        let indexing = (0..small.stack.layers)
            .filter(|&l| small.indexes(l))
            .count();
        assert_eq!(
            count(&plan, "join_rope"),
            small.stack.layers as usize + 2 * indexing
        );
    }
    let full = mla::Config::glm53(4, 512);
    let device = Device::guess(&host());
    for rows in [1u64, 8] {
        check_launch_drop(&full, rows, &device);
    }
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
        .wired(DType::F32);
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
        .wired(DType::F32);
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
    let recipe = Recipe {
        favor: Some("add_rms_norm".into()),
        require_favored: true,
        ..Recipe::default()
    };
    let inputs = step_feeds(&[(0, 3, 0), (1, 7, 0)]);
    let mut store = Store::default();
    let ref_out = run_kept(&small, &w, &mut store, 0, &inputs).unwrap();
    let want = ref_out[0].indices().unwrap();
    let sizes = [(Seqs, 4), (Rows, 2), (Context, 16)];
    let feeds: Vec<_> = inputs.iter().map(Tensor::view).collect();
    for settings in runs() {
        let device = Device::guess(&settings).with(&recipe);
        let plan = lower(&small, 0, &sizes, &device).unwrap();
        let mut runner = tessera_run::open(plan, &settings).unwrap();
        for (at, weight) in w.iter().enumerate() {
            runner.load(at as u32, weight.view()).unwrap();
        }
        let got = runner.run(&feeds).unwrap();
        assert_eq!(got[0].indices().unwrap(), want);
    }
}
