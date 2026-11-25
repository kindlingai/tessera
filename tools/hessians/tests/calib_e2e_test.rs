//! End-to-end offline calibration driver test with resume and reproducibility.

use std::collections::BTreeSet;
use std::fs;
use tessera_checkpoint::File;
use tessera_checkpoint::hessian::{dense_name, expert_name, inputs_name, route_name};
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_gpukit::platform::Asked;
use tessera_gpukit::{Limits, Platform};
use tessera_graph::{CalibTap, Composite, Role};
use tessera_hessians::args::{Command, Config, parse_args, parse_command, write_npy};
use tessera_hessians::driver::{calib_pins, layer_sizes, run_mla};
use tessera_hessians::parity::compare_dirs;
use tessera_lower::form::BF16_FORM;
use tessera_lower::{Compute, Device, Pins, cut_region, lower};
use tessera_models::{Stack, mla};
use tessera_testkit::{Noise, weights};

fn toy_mla() -> mla::Config {
    mla::Config {
        stack: Stack::sized([64, 16, 2, 2, 8, 8], 4, 8),
        dense_layers: 1,
        nope: 4,
        rope: 4,
        rope_theta: 1e4,
        q_latent: 8,
        latent: 8,
        dense_ffn: 16,
        experts: 4,
        top_k: 2,
        route_scale: 2.5,
        index: None,
        eps: 1e-6,
    }
}

fn run_once(tag: &str) -> std::path::PathBuf {
    let root = tessera_testkit::scratch(&format!("{}-{tag}", std::process::id()));
    let model = root.join("model");
    let work = root.join("work");
    let out = root.join("out");
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("config.json"), b"{\"hidden_size\":128}\n").unwrap();
    fs::write(
        model.join("model.safetensors.index.json"),
        b"{\"weight_map\":{}}\n",
    )
    .unwrap();
    let tok_path = root.join("tokens.npy");
    let toks: Vec<u32> = (0..24).map(|i| (i * 5 + 1) % 64).collect();
    write_npy(&tok_path, 6, 4, &toks).unwrap();

    let mla_cfg = toy_mla();
    let g = mla::calib_program::<mla::Moe>(&mla_cfg);
    let w = weights(&g, &mut Noise(19));

    let base = Config {
        model: model.clone(),
        tokens: tok_path.clone(),
        heldout: 2,
        layers: vec![0],
        hess_layers: BTreeSet::from([0, 1]),
        save_inputs: BTreeSet::from([1]),
        route_log: BTreeSet::from([1]),
        hess_id: "toy-calib-v1".into(),
        weights_id: None,
        batch: 2,
        experts_per_file: 2,
        dense_heldout: true,
        no_dense: false,
        no_experts: false,
        work: work.clone(),
        out: out.clone(),
        resume: false,
        state: None,
        keep_hidden: false,
        gpu: false,
        device: None,
        tile: 4,
    };
    // Pass 1 runs layer 0, then resumes through layer 1.
    run_mla(&base, &mla_cfg, &w).unwrap();
    assert!(work.join("h-001.bin").exists());
    assert!(!work.join("h-000.bin").exists());

    let mut resumed = base;
    resumed.layers = vec![0, 1];
    resumed.resume = true;
    run_mla(&resumed, &mla_cfg, &w).unwrap();
    root
}

#[test]
fn calib_e2e_resume() {
    let args: Vec<String> = [
        "--model",
        "m",
        "--tokens",
        "t.npy",
        "--work",
        "w",
        "--out",
        "o",
        "--hess-id",
        "id1",
        "--layers",
        "0-1",
        "--dense-heldout",
        "--resume",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let parsed = parse_args(&args).unwrap();
    assert_eq!(parsed.layers, vec![0, 1]);
    assert!(parsed.dense_heldout && parsed.resume);

    let r1 = run_once("hess-r1");
    let r2 = run_once("hess-r2");
    let o1 = r1.join("out");
    let o2 = r2.join("out");

    let want_wid = "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1";
    let files = [
        dense_name(0, false),
        dense_name(0, true),
        dense_name(1, false),
        dense_name(1, true),
        expert_name(1, 0, 1),
        expert_name(1, 2, 3),
        inputs_name(1),
        route_name(1),
    ];
    for name in &files {
        let b1 = fs::read(o1.join(name)).unwrap();
        let b2 = fs::read(o2.join(name)).unwrap();
        assert_eq!(b1, b2, "{name} must be byte-identical across runs");
        let f = File::open(&o1.join(name)).unwrap();
        assert_eq!(f.meta("weights_id"), Some(want_wid));
        assert_eq!(f.meta("hessian_id"), Some("toy-calib-v1"));
    }

    let e0 = File::open(&o1.join(expert_name(1, 0, 1))).unwrap();
    let e1 = File::open(&o1.join(expert_name(1, 2, 3))).unwrap();
    let c0 = e0.i64s("count").unwrap();
    let c1 = e1.i64s("count").unwrap();
    assert_eq!(c0.iter().chain(&c1).sum::<i64>(), 4 * 4 * 2);

    let d1 = File::open(&o1.join(dense_name(1, false))).unwrap();
    assert_eq!(d1.i64s("count_mlp_in").unwrap(), [16]);
    assert_eq!(d1.i64s("count_shared_down_in").unwrap(), [16]);
    let dh1 = File::open(&o1.join(dense_name(1, true))).unwrap();
    assert_eq!(dh1.meta("sequences"), Some("held out"));
    assert_eq!(dh1.i64s("count_mlp_in").unwrap(), [8]);

    let inp = File::open(&o1.join(inputs_name(1))).unwrap();
    assert_eq!(inp.meta("heldout_sequences"), Some("2"));

    let rep = compare_dirs(&o1, &o2, 1, 1e-6).unwrap();
    assert!(rep.passed && rep.worst_rel == 0.0);
    assert_eq!(rep.inputs.as_ref().unwrap().idx_match, 1.0);
    let bench_cmd = parse_command(&["--bench-kernels".into(), "--device".into(), "0".into()]);
    assert!(matches!(
        bench_cmd,
        Ok(Command::BenchKernels { device: Some(0) })
    ));
    let par_cmd = parse_command(&[
        "--verify-parity".into(),
        "a".into(),
        "b".into(),
        "--layer".into(),
        "3".into(),
    ]);
    assert!(matches!(
        par_cmd,
        Ok(Command::VerifyParity { layer: 3, .. })
    ));

    fs::remove_dir_all(r1).unwrap();
    fs::remove_dir_all(r2).unwrap();
}

#[test]
fn cuda_plan_lowering() {
    let cfg = mla::Config {
        stack: Stack::sized([256, 256, 2, 4, 32, 128], 2, 16),
        dense_layers: 1,
        nope: 16,
        rope: 16,
        rope_theta: 1e4,
        q_latent: 128,
        latent: 128,
        dense_ffn: 256,
        experts: 4,
        top_k: 2,
        route_scale: 2.5,
        index: None,
        eps: 1e-6,
    };
    let g = mla::calib_program::<mla::Moe>(&cfg);
    let region = g
        .regions
        .iter()
        .filter(|r| r.kind == Composite::Layer)
        .nth(1)
        .unwrap();
    let lg = cut_region(&g, region);
    let settings = Settings {
        backend: Pick::Gpu(Backend::Cuda),
        ..Settings::default()
    };
    let gb10 = Platform::new(Asked {
        backend: "cuda",
        arch: "sm_121".into(),
        name: "NVIDIA GB10".into(),
        driver: "13.1".into(),
        limits: Limits {
            lanes: 32,
            group_bytes: 101_376,
            group_threads: 1024,
            compiler: true,
            pdl: false,
        },
        units: 48,
        bandwidth: 273e9,
        integrated: true,
        memory: 128 * 1024 * 1024 * 1024,
    });
    let dev = Device::guess(&settings)
        .on(gb10)
        .pinned(calib_pins(Pins::all(BF16_FORM)))
        .with_compute(Compute::Bf16);
    let plan = lower(&lg, 0, &layer_sizes(2, 16), &dev).unwrap();

    // Taps must keep their natural pinned form.
    assert_eq!(plan.outputs[0].slot.form.name(), "bf16");
    let mut saw_mlp_bf16 = false;
    let mut saw_gates_f32 = false;
    for (o, po) in lg.entries[0].outputs[1..].iter().zip(&plan.outputs[1..]) {
        match o.role {
            Role::Calib(CalibTap::MlpIn) => {
                assert_eq!(po.slot.form.name(), "bf16");
                saw_mlp_bf16 = true;
            }
            Role::Calib(CalibTap::RouterGates) => {
                assert_eq!(po.slot.form.name(), "f32");
                saw_gates_f32 = true;
            }
            _ => {}
        }
    }
    assert!(saw_mlp_bf16 && saw_gates_f32);
    assert!(
        plan.steps.iter().any(|s| s.kernel.name() == "pbf16_gu"),
        "expected pbf16_gu CUDA kernel in MoE layer plan"
    );
    assert!(
        plan.steps.iter().any(|s| s.kernel.name() == "pbf16_gemm"),
        "expected pbf16_gemm CUDA kernel in MoE layer plan"
    );
    assert!(
        !plan.steps.iter().any(|s| s.rule == "picked"),
        "scalar picked must not beat grouped BF16 Tensor-Core GEMM"
    );
}
