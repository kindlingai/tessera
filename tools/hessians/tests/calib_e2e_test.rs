//! End-to-end host calibration run with resume.

use std::fs;
use tessera_checkpoint::File;
use tessera_checkpoint::hessian::{dense_name, expert_name, inputs_name, route_name};
use tessera_hessians::args::{parse_args, write_npy};
use tessera_hessians::driver::run_mla;
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

#[test]
fn calib_e2e_resume() {
    let root = tessera_testkit::scratch(&format!("{}-hess-e2e", std::process::id()));
    let (model, out) = (root.join("model"), root.join("out"));
    fs::create_dir_all(&model).unwrap();
    fs::write(model.join("config.json"), b"{\"hidden_size\":128}\n").unwrap();
    fs::write(
        model.join("model.safetensors.index.json"),
        b"{\"weight_map\":{}}\n",
    )
    .unwrap();
    let toks: Vec<u32> = (0..24).map(|i| (i * 5 + 1) % 64).collect();
    write_npy(&root.join("tokens.npy"), 6, 4, &toks).unwrap();
    let flags = "--hess-id toy-calib-v1 --heldout 2 --layers 0 --hess-layers 0-1 \
        --save-inputs 1 --route-log 1 --batch 2 --experts-per-file 2 --dense-heldout";
    let mut args: Vec<String> = flags.split_whitespace().map(String::from).collect();
    for (flag, dir) in [
        ("--model", "model"),
        ("--tokens", "tokens.npy"),
        ("--work", "work"),
        ("--out", "out"),
    ] {
        args.extend([flag.into(), root.join(dir).display().to_string()]);
    }
    let mut cfg = parse_args(&args).unwrap();
    cfg.resolve(2).unwrap();
    cfg.tile = 4;

    let mla_cfg = toy_mla();
    let w = weights(&mla::calib_program::<mla::Moe>(&mla_cfg), &mut Noise(19));
    run_mla(&cfg, &mla_cfg, &w).unwrap();
    assert!(root.join("work/h-001.bin").exists());
    assert!(!root.join("work/h-000.bin").exists());

    cfg.layers = vec![0, 1];
    assert!(run_mla(&cfg, &mla_cfg, &w).is_err());
    cfg.resume = true;
    run_mla(&cfg, &mla_cfg, &w).unwrap();

    let want_wid = "24b0d8c953197327441b27e43ecf94a2f3a7adbaf84b22c9ab0123b7c22519a1";
    for name in [
        dense_name(0, true),
        expert_name(1, 0, 1),
        inputs_name(1),
        route_name(1),
    ] {
        let f = File::open(&out.join(name)).unwrap();
        assert_eq!(f.meta("weights_id"), Some(want_wid));
    }
    let count =
        |name: String, tensor: &str| File::open(&out.join(name)).unwrap().i64s(tensor).unwrap();
    let picks: i64 = [expert_name(1, 0, 1), expert_name(1, 2, 3)]
        .into_iter()
        .map(|n| count(n, "count").iter().sum::<i64>())
        .sum();
    assert_eq!(picks, 4 * 4 * 2);
    assert_eq!(count(dense_name(1, false), "count_shared_down_in"), [16]);
    assert_eq!(count(dense_name(1, true), "count_mlp_in"), [8]);
    fs::remove_dir_all(root).unwrap();
}
