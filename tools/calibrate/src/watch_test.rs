use crate::watch::{model_target, watch};
use crate::watch_args::{build_options, parse_parts, parse_watch};
use crate::watch_report::{
    KernelReport, Meta, Row, build_report, format_diff, format_table, load_report, write_json,
};
use tessera_graph::{DimName, Parts, Size};
use tessera_lower::{Device, lower};
use tessera_run::KernelAttrs;
use tessera_testkit::scratch;

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn parse_watch_flags() {
    let spec = parse_watch(&args(&[
        "out_dir",
        "model",
        "glm53",
        "--recipe",
        "examples/recipes/glm53-moe",
        "--parts",
        "4",
        "--rank",
        "0",
        "--region",
        "moe.0",
        "--once",
        "--json",
        "out.json",
        "--against",
        "base.json",
        "--favor",
        "glm53_moe",
        "--bar",
        "slow_k",
        "--layers",
        "2",
        "--kv-dtype",
        "fp8",
        "--moe",
        "tp",
        "--sum-wire",
        "fp8",
        "--require-favored",
    ]))
    .unwrap();
    assert_eq!(spec.parts, Some(Parts::even(4)));
    assert_eq!(spec.rank, Some(0));
    assert!(spec.once);
    assert_eq!(spec.recipe.as_deref(), Some("examples/recipes/glm53-moe"));
    assert_eq!(spec.favor.as_deref(), Some("glm53_moe"));
    assert_eq!(spec.barred, vec!["slow_k"]);
    assert_eq!(
        spec.subject_args,
        args(&["model", "glm53", "--region", "moe.0"])
    );
    assert_eq!(parse_parts("2,1,1").unwrap(), Parts(vec![2, 1, 1]));
    assert!(parse_parts("0").is_err());
    assert!(parse_parts("1,0").is_err());
    assert!(
        parse_watch(&args(&[
            "d", "gemm", "4", "8", "8", "--parts", "2", "--rank", "2"
        ]))
        .is_err()
    );
    assert!(parse_watch(&args(&["d", "gemm", "4", "8", "8", "--bogus", "1"])).is_err());
}

#[test]
fn report_json_and_diff() {
    let meta = Meta {
        device: "test-gpu".into(),
        parts: vec![1, 1, 1, 1],
        rank: 0,
        region: Some("moe.0".into()),
        recipe: Some("glm53-moe".into()),
    };
    let rows = vec![Row {
        entry: "glm53_moe_k".into(),
        rule: "glm53_moe",
        steps: vec![0, 1],
        seconds: 120e-6,
        planned: 100e-6,
        bound: Some(80e-6),
        loaded: true,
        attrs: Some(KernelAttrs {
            regs: 64,
            local_bytes: 16,
            shared_bytes: 4096,
            ctas_per_sm: 2,
        }),
    }];
    let base = build_report(&meta, &rows, 150e-6, 1e-4, 2.0);
    let table = format_table(&base);
    assert!(table.contains("regs") && table.contains("ctas") && table.contains("*glm53_moe_k"));
    let dir = scratch("watch-report-json");
    let path = dir.join("base.json");
    write_json(&path, &base).unwrap();
    let loaded = load_report(&path).unwrap();
    assert_eq!(loaded, base);
    let mut cand = base.clone();
    cand.plan_us = 100.0;
    cand.kernels[0].us = 80.0;
    cand.kernels[0].regs = Some(48);
    cand.kernels[0].lmem = Some(0);
    cand.kernels[0].ctas = Some(3);
    cand.kernels.push(KernelReport {
        entry: "extra_k".into(),
        rule: "extra".into(),
        us: 10.0,
        planned_us: 10.0,
        bound_us: None,
        steps: 1,
        loaded: false,
        regs: None,
        smem: None,
        lmem: None,
        ctas: None,
    });
    let diff = format_diff(&base, &cand);
    assert!(diff.contains("1.50x") && diff.contains("r64->48") && diff.contains("lmem 16->0"));
    assert!(diff.contains("ctas 2->3") && diff.contains("extra_k"));
    let mut inf_report = base;
    inf_report.err = f32::INFINITY;
    let inf_path = dir.join("inf.json");
    write_json(&inf_path, &inf_report).unwrap();
    assert!(load_report(&inf_path).unwrap().err.is_infinite());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn watch_once_offline() {
    let settings = tessera_run::testing::host();
    let dir = scratch("watch-once-offline");
    let kdir = dir.join("kernels");
    let base_json = dir.join("base.json");
    let cand_json = dir.join("cand.json");
    watch(
        &settings,
        &args(&[
            kdir.to_str().unwrap(),
            "gemm",
            "4",
            "16",
            "16",
            "--once",
            "--json",
            base_json.to_str().unwrap(),
        ]),
    )
    .unwrap();
    let base = load_report(&base_json).unwrap();
    assert_eq!(base.parts, vec![1]);
    assert_eq!(base.rank, 0);
    assert!(!base.kernels.is_empty() && base.err < 1e-4);
    watch(
        &settings,
        &args(&[
            kdir.to_str().unwrap(),
            "gemm",
            "4",
            "16",
            "16",
            "--once",
            "--json",
            cand_json.to_str().unwrap(),
            "--against",
            base_json.to_str().unwrap(),
        ]),
    )
    .unwrap();
    let moe_json = dir.join("moe.json");
    watch(
        &settings,
        &args(&[
            kdir.to_str().unwrap(),
            "moe",
            "4",
            "2",
            "16",
            "32",
            "4",
            "--parts",
            "2",
            "--rank",
            "0",
            "--once",
            "--json",
            moe_json.to_str().unwrap(),
        ]),
    )
    .unwrap();
    let moe = load_report(&moe_json).unwrap();
    assert_eq!(moe.parts, vec![1, 1]);
    assert_eq!(moe.rank, 0);
    assert!(!moe.kernels.is_empty() && moe.err < 1e-4);
    let _ = std::fs::remove_dir_all(&dir);
}

fn write_tiny_qwen2(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    let cfg = serde_json::json!({
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true
    });
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let tensors: [(&str, &[usize]); 13] = [
        ("model.embed_tokens.weight", &[32, 16]),
        ("model.norm.weight", &[16]),
        ("model.layers.0.input_layernorm.weight", &[16]),
        ("model.layers.0.post_attention_layernorm.weight", &[16]),
        ("model.layers.0.self_attn.q_proj.weight", &[16, 16]),
        ("model.layers.0.self_attn.q_proj.bias", &[16]),
        ("model.layers.0.self_attn.k_proj.weight", &[16, 16]),
        ("model.layers.0.self_attn.k_proj.bias", &[16]),
        ("model.layers.0.self_attn.v_proj.weight", &[16, 16]),
        ("model.layers.0.self_attn.v_proj.bias", &[16]),
        ("model.layers.0.self_attn.o_proj.weight", &[16, 16]),
        ("model.layers.0.mlp.gate_proj.weight", &[32, 16]),
        ("model.layers.0.mlp.up_proj.weight", &[32, 16]),
    ];
    let extra = ("model.layers.0.mlp.down_proj.weight", &[16usize, 32][..]);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in tensors.into_iter().chain(std::iter::once(extra)) {
        let from = data.len();
        let elems: usize = shape.iter().product();
        data.resize(from + elems * 4, 0u8);
        header.insert(
            name.into(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [from, data.len()]}),
        );
    }
    let mut text = serde_json::to_string(&header).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn serve_dims_change_plan() {
    let dir = scratch(&format!("watch-dims-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model_dir = dir.join("model");
    let recipe_dir = dir.join("recipe");
    write_tiny_qwen2(&model_dir);
    std::fs::create_dir_all(&recipe_dir).unwrap();
    std::fs::write(
        recipe_dir.join("recipe.toml"),
        "name = \"tiny-dims\"\n[weights]\npath = \"../model\"\n[serve]\nslots = 5\nmax_model_len = 96\nmax_batch_rows = 12\n",
    )
    .unwrap();
    let settings = tessera_run::testing::host();
    let parts = Parts::even(1);
    let resolve = |extra: &[&str]| {
        let mut argv = vec!["kdir", "model", model_dir.to_str().unwrap()];
        argv.extend_from_slice(extra);
        let parsed = parse_watch(&args(&argv)).unwrap();
        let (opts, dims, _) = build_options(&parsed, &settings, true).unwrap();
        let rest = &parsed.subject_args[2..];
        let t = model_target(&model_dir, rest, &opts, dims, &[], &parsed, &parts)
            .unwrap()
            .unwrap();
        let bound = |name: DimName| {
            t.g.dims
                .iter()
                .find(|d| d.name == name)
                .map(|d| d.size)
                .unwrap()
        };
        let dev = Device::guess(&settings).with(&t.recipe);
        let plan = lower(&t.g, 0, &t.sizes, &dev).unwrap();
        let kept_bytes: usize = plan.buffers.iter().sum();
        (
            bound(DimName::Seqs),
            bound(DimName::Context),
            t.sizes,
            kept_bytes,
        )
    };

    // CLI --slots and --max-model-len populate graph bounds and lowered plan.
    let (s_small, c_small, sizes_small, bytes_small) =
        resolve(&["--slots", "2", "--max-model-len", "64", "Rows=4"]);
    let (s_big, c_big, sizes_big, bytes_big) =
        resolve(&["--slots", "6", "--max-model-len", "128", "Rows=4"]);
    assert_eq!(s_small, Size::Bounded(2));
    assert_eq!(c_small, Size::Bounded(64));
    assert_eq!(s_big, Size::Bounded(6));
    assert_eq!(c_big, Size::Bounded(128));
    assert!(sizes_small.contains(&(DimName::Seqs, 2)));
    assert!(sizes_small.contains(&(DimName::Context, 64)));
    assert!(sizes_big.contains(&(DimName::Seqs, 6)));
    assert!(sizes_big.contains(&(DimName::Context, 128)));
    assert!(bytes_big > bytes_small, "{bytes_big} <= {bytes_small}");

    // Recipe flags apply, CLI overrides recipe, positional Dim=value overrides CLI.
    let (s_rec, c_rec, _, _) = resolve(&["--recipe", recipe_dir.to_str().unwrap(), "Rows=4"]);
    assert_eq!((s_rec, c_rec), (Size::Bounded(5), Size::Bounded(96)));
    let (s_cli, c_cli, _, _) = resolve(&[
        "--recipe",
        recipe_dir.to_str().unwrap(),
        "--slots",
        "7",
        "--max-model-len",
        "160",
        "Rows=4",
    ]);
    assert_eq!((s_cli, c_cli), (Size::Bounded(7), Size::Bounded(160)));
    let (s_pos, c_pos, sizes_pos, _) = resolve(&[
        "--recipe",
        recipe_dir.to_str().unwrap(),
        "--slots",
        "7",
        "--max-model-len",
        "160",
        "Seqs=3",
        "Context=48",
        "Rows=4",
    ]);
    assert_eq!((s_pos, c_pos), (Size::Bounded(3), Size::Bounded(48)));
    assert!(sizes_pos.contains(&(DimName::Seqs, 3)));
    assert!(sizes_pos.contains(&(DimName::Context, 48)));
    let _ = std::fs::remove_dir_all(&dir);
}
