use crate::mla::{config, rules};
use crate::names::resolve;
use tessera_checkpoint::{Slice, Stored};
use tessera_engine::PathSeg;
use tessera_engine::mla::{RotatedMoe, program};

/// Two layers, one routed with two experts, both indexing.
const CONFIG: &str = r#"{
  "vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 2,
  "num_attention_heads": 2, "q_lora_rank": 32, "kv_lora_rank": 32,
  "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 32,
  "intermediate_size": 128, "moe_intermediate_size": 128,
  "n_routed_experts": 2, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
  "routed_scaling_factor": 2.5, "rope_parameters": {"rope_theta": 8000000},
  "rms_norm_eps": 1e-5, "index_n_heads": 2, "index_head_dim": 32,
  "index_topk": 4, "index_topk_freq": 1
}"#;

/// Every weight finds the slice an exlr8 release names.
#[test]
fn rotated_names_resolve() {
    let c = config(CONFIG, 1, 8).unwrap();
    let g = program::<RotatedMoe>(&c);
    let rules = rules(&c, true);
    let mut slices = Vec::new();
    for w in &g.weights {
        let seg = |s: &PathSeg| match s {
            PathSeg::Field(f) => f.clone(),
            PathSeg::Index(i) => i.to_string(),
        };
        let mut path: Vec<String> = w.path.iter().map(seg).collect();
        let part = path
            .last()
            .filter(|p| ["w", "suh", "svh"].contains(&p.as_str()))
            .cloned();
        if part.is_some() {
            path.pop();
        }
        let (_, stem) = rules.iter().find(|(p, _)| *p == path.join(".")).unwrap();
        let bare = ["bias", ".su", ".sv", "signs_in", "signs_out"];
        let name = match part.as_deref() {
            Some("w") => format!("{stem}.codes"),
            Some(end) => format!("{stem}.{end}"),
            None if bare.iter().any(|b| stem.ends_with(b)) => stem.clone(),
            None => format!("{stem}.weight"),
        };
        let axis = |e: &tessera_engine::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let shape = w.ty.shape.0.iter().map(axis).collect();
        slices.push(Slice {
            name,
            shape,
            stored: Stored::F32,
        });
    }
    let claims = resolve(&g, &slices, &[&rules]).unwrap();
    assert_eq!(claims.len(), g.weights.len());
    let at = "model.layers.1.mlp.";
    assert!(
        slices
            .iter()
            .any(|s| s.name == format!("{at}experts.down.su"))
    );
    assert!(
        slices
            .iter()
            .any(|s| s.name == format!("{at}shared_experts.up_proj.weight"))
    );
}

use serde_json::json;
use std::path::Path;
use tessera_engine::storage::Form;
use tessera_engine::{Bf16, E4M3, Engine, Pick, PleStore, Recipe, Row, Settings, block};

fn write_nvfp4_qwen4(dir: &Path) {
    let cfg = r#"{"model_type":"qwen4_exp","vocab_size":32,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"moe_intermediate_size":16,"shared_expert_intermediate_size":16,"num_experts":2,"num_experts_per_tok":1,"hc_count":4,"hc_lowrank":8,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"layer_types":["linear_attention","full_attention"],"partial_rotary_factor":0.5,"rope_theta":10000.0,"rms_norm_eps":1e-6,"ple_layer_ids":[2],"ple_embed_dim":32,"ple_conv_kernel_size":4,"ngram_size":3,"eos_token_id":15,"tie_word_embeddings":false}"#;
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    let offered = [tessera_engine::keys::Offered::parse("c", cfg).unwrap()];
    let mut keys = tessera_engine::keys::Keys::new(&offered);
    let c = tessera_engine::qwen4_exp::Config::read(&mut keys, 1, 8);
    let g = tessera_engine::qwen4_exp::stepper::<tessera_engine::qwen4_exp::Moe>(&c);
    let rules = crate::qwen4_exp::rules(&c, false);
    let (mut hdr, mut data) = (serde_json::Map::new(), Vec::new());
    let mut add = |k: String, dt: &str, sh: Vec<usize>, b: &[u8]| {
        let a = data.len();
        data.extend_from_slice(b);
        hdr.insert(
            k,
            json!({"dtype": dt, "shape": sh, "data_offsets": [a, data.len()]}),
        );
    };
    for (idx, w) in g.weights.iter().enumerate() {
        let path = PathSeg::dotted(&w.path);
        let stem = rules.iter().find(|(p, _)| *p == path).unwrap().1.clone();
        let sh: Vec<usize> =
            w.ty.shape
                .0
                .iter()
                .map(|e| e.value(&g.dims, &|_| None).unwrap() as usize)
                .collect();
        let len: usize = sh.iter().product();
        let vals: Vec<f32> = (0..len)
            .map(|i| (((i + idx) % 15) as f32 - 7.0) * 0.02)
            .collect();
        if stem.contains(".mlp.experts.{}.") {
            for e in 0..sh[0] {
                let p = block::NVFP4.pack(&vals[e * sh[1] * sh[2]..(e + 1) * sh[1] * sh[2]]);
                let (mut s, mut q) = (Vec::new(), Vec::new());
                for blk in p[4..].as_chunks::<9>().0 {
                    s.push(blk[0]);
                    q.extend_from_slice(&blk[1..]);
                }
                let k = stem.replace("{}", &e.to_string());
                add(format!("{k}.weight"), "U8", vec![sh[1], sh[2] / 2], &q);
                add(
                    format!("{k}.weight_scale"),
                    "F8_E4M3",
                    vec![sh[1], sh[2] / 16],
                    &s,
                );
                add(format!("{k}.weight_scale_2"), "F32", vec![], &p[..4]);
            }
        } else if stem.contains(".self_attn.") && sh.len() == 2 && sh[1].is_multiple_of(32) {
            let p = block::MXFP8.pack(&vals);
            let (mut s, mut q) = (Vec::new(), Vec::new());
            for blk in p.as_chunks::<33>().0 {
                s.push(blk[0]);
                q.extend_from_slice(&blk[1..]);
            }
            add(format!("{stem}.weight"), "F8_E4M3", sh.clone(), &q);
            add(
                format!("{stem}.weight_scale"),
                "U8",
                vec![sh[0], sh[1] / 32],
                &s,
            );
        } else {
            let b: Vec<u8> = vals
                .iter()
                .flat_map(|&v| Bf16::from(v).0.to_le_bytes())
                .collect();
            add(format!("{stem}.weight"), "BF16", sh, &b);
        }
    }
    let pb = "model.language_model.layers.1.ple.ple_embedding";
    let ple: Vec<u8> = (0..256)
        .map(|i| E4M3::from((i % 9) as f32 * 0.1).0)
        .collect();
    add(
        format!("{pb}.ngram_embedding.weight"),
        "F8_E4M3",
        vec![128, 2],
        &ple,
    );
    add(
        format!("{pb}.ngram_embedding.weight_scale"),
        "BF16",
        vec![1],
        &Bf16::from(0.25).0.to_le_bytes(),
    );
    let i64b = |xs: &[i64]| xs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    add(
        format!("{pb}.layer_multipliers"),
        "I64",
        vec![3],
        &i64b(&[100003, 200003, 300007]),
    );
    let offs: Vec<i64> = (0..16).map(|h| h * 8).collect();
    add(
        format!("{pb}.ngram_heads_offsets"),
        "I64",
        vec![16],
        &i64b(&offs),
    );
    add(
        format!("{pb}.ngram_heads_vocab_sizes"),
        "I64",
        vec![16],
        &i64b(&[8; 16]),
    );
    let text = serde_json::to_string(&hdr).unwrap();
    let mut raw = (text.len() as u64).to_le_bytes().to_vec();
    raw.extend_from_slice(text.as_bytes());
    raw.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), raw).unwrap();
}

#[test]
fn nvfp4_qwen4_loads() {
    let dir = tessera_testkit::scratch(&format!("qwen4-nvfp4-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_nvfp4_qwen4(&dir);
    let b = crate::load(&dir, 1, 8).unwrap();
    let pins = b.pins.clone().unwrap();
    let pin = |n: &str| {
        let w = b
            .graph
            .weights
            .iter()
            .find(|w| PathSeg::dotted(&w.path) == n)
            .unwrap();
        pins.form(&w.path).name()
    };
    assert_eq!(
        (pin("layers.0.moe.gate"), pin("layers.1.attn.q")),
        ("nvfp4", "mxfp8")
    );
    let ple_cfg = b.ple.unwrap();
    assert_eq!(ple_cfg.fp8_scale, Some(0.25));
    let recipe = Recipe {
        pins,
        ..Recipe::default()
    };
    let settings = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let mut eng = Engine::open(b.graph, 8, &recipe, &settings).unwrap();
    eng.set_ple(PleStore::open(ple_cfg).unwrap());
    b.weights.each(|at, w| eng.load_weight(at, w)).unwrap();
    let tok = Row {
        slot: 0,
        token: 1,
        position: 0,
    };
    assert!(eng.step(&[tok]).unwrap()[0] < 32);
    let _ = std::fs::remove_dir_all(&dir);
}
