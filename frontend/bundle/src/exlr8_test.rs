use crate::arch::Ask;
use crate::mla::{config, rules};
use crate::names::resolve;
use crate::qwen4_exp::{parse_config, ple, rules as qwen4_rules};
use crate::{Overrides, Stream, Weights};
use std::collections::HashMap;
use tessera_checkpoint::{Slice, Stored};
use tessera_engine::PathSeg;
use tessera_engine::Weight;
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

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A two-layer `qwen4_exp` config with PLE on the second layer.
const QWEN4_EXP: &str = r#"{
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "model_type": "qwen4_exp",
  "text_config": {
    "vocab_size": 16, "hidden_size": 16, "num_hidden_layers": 2,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
    "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
    "partial_rotary_factor": 0.5,
    "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.5},
    "rms_norm_eps": 1e-6, "hc_count": 4, "hc_lowrank": 8,
    "num_experts": 4, "num_experts_per_tok": 2,
    "ple_layer_ids": [2], "ple_embed_dim": 12, "ple_conv_kernel_size": 4,
    "ngram_size": 4, "heads_per_ngram": 2,
    "eos_token_id": 15, "hidden_act": "silu", "attention_bias": false,
    "output_gate_type": "sigmoid", "tie_word_embeddings": false
  }
}"#;

/// Integer tensors by name, and no weights.
struct Ints(HashMap<String, Vec<i64>>);

impl Stream for Ints {
    fn stream(
        &self,
        _: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        _: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.0.get(name).ok_or(name)?.clone())
    }
}

#[test]
fn config_reads_ple() {
    let c = parse_config(QWEN4_EXP, [1, 16]).unwrap();
    let p = c.ple.unwrap();
    assert_eq!((p.layer, p.dim, p.dilation, p.heads), (1, 12, 4, 6));
    let rules = qwen4_rules(&c, false);
    assert!(rules.iter().any(|(from, _)| from == "layers.1.ple.conv"));
    assert!(
        !rules
            .iter()
            .any(|(from, _)| from.starts_with("layers.0.ple"))
    );
}

#[test]
fn ple_sizes_from_config() {
    let dir = tessera_testkit::scratch(&format!("{}-qwen4-exp-ple", std::process::id()));
    std::fs::create_dir_all(dir.join("ple")).unwrap();
    std::fs::write(dir.join("ple/exlr8-ple-L001.bin"), [0u8; 4]).unwrap();
    let overrides = Overrides::default();
    let ask = Ask::plain(&dir, [1, 16], &overrides, QWEN4_EXP);
    let base = "model.language_model.layers.1.ple.ple_embedding";
    let ints = |multipliers: usize| {
        let mut by = HashMap::new();
        by.insert(format!("{base}.layer_multipliers"), vec![3; multipliers]);
        by.insert(format!("{base}.ngram_heads_offsets"), (0..6).collect());
        by.insert(format!("{base}.ngram_heads_vocab_sizes"), vec![1; 6]);
        Weights::Streamed(Box::new(Ints(by)))
    };
    let got = ple(&ask, &ints(4)).unwrap().unwrap();
    assert_eq!((got.heads(), got.row_dim, got.multipliers.len()), (6, 2, 4));
    assert!(ple(&ask, &ints(3)).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
