use crate::mla::config;
use crate::mtp::rules;
use crate::names::resolve;
use tessera_checkpoint::{Slice, Stored};
use tessera_engine::PathSeg;
use tessera_engine::mla::{Moe, RotatedMoe};
use tessera_engine::mtp_model;

const GLM_CONFIG: &str = r#"{
  "vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 78,
  "num_attention_heads": 2, "q_lora_rank": 32, "kv_lora_rank": 32,
  "qk_nope_head_dim": 16, "qk_rope_head_dim": 16, "v_head_dim": 32,
  "intermediate_size": 128, "moe_intermediate_size": 128,
  "n_routed_experts": 2, "num_experts_per_tok": 2, "first_k_dense_replace": 3,
  "routed_scaling_factor": 2.5, "rope_parameters": {"rope_theta": 8000000},
  "rms_norm_eps": 1e-5, "index_n_heads": 2, "index_head_dim": 32,
  "index_topk": 4, "index_topk_freq": 4
}"#;

fn mock_slices(g: &tessera_engine::Graph, rules: &[crate::names::Rule]) -> Vec<Slice> {
    let bare = ["bias", ".su", ".sv", "signs_in", "signs_out"];
    let mut slices = Vec::new();
    for w in &g.weights {
        let path = PathSeg::dotted(&w.path);
        let (_, stem) = rules.iter().find(|(p, _)| *p == path).unwrap();
        let axis = |e: &tessera_engine::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let shape: Vec<usize> = w.ty.shape.0.iter().map(axis).collect();
        if stem.contains("{}") {
            for e in 0..shape[0] {
                let name = format!("{}.weight", stem.replace("{}", &e.to_string()));
                slices.push(Slice {
                    name,
                    shape: shape[1..].to_vec(),
                    stored: Stored::F32,
                });
            }
            continue;
        }
        let name = match bare.iter().any(|b| stem.ends_with(b)) {
            true => stem.clone(),
            false => format!("{stem}.weight"),
        };
        slices.push(Slice {
            name,
            shape,
            stored: Stored::F32,
        });
    }
    slices
}

#[test]
fn mtp_names_resolve() {
    let target = config(GLM_CONFIG, 1, 8).unwrap();
    let layer = target.stack.layers;
    let rot_cfg = mtp_model::config(&target, true);
    let rot_g = mtp_model::stepper::<RotatedMoe>(&rot_cfg);
    let rot_rules = rules(layer, true, true);
    let rot_slices = mock_slices(&rot_g, &rot_rules);
    let claims = resolve(&rot_g, &rot_slices, &[&rot_rules]).unwrap();
    assert_eq!(claims.len(), rot_g.weights.len());
    for expected in [
        "model.layers.78.eh_proj.weight",
        "model.layers.78.enorm.weight",
        "model.layers.78.hnorm.weight",
        "model.layers.78.shared_head.norm.weight",
        "model.layers.78.self_attn.indexer.wq_b.weight",
        "model.layers.78.mlp.experts.gate.weight",
    ] {
        assert!(rot_slices.iter().any(|s| s.name == expected), "{expected}");
    }
    let std_cfg = mtp_model::config(&target, false);
    let std_g = mtp_model::stepper::<Moe>(&std_cfg);
    let std_rules = rules(layer, false, false);
    let std_slices = mock_slices(&std_g, &std_rules);
    assert_eq!(
        resolve(&std_g, &std_slices, &[&std_rules]).unwrap().len(),
        std_g.weights.len()
    );
}
