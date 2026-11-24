use crate::mla::config;
use crate::mtp::{mtp_pins, rules};
use crate::names::resolve;
use crate::repack::Streamed;
use tessera_checkpoint::{Error, Offer, Slice, Stored};
use tessera_engine::block::MXFP8;
use tessera_engine::form::named;
use tessera_engine::mla::{Moe, RotatedMoe};
use tessera_engine::{PathSeg, Weight, mtp_model};

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

struct MockOffer(Vec<Slice>);

impl Offer for MockOffer {
    fn slices(&self) -> Vec<Slice> {
        self.0.clone()
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let s = self.0.iter().find(|s| s.name == name).unwrap();
        Ok(vec![0.0; s.elements()])
    }
    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        let s = self.0.iter().find(|s| s.name == name).unwrap();
        let Stored::Form(f) = s.stored else {
            return Err(Error::Format(name.into()));
        };
        Ok(vec![0u8; f.bytes(s.elements())])
    }
    fn bf16(&self, name: &str) -> Result<Vec<u8>, Error> {
        let s = self.0.iter().find(|s| s.name == name).unwrap();
        Ok(vec![0u8; s.elements() * 2])
    }
}

fn mock_slices(
    g: &tessera_engine::Graph,
    rules: &[crate::names::Rule],
    stored: Stored,
    head_stored: Stored,
) -> Vec<Slice> {
    let bare = ["bias", ".su", ".sv", "signs_in", "signs_out"];
    let mut slices = Vec::new();
    for w in &g.weights {
        let path = PathSeg::dotted(&w.path);
        let (_, stem) = rules.iter().find(|(p, _)| *p == path).unwrap();
        let axis = |e: &tessera_engine::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let shape: Vec<usize> = w.ty.shape.0.iter().map(axis).collect();
        let kind = if path == "head" { head_stored } else { stored };
        if stem.contains("{}") {
            for e in 0..shape[0] {
                let name = format!("{}.weight", stem.replace("{}", &e.to_string()));
                slices.push(Slice {
                    name,
                    shape: shape[1..].to_vec(),
                    stored: kind,
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
            stored: kind,
        });
    }
    slices
}

fn check_head_mxfp8() {
    let head_path = [PathSeg::Field("head".into())];
    let embed_path = [PathSeg::Field("embed".into())];
    let router_path = [
        PathSeg::Field("moe".into()),
        PathSeg::Field("router".into()),
    ];
    let mut target = config(GLM_CONFIG, 1, 8).unwrap();
    let layer = target.stack.layers;
    let g_small = mtp_model::stepper::<RotatedMoe>(&mtp_model::config(&target, true));
    let r = rules(layer, true, true);
    let native = MockOffer(mock_slices(
        &g_small,
        &r,
        Stored::Bf16,
        Stored::Form(&MXFP8),
    ));
    let native_pins = mtp_pins(&g_small, &native, &[&r]).unwrap();
    assert_eq!(native_pins.form(&head_path).name(), "mxfp8");
    assert_eq!(native_pins.form(&embed_path).name(), "bf16");
    let small_bf16 = MockOffer(mock_slices(&g_small, &r, Stored::Bf16, Stored::Bf16));
    assert_eq!(
        mtp_pins(&g_small, &small_bf16, &[&r])
            .unwrap()
            .form(&head_path)
            .name(),
        "bf16"
    );
    target.stack.hidden = 512;
    let g_wide = mtp_model::stepper::<RotatedMoe>(&mtp_model::config(&target, true));
    let wide_bf16 = MockOffer(mock_slices(&g_wide, &r, Stored::Bf16, Stored::Bf16));
    let wide_pins = mtp_pins(&g_wide, &wide_bf16, &[&r]).unwrap();
    assert_eq!(wide_pins.form(&head_path).name(), "mxfp8");
    assert_eq!(wide_pins.form(&embed_path).name(), "bf16");
    assert_eq!(wide_pins.form(&router_path).name(), "bf16");
    let head_at = g_wide
        .weights
        .iter()
        .position(|w| w.path == head_path)
        .unwrap() as u32;
    let mw = crate::Weights::Streamed(Box::new(Streamed {
        graph: g_wide,
        offer: Box::new(wide_bf16),
        rules: r,
        pins: wide_pins,
    }));
    let mut got_head = None;
    mw.each_share(
        |at| (at == head_at).then_some((0, 0, 32)),
        |at, w| -> Result<(), Error> {
            if at == head_at
                && let Weight::Packed(f, bytes) = w
            {
                got_head = Some((f.name(), bytes.len()));
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        got_head,
        Some(("mxfp8", 32 * named("mxfp8").unwrap().bytes(512)))
    );
}

#[test]
fn mtp_names_resolve() {
    let target = config(GLM_CONFIG, 1, 8).unwrap();
    let layer = target.stack.layers;
    let rot_cfg = mtp_model::config(&target, true);
    let rot_g = mtp_model::stepper::<RotatedMoe>(&rot_cfg);
    let rot_rules = rules(layer, true, true);
    let rot_slices = mock_slices(&rot_g, &rot_rules, Stored::F32, Stored::F32);
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
    let std_slices = mock_slices(&std_g, &std_rules, Stored::F32, Stored::F32);
    assert_eq!(
        resolve(&std_g, &std_slices, &[&std_rules]).unwrap().len(),
        std_g.weights.len()
    );
    check_head_mxfp8();
}
