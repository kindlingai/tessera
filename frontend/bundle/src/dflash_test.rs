use crate::dflash::{Overlay, draft_pins, extra_rules, has_norms};
use crate::names;
use tessera_checkpoint::{Error, Offer, Slice, Stored};
use tessera_engine::dflash::{Config, DraftQuant, TargetSpec, default_layers, stepper};
use tessera_engine::keys::{Keys, Offered};
use tessera_engine::{Norm, PathSeg};

struct Mock(Vec<Slice>);

impl Offer for Mock {
    fn slices(&self) -> Vec<Slice> {
        self.0.clone()
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        Err(Error::Format(name.into()))
    }
    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        Err(Error::Format(name.into()))
    }
    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Result<Vec<u8>, Error> {
        Ok(format!("{name}:{axis}:{}:{}", range.0, range.1).into_bytes())
    }
}

fn sl(name: impl Into<String>, shape: &[usize]) -> Slice {
    Slice {
        name: name.into(),
        shape: shape.to_vec(),
        stored: Stored::Bf16,
    }
}

fn draft_slices(layers: usize) -> Vec<Slice> {
    let mut out = vec![
        sl("fc.weight", &[64, 128]),
        sl("hidden_norm.weight", &[64]),
        sl("norm.weight", &[64]),
    ];
    let groups: &[(&[usize], &[&str])] = &[
        (&[64], &["input_layernorm", "post_attention_layernorm"]),
        (&[16], &["self_attn.q_norm", "self_attn.k_norm"]),
        (&[64, 64], &["self_attn.q_proj", "self_attn.o_proj"]),
        (&[32, 64], &["self_attn.k_proj", "self_attn.v_proj"]),
        (&[128, 64], &["mlp.gate_proj", "mlp.up_proj"]),
        (&[64, 128], &["mlp.down_proj"]),
    ];
    for i in 0..layers {
        for (shape, stems) in groups {
            for stem in *stems {
                out.push(sl(format!("layers.{i}.{stem}.weight"), shape));
            }
        }
    }
    out
}

fn read_cfg(json: &str, target: &TargetSpec) -> Config {
    let offer = Offered::parse("config.json", json).unwrap();
    let mut keys = Keys::new(std::slice::from_ref(&offer));
    let cfg = Config::read(&mut keys, target, 1, 64);
    keys.finish().unwrap();
    cfg
}

fn bridge_slices() -> Vec<Slice> {
    let mut d = draft_slices(2);
    let extra: &[(&str, &[usize])] = &[
        ("codebook", &[256, 32]),
        ("gate.weight", &[16, 32]),
        ("gate.bias", &[16]),
        ("predecessor_gate.weight", &[16, 32]),
        ("prev.weight", &[16, 32]),
        ("early_read_weights", &[1, 32, 64]),
        ("read_weight", &[32, 64]),
        ("feature_write_weights", &[2, 64, 16]),
    ];
    for &(tail, shape) in extra {
        d.push(sl(format!("selected_prefix_bridge.{tail}"), shape));
    }
    d
}

#[test]
fn config_and_names() {
    assert_eq!(default_layers(36, 5), vec![1, 9, 17, 25, 33]);
    let target = TargetSpec {
        vocab: 256,
        hidden: 64,
        layers: 8,
        tied: false,
    };
    let json = r#"{"block_size":16,"hidden_size":64,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"rms_norm_eps":1e-6,"rope_theta":1e6,"dflash_config":{"mask_token_id":99,"target_layer_ids":[1,5]}}"#;
    let cfg = read_cfg(json, &target);
    assert_eq!((cfg.block_size, cfg.mask_token), (16, 99));
    assert_eq!((cfg.tap_layers(), cfg.tap_width()), (vec![2, 6], 128));
    let draft = draft_slices(2);
    assert!(has_norms(&draft));
    let tgt = vec![
        sl("model.embed_tokens.weight", &[256, 64]),
        sl("lm_head.weight", &[256, 64]),
        sl("model.norm.weight", &[64]),
    ];
    let (a, b) = (Mock(draft), Mock(tgt.clone()));
    let graph = stepper::<Norm>(&cfg);
    let claims = names::resolve(&graph, &Overlay::new(&a, &b).slices(), &[]).unwrap();
    assert_eq!(claims.len(), graph.weights.len());
    assert_eq!(claims[0].name, "model.embed_tokens.weight");
    assert_eq!(claims.last().unwrap().name, "lm_head.weight");

    let json2 = r#"{"block_size":8,"hidden_size":64,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"rms_norm_eps":1e-5,"rope_parameters":{"rope_theta":1e6},"is_causal":false,"use_sliding_window":true,"sliding_window":2048,"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":99,"selector_rank":16,"selector_top_k":4,"target_layer_ids":[1,5]}}"#;
    let cfg2 = read_cfg(json2, &target);
    assert!(!cfg2.causal && cfg2.sliding_window == Some(2048));
    let mut d2 = draft_slices(2);
    for i in 0..2 {
        for kind in ["attention_conv", "mlp_conv"] {
            d2.push(sl(format!("layers.{i}.{kind}.base_kernel"), &[2, 2, 64]));
            d2.push(sl(
                format!("layers.{i}.{kind}.kernel_projection.weight"),
                &[16, 64],
            ));
        }
    }
    d2.push(sl("candidate_selector.hidden_projection.weight", &[16, 64]));
    d2.push(sl("candidate_selector.predecessor_codebook", &[256, 16]));
    d2.push(sl("candidate_selector.successor_codebook", &[256, 16]));
    let g2 = stepper::<Norm>(&cfg2);
    let extra = extra_rules(&cfg2, &d2);
    let (a2, b2) = (Mock(d2), Mock(tgt.clone()));
    let claims2 = names::resolve(&g2, &Overlay::new(&a2, &b2).slices(), &[&extra]).unwrap();
    assert_eq!(claims2.len(), g2.weights.len());

    let json3 = r#"{"block_size":16,"hidden_size":64,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"rms_norm_eps":1e-6,"rope_theta":1e6,"dflash_config":{"draft_block_causal":true,"mask_token_id":99,"target_layer_ids":[1,5],"selected_prefix_code_bridge":{"code_size":32,"rank":16,"eps":1e-6},"selected_prefix_last_write_refine":{"head_topk":4}}}"#;
    let cfg3 = read_cfg(json3, &target);
    assert!(cfg3.causal && cfg3.bridge.is_some());
    let d3 = bridge_slices();
    let g3 = stepper::<Norm>(&cfg3);
    let extra3 = extra_rules(&cfg3, &d3);
    let (a3, b3) = (Mock(d3), Mock(tgt));
    let overlay3 = Overlay::new(&a3, &b3);
    let claims3 = names::resolve(&g3, &overlay3.slices(), &[&extra3]).unwrap();
    assert_eq!(claims3.len(), g3.weights.len());
    assert_eq!(
        overlay3.packed_axis("fc.weight", 0, (0, 32)).unwrap(),
        b"fc.weight:0:0:32"
    );
    assert_eq!(
        overlay3.packed_axis("lm_head.weight", 0, (64, 64)).unwrap(),
        b"lm_head.weight:0:64:64"
    );
}

fn mxfp8_slices() -> (Vec<Slice>, Vec<Slice>) {
    let draft = vec![
        sl("fc.weight", &[512, 1024]),
        sl("hidden_norm.weight", &[512]),
        sl("norm.weight", &[512]),
        sl("layers.0.input_layernorm.weight", &[512]),
        sl("layers.0.post_attention_layernorm.weight", &[512]),
        sl("layers.0.self_attn.q_norm.weight", &[128]),
        sl("layers.0.self_attn.k_norm.weight", &[128]),
        sl("layers.0.self_attn.q_proj.weight", &[512, 512]),
        sl("layers.0.self_attn.k_proj.weight", &[256, 512]),
        sl("layers.0.self_attn.v_proj.weight", &[256, 512]),
        sl("layers.0.self_attn.o_proj.weight", &[512, 512]),
        sl("layers.0.mlp.gate_proj.weight", &[1024, 512]),
        sl("layers.0.mlp.up_proj.weight", &[1024, 512]),
        sl("layers.0.mlp.down_proj.weight", &[512, 1024]),
        sl("layers.0.attention_conv.base_kernel", &[2, 2, 512]),
        sl(
            "layers.0.attention_conv.kernel_projection.weight",
            &[32, 512],
        ),
        sl("layers.0.mlp_conv.base_kernel", &[2, 2, 512]),
        sl("layers.0.mlp_conv.kernel_projection.weight", &[32, 512]),
        sl("candidate_selector.hidden_projection.weight", &[16, 512]),
        sl("candidate_selector.predecessor_codebook", &[64, 16]),
        sl("candidate_selector.successor_codebook", &[64, 16]),
    ];
    let tgt = vec![
        sl("model.embed_tokens.weight", &[64, 512]),
        sl("lm_head.weight", &[64, 512]),
    ];
    (draft, tgt)
}

#[test]
fn mxfp8_quant() {
    let target = TargetSpec {
        vocab: 64,
        hidden: 512,
        layers: 8,
        tied: false,
    };
    let json = r#"{"block_size":8,"hidden_size":512,"intermediate_size":1024,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":128,"num_hidden_layers":1,"rms_norm_eps":1e-6,"rope_theta":1e6,"dflash_config":{"block_size":8,"conv_group_size":64,"conv_kernel_size":2,"mask_token_id":0,"selector_rank":16,"selector_top_k":4,"target_layer_ids":[1,5]}}"#;
    let cfg = read_cfg(json, &target);
    let (draft, tgt) = mxfp8_slices();
    let g = stepper::<Norm>(&cfg);
    let extra = extra_rules(&cfg, &draft);
    let (a, b) = (Mock(draft), Mock(tgt));
    let ov = Overlay::new(&a, &b);
    let pins = draft_pins(&g, &ov, cfg.stack.layers, &[&extra], DraftQuant::MxFp8).unwrap();
    let layer_mx = [
        "head",
        "layers.0.q",
        "layers.0.k",
        "layers.0.v",
        "layers.0.o",
        "layers.0.mlp.gate",
        "layers.0.mlp.up",
        "layers.0.mlp.down",
        "layers.0.attention_conv.kernel_projection",
        "layers.0.mlp_conv.kernel_projection",
    ];
    for w in &g.weights {
        let dot = PathSeg::dotted(&w.path);
        let want = if layer_mx.contains(&dot.as_str()) {
            "mxfp8"
        } else {
            "bf16"
        };
        assert_eq!(pins.form(&w.path).name(), want, "weight {dot}");
    }
    let all_bf16 = draft_pins(&g, &ov, cfg.stack.layers, &[&extra], DraftQuant::Bf16).unwrap();
    assert!(
        g.weights
            .iter()
            .all(|w| all_bf16.form(&w.path).name() == "bf16")
    );
    let all_mx = draft_pins(&g, &ov, cfg.stack.layers, &[&extra], DraftQuant::MxFp8All).unwrap();
    assert_eq!(all_mx.form(&g.weights[1].path).name(), "mxfp8");
}
