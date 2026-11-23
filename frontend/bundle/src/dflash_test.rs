use crate::dflash::{Overlay, extra_rules, has_norms};
use crate::names;
use tessera_checkpoint::{Error, Offer, Slice, Stored};
use tessera_engine::Norm;
use tessera_engine::dflash::{Config, TargetSpec, default_layers, stepper};
use tessera_engine::keys::{Keys, Offered};

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

#[test]
fn config_and_names() {
    assert_eq!(default_layers(36, 5), vec![1, 9, 17, 25, 33]);
    let json = r#"{"block_size":16,"hidden_size":64,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"rms_norm_eps":1e-6,"rope_theta":1e6,"dflash_config":{"mask_token_id":99,"target_layer_ids":[1,5]}}"#;
    let target = TargetSpec {
        vocab: 256,
        hidden: 64,
        layers: 8,
        tied: false,
    };
    let offer = Offered::parse("config.json", json).unwrap();
    let mut keys = Keys::new(std::slice::from_ref(&offer));
    let cfg = Config::read(&mut keys, &target, 1, 64);
    keys.finish().unwrap();
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
    let offer2 = Offered::parse("config.json", json2).unwrap();
    let mut keys2 = Keys::new(std::slice::from_ref(&offer2));
    let cfg2 = Config::read(&mut keys2, &target, 1, 64);
    keys2.finish().unwrap();
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
    let (a2, b2) = (Mock(d2), Mock(tgt));
    let claims2 = names::resolve(&g2, &Overlay::new(&a2, &b2).slices(), &[&extra]).unwrap();
    assert_eq!(claims2.len(), g2.weights.len());
}
