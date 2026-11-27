use super::*;

const TEXT: &str = r#"{
    "model_type": "inkling2_loop",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "text_config": {
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "log_scaling_n_floor": 8192,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "intermediate_size": 1152,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64}
    }
}"#;

#[test]
fn parses_inkling2_config() {
    let cfg = parse_config(TEXT, [1, 16]).unwrap();
    assert_eq!(cfg.n_virtual_layers(), 48);
    assert_eq!(cfg.log_scaling_n_floor, 8192.0);
    assert_eq!(cfg.global.iter().filter(|g| **g).count(), 4);
    assert!(cfg.global[5] && cfg.global[23]);
    assert_eq!((cfg.loop_start, cfg.loop_end, cfg.gate_rank), (6, 18, 64));
    assert!(crate::arch::find("inkling2").is_some());
}

#[test]
fn turns_lag_layouts() {
    let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
    assert_eq!(Turn::Swap.apply([2, 3], &x), [0., 3., 1., 4., 2., 5.]);
    assert_eq!(Turn::SwapFlip.apply([2, 3], &x), [3., 0., 4., 1., 5., 2.]);
    assert_eq!(Turn::SwapFlip.shape(&[2, 3]), [3, 2]);
    let rows: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let flipped = Turn::FlipRows(2).apply([4, 2], &rows);
    assert_eq!(flipped, [2., 3., 0., 1., 6., 7., 4., 5.]);
}

/// Needs `TESSERA_INKLING2_DIR` set to a released directory.
#[test]
#[ignore]
#[expect(
    clippy::disallowed_methods,
    reason = "the release directory comes from the caller"
)]
fn release_rules_resolve() {
    let dir = std::env::var("TESSERA_INKLING2_DIR").unwrap();
    let dir = Path::new(&dir);
    let json = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let cfg = parse_config(&json, [1, 16]).unwrap();
    let manifest = std::env::var("TESSERA_INKLING2_MANIFEST").ok();
    let slices = open(dir, manifest.as_deref()).unwrap().slices();
    let g = stepper::<Coded>(&cfg);
    let claims = crate::names::resolve(&g, &slices, &[&rules(&cfg, true)]).unwrap();
    assert_eq!(claims.len(), g.weights.len());
}

/// Stored slices of a small release, in checkpoint layout.
const H: usize = 256;
const F: usize = 128;

fn small_slices(cfg: &Config, rotated: bool) -> Vec<Slice> {
    let mut out = Vec::new();
    let mut s = |name: String, shape: &[usize]| {
        out.push(Slice {
            name,
            shape: shape.to_vec(),
            stored: Stored::F32,
        })
    };
    let top = "model.llm";
    s(format!("{top}.embed.weight"), &[64, H]);
    s(format!("{top}.unembed.weight"), &[64, H]);
    s(format!("{top}.embed_norm.weight"), &[H]);
    s(format!("{top}.norm.weight"), &[H]);
    s(format!("{top}.attn_res.output_query"), &[H]);
    s(format!("{top}.middle_cycle_cross_pass_queries"), &[8, H]);
    s(format!("{top}.middle_cycle_cross_pass_kv_queries"), &[8, H]);
    for l in 0..8 {
        let at = format!("{top}.layers.{l}");
        let global = cfg.global[l];
        s(format!("{top}.attn_res.mlp_queries.{l}"), &[H]);
        if l > 0 {
            let name = format!("{top}.attn_res.attn_queries.{}", l - 1);
            s(name, if global { &[3, H] } else { &[H] });
        }
        for (name, shape) in [
            ("attn.wq_du.weight", &[16, H][..]),
            ("attn.wk_dv.weight", &[8, H]),
            ("attn.wv_dv.weight", &[8, H]),
            ("attn.wo_ud.weight", &[H, 16]),
            ("attn.wr_du.weight", &[16, H]),
            ("attn.gate_attn.weight", &[16, H]),
            ("attn.attn_sink_proj.weight", &[4, H]),
            ("attn.q_l2_clip.scale", &[4, 4]),
            ("attn.rel_l2_clip.scale", &[4, 4]),
            ("attn.rel_logits_proj.proj", &[4, 8]),
            ("attn_sconv.weight", &[H, 1, 4]),
            ("mlp_sconv.weight", &[H, 1, 4]),
            ("mlp_norm.weight", &[H]),
            ("mlp.gate.weight", &[4, H]),
            ("mlp.gate.bias", &[4]),
        ] {
            s(format!("{at}.{name}"), shape);
        }
        for kv in ["k", "v"] {
            if global {
                s(format!("{at}.{kv}_input_norm.weight"), &[H]);
                s(format!("{at}.attn.{kv}_sconv.static_w"), &[4, 8]);
                s(format!("{at}.attn.{kv}_sconv.weight_proj"), &[8, H]);
            } else {
                s(format!("{at}.attn.{kv}_sconv.weight"), &[8, 1, 4]);
            }
        }
        let norm = if global { "q_input_norm" } else { "attn_norm" };
        s(format!("{at}.{norm}.weight"), &[H]);
        if (2..6).contains(&l) {
            for gate in ["attn", "k", "v", "mlp"] {
                let ctrl = format!("{at}.controllers.{gate}");
                s(format!("{ctrl}.down_weight"), &[4, 2 * H]);
                s(format!("{ctrl}.forget_weight"), &[H, 4]);
                s(format!("{ctrl}.gain_weight"), &[H, 4]);
            }
        }
        let experts = format!("{at}.mlp.experts");
        for (p, [rows, cols]) in [("gate", [F, H]), ("up", [F, H]), ("down", [H, F])] {
            if rotated {
                s(format!("{experts}.{p}.weight"), &[4, rows, cols]);
                s(format!("{experts}.{p}.su"), &[4, cols]);
                s(format!("{experts}.{p}.sv"), &[4, rows]);
            } else {
                for e in 0..4 {
                    s(format!("{experts}.{e}.{p}_proj.weight"), &[rows, cols]);
                }
            }
        }
        if rotated {
            s(format!("{experts}.signs_in"), &[H]);
            s(format!("{experts}.signs_out"), &[H]);
        }
    }
    let turned = turns(&out);
    assert_eq!(turned.len(), 8 + 4 * 4);
    turn_slices(out, &turned)
}

#[test]
fn small_rules_resolve() {
    let mut cfg = Config::small();
    (cfg.stack.hidden, cfg.stack.ffn) = (256, 128);
    for (g, rotated) in [
        (stepper::<Coded>(&cfg), true),
        (stepper::<Plain>(&cfg), false),
    ] {
        let slices = small_slices(&cfg, rotated);
        let claims = crate::names::resolve(&g, &slices, &[&rules(&cfg, rotated)]).unwrap();
        assert_eq!(claims.len(), g.weights.len());
    }
}
