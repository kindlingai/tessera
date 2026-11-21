use crate::Overrides;
use crate::names::{resolve, rules};
use serde_json::json;
use std::path::Path;
use tessera_checkpoint::{Shards, Slice, Stored};
use tessera_engine::decoder::{Config, Swiglu, stepper};
use tessera_engine::keys::Unresolved;
use tessera_engine::{Graph, Norm, PathSeg, Weight};

/// The toy decoder, with its own head.
fn graph() -> Graph {
    let mut c = Config::toy(1, 8);
    c.tied = false;
    stepper::<Norm, Swiglu>(&c)
}

/// Each field's Hugging Face and GGUF spellings.
const SPELLINGS: [(&str, &str, &str); 14] = [
    ("embed", "model.embed_tokens", "token_embd"),
    ("head", "lm_head", "output"),
    ("norm", "model.norm", "output_norm"),
    ("ln1", "input_layernorm", "attn_norm"),
    ("ln2", "post_attention_layernorm", "ffn_norm"),
    ("q", "self_attn.q_proj", "attn_q"),
    ("k", "self_attn.k_proj", "attn_k"),
    ("v", "self_attn.v_proj", "attn_v"),
    ("o", "self_attn.o_proj", "attn_output"),
    ("q_norm", "self_attn.q_norm", "attn_q_norm"),
    ("k_norm", "self_attn.k_norm", "attn_k_norm"),
    ("gate", "mlp.gate_proj", "ffn_gate"),
    ("up", "mlp.up_proj", "ffn_up"),
    ("down", "mlp.down_proj", "ffn_down"),
];

/// A weight path's name, Hugging Face's or GGUF's.
pub(crate) fn tensor_name(path: &[PathSeg], gguf: bool) -> Option<String> {
    use PathSeg::{Field, Index};
    let (field, layer) = match path {
        [Field(f)] => (f, None),
        [Field(layers), Index(i), .., Field(f)] if layers == "layers" => (f, Some(i)),
        _ => return None,
    };
    let (_, hf, ggml) = SPELLINGS.iter().find(|(name, ..)| name == field)?;
    let (name, blocks) = match gguf {
        true => (ggml, "blk"),
        false => (hf, "model.layers"),
    };
    Some(match layer {
        Some(i) => format!("{blocks}.{i}.{name}.weight"),
        None => format!("{name}.weight"),
    })
}

/// Every weight in one spelling, sized to fit.
fn offered(g: &Graph, gguf: bool) -> Vec<Slice> {
    let mut out = Vec::new();
    for w in &g.weights {
        let name = tensor_name(&w.path, gguf).unwrap();
        let axis = |e: &tessera_engine::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        out.push(Slice {
            name,
            shape: w.ty.shape.0.iter().map(axis).collect(),
            stored: Stored::F32,
        });
    }
    out
}

#[test]
fn both_spellings_resolve() {
    let g = graph();
    for gguf in [false, true] {
        let slices = offered(&g, gguf);
        let names: Vec<String> = resolve(&g, &slices, &[])
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        let want: Vec<String> = slices.iter().map(|s| s.name.clone()).collect();
        assert_eq!(names, want);
    }
}

#[test]
fn refuses_doubt() {
    let g = graph();
    let mut slices = offered(&g, false);
    let q = slices
        .iter()
        .find(|s| s.name.contains("0.self_attn.q_proj"))
        .unwrap();
    let extra = Slice {
        name: "model.layers.0.self_attn.q.weight".into(),
        ..q.clone()
    };
    slices.push(extra);
    slices.retain(|s| s.name != "model.norm.weight");
    let down = slices
        .iter_mut()
        .find(|s| s.name.contains("0.mlp.down"))
        .unwrap();
    down.shape.reverse();
    let Err(Unresolved {
        missing, ambiguous, ..
    }) = resolve(&g, &slices, &[])
    else {
        panic!("doubt refused");
    };
    assert_eq!(missing, ["layers.0.mlp.down", "norm"]);
    assert_eq!(ambiguous.len(), 1);
    assert!(ambiguous[0].0.ends_with(".q") && ambiguous[0].1.len() == 2);
}

#[test]
fn rules_decide() {
    let g = graph();
    let mut slices = offered(&g, false);
    slices
        .iter_mut()
        .for_each(|s| s.name = s.name.replace("q_proj", "query"));
    assert!(resolve(&g, &slices, &[]).is_err());
    let repo = rules(r#"{"layers.*.q": "model.layers.*.self_attn.query"}"#).unwrap();
    let names = resolve(&g, &slices, &[&repo]).unwrap();
    let q = "model.layers.0.self_attn.q_proj.weight";
    let at = g
        .weights
        .iter()
        .position(|w| tensor_name(&w.path, false).unwrap() == q);
    assert_eq!(
        names[at.unwrap()].name,
        "model.layers.0.self_attn.query.weight"
    );
    let user = rules(r#"{"layers.*.q": "elsewhere.*"}"#).unwrap();
    let failed = resolve(&g, &slices, &[&repo, &user]).unwrap_err();
    assert_eq!(failed.missing, ["layers.0.q", "layers.1.q"]);
    let wrong = rules(r#"{"norm": "model.layers.0.self_attn.query"}"#).unwrap();
    let failed = resolve(&g, &slices, &[&repo, &wrong]).unwrap_err();
    assert!(
        failed
            .to_string()
            .contains("norm: model.layers.0.self_attn.query.weight")
    );
}

/// `slices` as f32 safetensors. Each slice's values are its index.
fn written(dir: &Path, slices: &[Slice]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (at, s) in slices.iter().enumerate() {
        let from = data.len();
        for _ in 0..s.shape.iter().product::<usize>() {
            data.extend_from_slice(&(at as f32).to_le_bytes());
        }
        let entry = json!({"dtype": "F32", "shape": s.shape, "data_offsets": [from, data.len()]});
        header.insert(s.name.clone(), entry);
    }
    let mut text = serde_json::to_string(&header).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend(text.as_bytes());
    bytes.extend(data);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn untied_head_loads() {
    let dir = tessera_testkit::scratch(&format!("untied-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = json!({
        "model_type": "qwen3", "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 4, "head_dim": 8, "intermediate_size": 32,
        "num_key_value_heads": 2, "rope_theta": 1e4, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let (arch, c) = crate::config(&dir, [1, 8], &Overrides::default()).unwrap();
    let g = crate::stepper(&arch, &c).unwrap();
    written(&dir, &offered(&g, false));
    let shards = Shards::open(&dir).unwrap();
    let (weights, _) = crate::pull(&g, &shards, c.stack.layers, &[]).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let head = [PathSeg::Field("head".into())];
    let at = g.weights.iter().position(|w| w.path == head).unwrap();
    let Weight::Floats(t) = &weights[at] else {
        panic!("f32 slices load as floats");
    };
    // lm_head's values, not the embedding's.
    assert!(t.floats().unwrap().iter().all(|v| *v == at as f32));
}
