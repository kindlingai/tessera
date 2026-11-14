//! One GGUF file: graph, weights, tokenizer and template.

use crate::names::Rule;
use crate::tokenize::hf::Hf;
use crate::{Bundle, Template};
use checkpoint::Gguf;
use graph::{Graph, PathSeg};
use lower::Pins;
use lower::form::{F32_FORM, Form, named};
use models::Stack;
use models::qwen3::{Config, stepper};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::Path;

type Failure = Box<dyn std::error::Error + Send + Sync>;

/// The stack's sizes, under the file's own architecture prefix.
fn config(gguf: &Gguf, arch: &str, slots: u64, context: u64) -> Result<Config, Failure> {
    let integer = |key: &str| {
        gguf.u64(&format!("{arch}.{key}"))
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("missing or invalid {arch}.{key}"))
    };
    let float = |key: &str| {
        gguf.f32(&format!("{arch}.{key}"))
            .filter(|n| n.is_finite() && *n > 0.)
            .ok_or_else(|| format!("missing or invalid {arch}.{key}"))
    };
    let hidden = integer("embedding_length")?;
    let vocab = match gguf.u64(&format!("{arch}.vocab_size")) {
        Some(n) => n,
        None => gguf.elements("token_embd.weight")? as u64 / hidden,
    };
    let config = Config {
        stack: Stack {
            vocab,
            hidden,
            layers: integer("block_count")?,
            heads: integer("attention.head_count")?,
            head_dim: integer("attention.key_length")?,
            ffn: integer("feed_forward_length")?,
            max_seqs: slots,
            max_context: context,
        },
        kv_heads: integer("attention.head_count_kv")?,
        rope_theta: float("rope.freq_base")?,
        eps: float("attention.layer_norm_rms_epsilon")?,
    };
    if vocab == 0
        || slots == 0
        || context == 0
        || !config.stack.heads.is_multiple_of(config.kv_heads)
    {
        return Err(format!("invalid {arch} dimensions").into());
    }
    Ok(config)
}

pub(crate) fn pins(graph: &Graph, forms: &[&'static dyn Form], layers: u64) -> Pins {
    let mut counts = BTreeMap::new();
    for f in forms {
        *counts.entry(f.name()).or_insert(0) += 1;
    }
    let base = counts
        .into_iter()
        .max_by_key(|(_, n)| *n)
        .and_then(|(name, _)| named(name))
        .unwrap_or(F32_FORM);
    let mut pins = Pins::all(base);
    let mut fields: BTreeMap<&str, Vec<(u64, &'static dyn Form)>> = BTreeMap::new();
    for (w, &f) in graph.weights.iter().zip(forms) {
        if let [
            PathSeg::Field(root),
            PathSeg::Index(i),
            PathSeg::Field(field),
        ] = w.path.as_slice()
            && root == "layers"
        {
            fields.entry(field).or_default().push((*i as u64, f));
        }
    }
    for (field, mut values) in fields {
        values.sort_by_key(|(i, _)| *i);
        let f = values[0].1;
        let uniform = values.len() as u64 == layers
            && values
                .iter()
                .enumerate()
                .all(|(at, (i, other))| *i == at as u64 && other.name() == f.name());
        if uniform && f.name() != base.name() {
            pins = pins.pin(&format!("layers.*.{field}"), f);
        }
    }
    for (w, &f) in graph.weights.iter().zip(forms) {
        if pins.form(&w.path).name() != f.name() {
            let segments: Vec<_> = w
                .path
                .iter()
                .map(|p| match p {
                    PathSeg::Field(s) => s.clone(),
                    PathSeg::Index(i) => i.to_string(),
                })
                .collect();
            pins = pins.pin(&segments.join("."), f);
        }
    }
    for (w, &f) in graph.weights.iter().zip(forms) {
        debug_assert_eq!(pins.form(&w.path), f);
    }
    pins
}

fn tokenizer(gguf: &Gguf) -> Result<Hf, Failure> {
    let tokens = gguf
        .strs("tokenizer.ggml.tokens")
        .ok_or("missing tokenizer tokens")?;
    let types = match gguf.meta("tokenizer.ggml.token_type") {
        Some(_) => gguf
            .u32s("tokenizer.ggml.token_type")
            .ok_or("invalid token types")?,
        None => vec![1; tokens.len()],
    };
    if tokens.len() != types.len() {
        return Err("token type count differs from token count".into());
    }
    let mut vocab = Map::new();
    let mut added = Vec::new();
    for (id, (token, ty)) in tokens.iter().zip(types).enumerate() {
        match ty {
            1 => {
                vocab.insert((*token).into(), json!(id));
            }
            3 | 4 => added.push(json!({
                "id": id, "content": token, "special": ty == 3,
                "single_word": false, "lstrip": false, "rstrip": false, "normalized": false,
            })),
            _ => {}
        }
    }
    let raw = gguf
        .strs("tokenizer.ggml.merges")
        .ok_or("missing tokenizer merges")?;
    let merges = raw
        .iter()
        .map(|s| {
            let (a, b) = s.split_once(' ').ok_or("invalid tokenizer merge")?;
            Ok::<_, Failure>(json!([a, b]))
        })
        .collect::<Result<Vec<Value>, _>>()?;
    let byte = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let data = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added, "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false}, byte.clone()
        ]},
        "post_processor": byte.clone(), "decoder": byte,
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges}
    });
    Hf::from_json(&serde_json::to_vec(&data)?)
}

pub(crate) fn load(
    path: &Path,
    slots: u64,
    context: u64,
    user: &[Rule],
) -> Result<Bundle, Failure> {
    let gguf = Gguf::open(path)?;
    let arch = gguf
        .str("general.architecture")
        .ok_or("a GGUF file names its architecture")?;
    // The one place a model is chosen.
    let (graph, layers) = match arch {
        "qwen3" => {
            let config = config(&gguf, arch, slots, context)?;
            (stepper(&config), config.stack.layers)
        }
        other => return Err(format!("{other} is not a model this engine traces").into()),
    };
    if gguf.has("output.weight")
        && gguf.bytes("output.weight")? != gguf.bytes("token_embd.weight")?
    {
        return Err("an untied output.weight is not supported".into());
    }
    let (weights, pins) = crate::pull(&graph, &gguf, layers, &[user])?;
    let template = gguf
        .str("tokenizer.chat_template")
        .map(|s| Template::new(s.to_owned()))
        .transpose()?;
    let mut stop = Vec::new();
    for key in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
        if let Some(id) = gguf.u64(key) {
            let id = u32::try_from(id)?;
            if !stop.contains(&id) {
                stop.push(id);
            }
        }
    }
    Ok(Bundle {
        graph,
        weights,
        pins: Some(pins),
        tokenizer: tokenizer(&gguf)?,
        template,
        stop,
    })
}
