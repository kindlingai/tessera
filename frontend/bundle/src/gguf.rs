//! Qwen3 weights and tokenizer from one GGUF file.

use crate::{Bundle, Template};
use checkpoint::{Gguf, GgufType};
use graph::{Graph, PathSeg};
use interp::{Tensor, bind};
use lower::Pins;
use lower::form::{F32_FORM, Form, named};
use models::Stack;
use models::qwen3::{Config, stepper};
use serde_json::{Map, Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

type Failure = Box<dyn std::error::Error + Send + Sync>;

fn config(gguf: &Gguf, slots: u64, context: u64) -> Result<Config, Failure> {
    let integer = |key: &str| {
        gguf.u64(key)
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    let float = |key: &str| {
        gguf.f32(key)
            .filter(|n| n.is_finite() && *n > 0.)
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    let hidden = integer("qwen3.embedding_length")?;
    let vocab = match gguf.u64("qwen3.vocab_size") {
        Some(n) => n,
        None => gguf.elements("token_embd.weight")? as u64 / hidden,
    };
    let config = Config {
        stack: Stack {
            vocab,
            hidden,
            layers: integer("qwen3.block_count")?,
            heads: integer("qwen3.attention.head_count")?,
            head_dim: integer("qwen3.attention.key_length")?,
            ffn: integer("qwen3.feed_forward_length")?,
            max_seqs: slots,
            max_context: context,
        },
        kv_heads: integer("qwen3.attention.head_count_kv")?,
        rope_theta: float("qwen3.rope.freq_base")?,
        eps: float("qwen3.attention.layer_norm_rms_epsilon")?,
    };
    if vocab == 0
        || slots == 0
        || context == 0
        || !config.stack.heads.is_multiple_of(config.kv_heads)
    {
        return Err("invalid qwen3 dimensions".into());
    }
    Ok(config)
}

pub(crate) fn tensor_name(path: &[PathSeg]) -> Option<String> {
    use PathSeg::{Field, Index};
    Some(match path {
        [Field(f)] if f == "embed" => "token_embd.weight".into(),
        [Field(f)] if f == "norm" => "output_norm.weight".into(),
        [Field(root), Index(i), Field(f)] if root == "layers" => {
            let field = match f.as_str() {
                "ln1" => "attn_norm",
                "ln2" => "ffn_norm",
                "q_norm" => "attn_q_norm",
                "k_norm" => "attn_k_norm",
                "q" => "attn_q",
                "k" => "attn_k",
                "v" => "attn_v",
                "o" => "attn_output",
                "gate" => "ffn_gate",
                "up" => "ffn_up",
                "down" => "ffn_down",
                _ => return None,
            };
            format!("blk.{i}.{field}.weight")
        }
        _ => return None,
    })
}

fn form(ty: GgufType) -> &'static dyn Form {
    match ty {
        GgufType::Q8_0 => named("q8_0").expect("registered q8_0"),
        GgufType::Q4_0 => named("q4_0").expect("registered q4_0"),
        _ => F32_FORM,
    }
}

fn tensor(gguf: &Gguf, name: &str) -> Result<Tensor, checkpoint::Error> {
    match gguf.tensor_type(name)? {
        GgufType::F32 | GgufType::F16 => {
            let values = gguf.floats(name)?;
            Ok(Tensor::f32(vec![values.len()], values))
        }
        ty => {
            let (_, elements, bytes) = gguf.bytes(name)?;
            Ok(Tensor::packed(vec![elements], form(ty).name(), &bytes))
        }
    }
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

fn tokenizer(gguf: &Gguf) -> Result<Tokenizer, Failure> {
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
    Tokenizer::from_bytes(serde_json::to_vec(&data)?)
}

pub(crate) fn load(path: &Path, slots: u64, context: u64) -> Result<Bundle, Failure> {
    let gguf = Gguf::open(path)?;
    if gguf.str("general.architecture") != Some("qwen3") {
        return Err("GGUF architecture must be qwen3".into());
    }
    let config = config(&gguf, slots, context)?;
    if gguf.has("output.weight")
        && gguf.bytes("output.weight")? != gguf.bytes("token_embd.weight")?
    {
        return Err("untied output.weight is not supported for qwen3".into());
    }
    let graph = stepper(&config);
    let forms = graph
        .weights
        .iter()
        .map(|w| {
            let name = tensor_name(&w.path).ok_or("unknown qwen3 weight path")?;
            Ok::<_, Failure>(form(gguf.tensor_type(&name)?))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pins = pins(&graph, &forms, config.stack.layers);
    let failure = RefCell::new(None);
    let weights = bind(&graph, |path| match tensor(&gguf, &tensor_name(path)?) {
        Ok(tensor) => Some(tensor),
        Err(e) => {
            *failure.borrow_mut() = Some(e);
            None
        }
    });
    if let Some(e) = failure.into_inner() {
        return Err(e.into());
    }
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
        weights: weights?,
        tokenizer: tokenizer(&gguf)?,
        template,
        stop,
        pins: Some(pins),
    })
}
