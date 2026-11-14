//! A Hugging Face model directory: graph, weights, tokenizer, template.

mod gguf;
#[cfg(test)]
mod gguf_test;
pub mod tag;
mod template;
#[cfg(test)]
mod template_test;
pub mod tokenize;

pub use template::{Message, Template};

use checkpoint::File;
use graph::Graph;
use interp::bind;
use lower::{Pins, Weight};
use models::Stack;
use models::qwen3::{Config, stepper, tensor_name};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tokenize::hf::Hf;

/// What the graph needs from a Hugging Face `config.json`.
#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    vocab_size: u64,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    intermediate_size: u64,
    rope_theta: f32,
    rms_norm_eps: f32,
}

/// One id or a list, as `generation_config.json` writes it.
#[derive(Deserialize)]
#[serde(untagged)]
enum Ids {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Deserialize)]
struct Generation {
    eos_token_id: Option<Ids>,
}

pub struct Bundle {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    /// Forms packed in the file, when any are.
    pub pins: Option<Pins>,
    pub tokenizer: Hf,
    pub template: Option<Template>,
    /// Ids that end an answer.
    pub stop: Vec<u32>,
}

/// Reads a Qwen3 directory: `slots` sequences of `context` rows.
pub fn load(dir: &Path, slots: u64, context: u64) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    if dir.is_file() {
        return gguf::load(dir, slots, context);
    }
    let hf: HfConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    if hf.model_type != "qwen3" {
        return Err(format!("{} is not a model this engine traces", hf.model_type).into());
    }
    let config = Config {
        stack: Stack {
            vocab: hf.vocab_size,
            hidden: hf.hidden_size,
            layers: hf.num_hidden_layers,
            heads: hf.num_attention_heads,
            head_dim: hf.head_dim,
            ffn: hf.intermediate_size,
            max_seqs: slots,
            max_context: context,
        },
        kv_heads: hf.num_key_value_heads,
        rope_theta: hf.rope_theta,
        eps: hf.rms_norm_eps,
    };
    let graph = stepper(&config);
    let file = File::open(&dir.join("model.safetensors"))?;
    let find = |p: &[graph::PathSeg]| {
        let v = file.floats(&tensor_name(p)?).ok()?;
        Some(interp::Tensor::f32(vec![v.len()], v))
    };
    let weights = bind(&graph, find)?
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    let template = Template::load(dir)?;
    let stop = match std::fs::read_to_string(dir.join("generation_config.json")) {
        Ok(text) => match serde_json::from_str::<Generation>(&text)?.eos_token_id {
            Some(Ids::One(id)) => vec![id],
            Some(Ids::Many(ids)) => ids,
            None => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    Ok(Bundle {
        graph,
        weights,
        pins: None,
        tokenizer,
        template,
        stop,
    })
}
