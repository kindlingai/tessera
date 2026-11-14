//! A Hugging Face model directory: graph, weights, tokenizer, template.

mod gguf;
#[cfg(test)]
mod gguf_test;
pub mod names;
#[cfg(test)]
mod names_eg2_test;
#[cfg(test)]
mod names_test;
pub mod tag;
mod template;
#[cfg(test)]
mod template_test;
pub mod tokenize;

pub use template::{Message, Template};

use checkpoint::{File, Offer, Stored};
use graph::Graph;
use interp::Tensor;
use lower::form::{BF16_FORM, F32_FORM, Form};
use lower::{Pins, Weight};
use models::Stack;
use models::qwen3::{Config, stepper};
use names::Rule;
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

/// The weights and their forms, as `offer` stores them.
pub fn pull(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
) -> Result<(Vec<Weight>, Pins), Box<dyn Error + Send + Sync>> {
    let slices = offer.slices();
    let claims = names::resolve(graph, &slices, rules)?;
    let mut weights = Vec::with_capacity(claims.len());
    let mut forms: Vec<&'static dyn Form> = Vec::with_capacity(claims.len());
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let axis = |e: &graph::DimExpr| e.value(&graph.dims, &|_| None).map(|n| n as usize);
        let shape: Option<Vec<usize>> = w.ty.shape.0.iter().map(axis).collect();
        let (weight, form) = match (slice.stored, c.rows) {
            (Stored::Form(f), Some(rows)) => {
                (Weight::Packed(f, offer.packed_rows(&c.name, rows)?), f)
            }
            (Stored::Form(f), None) => (Weight::Packed(f, offer.packed(&c.name)?), f),
            (Stored::Bf16, _) => (floats(offer, slice, c.rows, shape)?, BF16_FORM),
            _ => (floats(offer, slice, c.rows, shape)?, F32_FORM),
        };
        weights.push(weight);
        forms.push(form);
    }
    Ok((weights, gguf::pins(graph, &forms, layers)))
}

/// A slice's floats, or its `rows`, in the graph's shape.
fn floats(
    offer: &dyn Offer,
    slice: &checkpoint::Slice,
    rows: Option<(usize, usize)>,
    shape: Option<Vec<usize>>,
) -> Result<Weight, checkpoint::Error> {
    let mut v = offer.floats(&slice.name)?;
    if let Some((first, count)) = rows {
        let stride = slice.elements() / slice.shape[0];
        v = v[first * stride..(first + count) * stride].to_vec();
    }
    let shape = shape.unwrap_or_else(|| vec![v.len()]);
    Ok(Weight::Floats(Tensor::f32(shape, v)))
}

/// Reads a Qwen3 directory: `slots` sequences of `context` rows.
pub fn load(dir: &Path, slots: u64, context: u64) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    load_named(dir, slots, context, &[])
}

/// The same, with the user's weight-name rules applied last.
pub fn load_named(
    dir: &Path,
    slots: u64,
    context: u64,
    user: &[Rule],
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    if dir.is_file() {
        return gguf::load(dir, slots, context, user);
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
    let (weights, pins) = pull(&graph, &file, config.stack.layers, &[user])?;
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
        pins: Some(pins),
        tokenizer,
        template,
        stop,
    })
}
