//! A Hugging Face model directory: graph, weights, tokenizer, template.

pub mod encoder;
pub mod flux2;
mod gguf;
#[cfg(test)]
mod gguf_test;
pub mod mla;
#[cfg(test)]
mod mla_test;
pub mod names;
#[cfg(test)]
mod names_test;
pub mod spacy;
pub mod sprite;
#[cfg(test)]
mod sprite_test;
pub mod taps;
mod template;
#[cfg(test)]
mod template_test;
pub mod tokenize;
pub mod vae;

pub use template::{Clock, Message, Template};

use names::Rule;
use sensai_checkpoint::{File, Offer, Stored};
use sensai_engine::decoder::{self, Config, Moe, Scales, Swiglu};
use sensai_engine::form::{BF16_FORM, F32_FORM, Load, named};
use sensai_engine::{Graph, Norm, Pins, Stack, Tensor, Weight};
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
    /// Older files leave it out: `hidden_size / num_attention_heads`.
    head_dim: Option<u64>,
    intermediate_size: u64,
    rope_theta: f32,
    rms_norm_eps: f32,
    /// Routed models name their experts and the picks per row.
    num_local_experts: Option<u64>,
    num_experts_per_tok: Option<u64>,
    /// Granite's multipliers.
    embedding_multiplier: Option<f32>,
    attention_multiplier: Option<f32>,
    residual_multiplier: Option<f32>,
    logits_scaling: Option<f32>,
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
    let mut weights = Vec::with_capacity(graph.weights.len());
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(graph.weights.len());
    each(graph, offer, rules, |weight, form| {
        weights.push(weight);
        forms.push(form);
        Ok(())
    })?;
    Ok((weights, gguf::pins(graph, &forms, layers)))
}

/// The forms `offer` stores. No weight bytes are read.
pub fn pins(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
) -> Result<Pins, Box<dyn Error + Send + Sync>> {
    let slices = offer.slices();
    let claims = names::resolve(graph, &slices, rules)?;
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for c in &claims {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        forms.push(form(slice.stored, !c.parts.is_empty())?);
    }
    Ok(gguf::pins(graph, &forms, layers))
}

/// A slice's form. A `set` joins many slices.
fn form(stored: Stored, set: bool) -> Result<&'static dyn Load, Box<dyn Error + Send + Sync>> {
    match stored {
        Stored::Bf16 => Ok(BF16_FORM),
        // Only a form without a header joins byte for byte.
        Stored::Form(f) if set && f.bytes(0) != 0 => {
            Err(format!("a set of {} matrices, each with a header", f.name()).into())
        }
        Stored::Form(f) => {
            named(f.name()).ok_or_else(|| format!("no loads for {}", f.name()).into())
        }
        _ => Ok(F32_FORM),
    }
}

/// Each weight in graph order, read one at a time.
pub fn each(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[&[Rule]],
    mut take: impl FnMut(Weight, &'static dyn Load) -> Result<(), Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let slices = offer.slices();
    let claims = names::resolve(graph, &slices, rules)?;
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let axis = |e: &sensai_engine::DimExpr| e.value(&graph.dims, &|_| None).map(|n| n as usize);
        let shape: Option<Vec<usize>> = w.ty.shape.0.iter().map(axis).collect();
        let f = form(slice.stored, !c.parts.is_empty())?;
        let weight = match (slice.stored, c.rows) {
            _ if !c.parts.is_empty() => stacked(offer, slice.stored, &c.parts, shape)?,
            (Stored::Form(_), Some(rows)) => Weight::Packed(f, offer.packed_rows(&c.name, rows)?),
            (Stored::Form(_), None) => Weight::Packed(f, offer.packed(&c.name)?),
            (Stored::Bf16, _) => halves(offer, slice, c.rows)?,
            _ => floats(offer, slice, c.rows, shape)?,
        };
        take(weight, f)?;
    }
    Ok(())
}

/// A set's matrices, one slice each, joined in order.
fn stacked(
    offer: &dyn Offer,
    stored: Stored,
    parts: &[String],
    shape: Option<Vec<usize>>,
) -> Result<Weight, Box<dyn Error + Send + Sync>> {
    let mut all = Vec::new();
    match stored {
        Stored::Bf16 => {
            for p in parts {
                all.extend(offer.bf16(p)?);
            }
            Ok(Weight::Packed(BF16_FORM, all))
        }
        Stored::Form(_) => {
            for p in parts {
                all.extend(offer.packed(p)?);
            }
            Ok(Weight::Packed(form(stored, true)?, all))
        }
        _ => {
            let mut floats = Vec::new();
            for p in parts {
                floats.extend(offer.floats(p)?);
            }
            let shape = shape.unwrap_or_else(|| vec![floats.len()]);
            Ok(Weight::Floats(Tensor::f32(shape, floats)))
        }
    }
}

/// A bf16 slice's bytes, or its `rows`, kept packed.
fn halves(
    offer: &dyn Offer,
    slice: &sensai_checkpoint::Slice,
    rows: Option<(usize, usize)>,
) -> Result<Weight, sensai_checkpoint::Error> {
    let mut v = offer.bf16(&slice.name)?;
    if let Some((first, count)) = rows {
        let stride = 2 * slice.elements() / slice.shape[0];
        v = v[first * stride..(first + count) * stride].to_vec();
    }
    Ok(Weight::Packed(BF16_FORM, v))
}

/// A slice's floats, or its `rows`, in the graph's shape.
fn floats(
    offer: &dyn Offer,
    slice: &sensai_checkpoint::Slice,
    rows: Option<(usize, usize)>,
    shape: Option<Vec<usize>>,
) -> Result<Weight, sensai_checkpoint::Error> {
    let mut v = offer.floats(&slice.name)?;
    if let Some((first, count)) = rows {
        let stride = slice.elements() / slice.shape[0];
        v = v[first * stride..(first + count) * stride].to_vec();
    }
    let shape = shape.unwrap_or_else(|| vec![v.len()]);
    Ok(Weight::Floats(Tensor::f32(shape, v)))
}

/// The step graph for an architecture name.
pub(crate) fn stepper(arch: &str, c: &Config) -> Option<Graph> {
    Some(match arch {
        "qwen3" => decoder::stepper::<Norm, Swiglu>(c),
        "llama" => decoder::stepper::<(), Swiglu>(c),
        "granitemoe" => decoder::stepper::<(), Moe>(c),
        _ => return None,
    })
}

/// A head's width, where a file leaves it out.
pub(crate) fn head_dim(given: Option<u64>, hidden: u64, heads: u64) -> u64 {
    given.unwrap_or(hidden / heads)
}

/// The architecture and graph sizes `dir/config.json` gives.
pub(crate) fn config(
    dir: &Path,
    slots: u64,
    context: u64,
) -> Result<(String, Config), Box<dyn Error + Send + Sync>> {
    let hf: HfConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let config = Config {
        stack: Stack {
            vocab: hf.vocab_size,
            hidden: hf.hidden_size,
            layers: hf.num_hidden_layers,
            heads: hf.num_attention_heads,
            head_dim: head_dim(hf.head_dim, hf.hidden_size, hf.num_attention_heads),
            ffn: hf.intermediate_size,
            max_seqs: slots,
            max_context: context,
            masked: false,
        },
        kv_heads: hf.num_key_value_heads,
        rope_theta: hf.rope_theta,
        eps: hf.rms_norm_eps,
        experts: hf.num_local_experts.unwrap_or(0),
        top_k: hf.num_experts_per_tok.unwrap_or(0),
        scales: Scales {
            embed: hf.embedding_multiplier.unwrap_or(1.0),
            attention: hf.attention_multiplier,
            residual: hf.residual_multiplier.unwrap_or(1.0),
            logits: 1.0 / hf.logits_scaling.unwrap_or(1.0),
        },
    };
    Ok((hf.model_type, config))
}

/// The trellis width an exlr8 release's experts load at.
const WIDTH: u32 = 3;

/// Reads a model directory: `slots` sequences of `context` rows.
pub fn load(dir: &Path, slots: u64, context: u64) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    load_named(dir, slots, context, &[], false)
}

/// The user's name rules go last. `masked` adds mask ports.
pub fn load_named(
    dir: &Path,
    slots: u64,
    context: u64,
    user: &[Rule],
    masked: bool,
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    if dir.is_file() {
        return gguf::load(dir, slots, context, user, masked);
    }
    let json = std::fs::read_to_string(dir.join("config.json"))?;
    if mla::config(&json, slots, context).is_ok() {
        let release = mla::stepping(dir, [slots, context], WIDTH, user, masked)?;
        let mut weights = Vec::with_capacity(release.graph.weights.len());
        release.each(|_, weight| {
            weights.push(weight);
            Ok(())
        })?;
        return dressed(dir, release.graph, weights, release.pins);
    }
    let (arch, mut config) = config(dir, slots, context)?;
    config.stack.masked = masked;
    let graph = stepper(&arch, &config)
        .ok_or_else(|| format!("{arch} is not a model this engine traces"))?;
    let file = File::open(&dir.join("model.safetensors"))?;
    let (weights, pins) = pull(&graph, &file, config.stack.layers, &[user])?;
    dressed(dir, graph, weights, pins)
}

/// The graph and weights with `dir`'s tokenizer, template, stops.
fn dressed(
    dir: &Path,
    graph: Graph,
    weights: Vec<Weight>,
    pins: Pins,
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
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
