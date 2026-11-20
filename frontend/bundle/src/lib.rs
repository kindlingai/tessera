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
pub mod parts;
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
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards, Stored};
use tessera_engine::decoder::{self, Config, Moe, Swiglu};
use tessera_engine::form::{BF16_FORM, F32_FORM, Load, named};
use tessera_engine::keys::{Keys, Offered};
use tessera_engine::{Graph, Norm, Pins, Tensor, Weight};
use tokenize::hf::Hf;

/// One id or a list, the `generation_config.json` shape.
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

/// A model's weights: held, or read as they load.
pub enum Weights {
    Held(Vec<Weight>),
    Streamed(Box<mla::Release>),
}

impl Weights {
    /// Each weight in graph order, one held at a time.
    pub fn each<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        take: impl FnMut(u32, &Weight) -> Result<(), E>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.each_rows(|_| None, take)
    }

    /// As `each`. Streams read only the kept `rows`.
    pub fn each_rows<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        rows: impl Fn(u32) -> Option<(usize, usize)>,
        mut take: impl FnMut(u32, &Weight) -> Result<(), E>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            Weights::Held(all) => {
                for (at, w) in all.iter().enumerate() {
                    take(at as u32, w).map_err(Into::into)?;
                }
                Ok(())
            }
            Weights::Streamed(release) => {
                release.each(&rows, |at, w| take(at, &w).map_err(Into::into))
            }
        }
    }
}

pub struct Bundle {
    pub graph: Graph,
    pub weights: Weights,
    /// Forms packed in the file, when any are.
    pub pins: Option<Pins>,
    pub tokenizer: Hf,
    pub template: Option<Template>,
    /// Ids that end an answer.
    pub stop: Vec<u32>,
}

/// Config values a recipe sets, by the file they override.
#[derive(Clone, Debug, Default)]
pub struct Overrides(serde_json::Map<String, serde_json::Value>);

impl Overrides {
    pub fn new(files: serde_json::Map<String, serde_json::Value>) -> Self {
        Self(files)
    }

    /// The tables for files in `folder`, keyed from inside it.
    pub fn under(&self, folder: &str) -> Self {
        let prefix = format!("{folder}/");
        let mut files = serde_json::Map::new();
        for (key, table) in &self.0 {
            if let Some(inner) = key.strip_prefix(&prefix) {
                files.insert(inner.into(), table.clone());
            }
        }
        Self(files)
    }

    /// The config `build` reads: the recipe's `key` table, then `path`.
    pub fn read<T>(
        &self,
        key: &str,
        path: Option<&Path>,
        build: impl FnOnce(&mut Keys) -> T,
    ) -> Result<T, Box<dyn Error + Send + Sync>> {
        let mut offers = Vec::new();
        if let Some(table) = self.0.get(key) {
            let (name, root) = (format!("recipe [config.\"{key}\"]"), table.clone());
            offers.push(Offered { name, root });
        }
        if let Some(path) = path {
            let text = std::fs::read_to_string(path)?;
            offers.push(Offered::parse(&path.display().to_string(), &text)?);
        }
        let mut keys = Keys::new(&offers);
        let config = build(&mut keys);
        keys.finish()?;
        Ok(config)
    }
}

/// The weights and their forms, in `offer`'s layout.
pub fn pull(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
) -> Result<(Vec<Weight>, Pins), Box<dyn Error + Send + Sync>> {
    let mut weights = Vec::with_capacity(graph.weights.len());
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(graph.weights.len());
    each(graph, offer, rules, &|_| None, |weight, form| {
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

/// Each weight in graph order, cut to `rows` where possible.
pub fn each(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[&[Rule]],
    rows: &dyn Fn(u32) -> Option<(usize, usize)>,
    mut take: impl FnMut(Weight, &'static dyn Load) -> Result<(), Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let slices = offer.slices();
    let claims = names::resolve(graph, &slices, rules)?;
    for (at, (c, w)) in claims.iter().zip(&graph.weights).enumerate() {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let axis =
            |e: &tessera_engine::DimExpr| e.value(&graph.dims, &|_| None).map(|n| n as usize);
        let shape: Option<Vec<usize>> = w.ty.shape.0.iter().map(axis).collect();
        let f = form(slice.stored, !c.parts.is_empty())?;
        let weight = match (slice.stored, c.rows) {
            _ if !c.parts.is_empty() => stacked(offer, slice.stored, &c.parts, shape)?,
            (Stored::Form(_), Some(rows)) => Weight::Packed(f, offer.packed_rows(&c.name, rows)?),
            (Stored::Form(_), None) => match rows(at as u32) {
                Some(keep) => Weight::Packed(f, offer.packed_share(&c.name, keep)?),
                None => Weight::Packed(f, offer.packed(&c.name)?),
            },
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
    slice: &tessera_checkpoint::Slice,
    rows: Option<(usize, usize)>,
) -> Result<Weight, tessera_checkpoint::Error> {
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
    slice: &tessera_checkpoint::Slice,
    rows: Option<(usize, usize)>,
    shape: Option<Vec<usize>>,
) -> Result<Weight, tessera_checkpoint::Error> {
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

/// The architecture and graph sizes `dir/config.json` gives.
pub(crate) fn config(
    dir: &Path,
    [slots, context]: [u64; 2],
    overrides: &Overrides,
) -> Result<(String, Config), Box<dyn Error + Send + Sync>> {
    let path = dir.join("config.json");
    overrides.read("config.json", Some(&path), |k| {
        (k.need(&["model_type"]), Config::read(k, slots, context))
    })
}

/// The trellis width an exlr8 release's experts load at.
const WIDTH: u32 = 3;

/// Reads a model directory: `slots` sequences of `context` rows.
pub fn load(dir: &Path, slots: u64, context: u64) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    load_named(dir, [slots, context], &[], false, &Overrides::default())
}

/// The user's name rules go last. `masked` adds mask ports.
pub fn load_named(
    dir: &Path,
    [slots, context]: [u64; 2],
    user: &[Rule],
    masked: bool,
    overrides: &Overrides,
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    if dir.is_file() {
        return gguf::load(dir, slots, context, user, masked);
    }
    let json = std::fs::read_to_string(dir.join("config.json"))?;
    if mla::config(&json, slots, context).is_ok() {
        let release = mla::stepping(dir, [slots, context], WIDTH, user, masked)?;
        let (graph, pins) = (release.graph.clone(), release.pins.clone());
        return dressed(dir, graph, Weights::Streamed(Box::new(release)), pins);
    }
    let (arch, mut config) = config(dir, [slots, context], overrides)?;
    config.stack.masked = masked;
    let graph = stepper(&arch, &config)
        .ok_or_else(|| format!("{arch} is not a model this engine traces"))?;
    let shards = Shards::open(dir)?;
    let (weights, pins) = pull(&graph, &shards, config.stack.layers, &[user])?;
    dressed(dir, graph, Weights::Held(weights), pins)
}

/// The graph and weights with `dir`'s tokenizer, template, stops.
fn dressed(
    dir: &Path,
    graph: Graph,
    weights: Weights,
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
