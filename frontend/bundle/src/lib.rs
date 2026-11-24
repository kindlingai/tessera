//! A Hugging Face model directory: graph, weights, tokenizer, template.

mod arch;
mod decoder;
pub mod dflash;
#[cfg(test)]
mod dflash_test;
pub mod encoder;
pub mod flux2;
mod gemma4;
mod gguf;
#[cfg(test)]
mod gguf_test;
pub mod mla;
#[cfg(test)]
mod mla_test;
pub mod mtp;
#[cfg(test)]
mod mtp_test;
pub mod names;
#[cfg(test)]
mod names_test;
pub mod parts;
pub mod qwen35;
pub mod qwen4_exp;
pub(crate) mod repack;
#[cfg(test)]
mod repack_test;
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
use tessera_checkpoint::{Offer, Shards, Stored, exlr8};
use tessera_engine::decoder::Config;
use tessera_engine::form::{BF16_FORM, F32_FORM, Load, named};
use tessera_engine::keys::{Keys, Offered};
use tessera_engine::wasm::Module;
use tessera_engine::{DimName, Graph, Pins, PleConfig, Steps, Tensor, Weight};
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

type Failure = Box<dyn Error + Send + Sync>;

/// Weights read one at a time as they load.
pub trait Stream {
    /// Each weight in graph order, cut to `shares` where possible.
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
    ) -> Result<(), Failure>;
}

/// A model's weights: held, or read as they load.
pub enum Weights {
    Held(Vec<Weight>),
    Streamed(Box<dyn Stream>),
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
        take: impl FnMut(u32, &Weight) -> Result<(), E>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.each_share(|at| rows(at).map(|(f, c)| (0, f, c)), take)
    }

    /// As `each`. Streams read only the kept share slice.
    pub fn each_share<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        shares: impl Fn(u32) -> Option<(u32, usize, usize)>,
        mut take: impl FnMut(u32, &Weight) -> Result<(), E>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            Weights::Held(all) => {
                for (at, w) in all.iter().enumerate() {
                    take(at as u32, w).map_err(Into::into)?;
                }
                Ok(())
            }
            Weights::Streamed(stream) => {
                stream.stream(&shares, &mut |at, w| take(at, &w).map_err(Into::into))
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
    /// Model `config.json` with recipe overrides applied.
    pub config: Option<String>,
    /// Host PLE table config, when needed.
    pub ple: Option<PleConfig>,
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

    /// The config `build` reads from `dir/config.json`, as `read`.
    pub fn config<T>(
        &self,
        dir: &Path,
        build: impl FnOnce(&mut Keys) -> T,
    ) -> Result<T, Box<dyn Error + Send + Sync>> {
        self.read("config.json", Some(&dir.join("config.json")), build)
    }

    /// `path`'s JSON with the recipe's `key` table overlaid on top.
    pub fn merged(&self, key: &str, path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        let Some(over) = self.0.get(key) else {
            return Ok(text);
        };
        let mut base: serde_json::Value = serde_json::from_str(&text)?;
        overlay(&mut base, over);
        Ok(serde_json::to_string(&base)?)
    }
}

/// Overlays JSON object keys from `over` onto `base`.
fn overlay(base: &mut serde_json::Value, over: &serde_json::Value) {
    if let (serde_json::Value::Object(dst), serde_json::Value::Object(src)) = (&mut *base, over) {
        for (k, v) in src {
            match dst.get_mut(k) {
                Some(existing) => overlay(existing, v),
                None => {
                    dst.insert(k.clone(), v.clone());
                }
            }
        }
    } else {
        *base = over.clone();
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
pub(crate) fn form(
    stored: Stored,
    set: bool,
) -> Result<&'static dyn Load, Box<dyn Error + Send + Sync>> {
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

/// As `each`, with each weight's place in graph order.
pub(crate) fn numbered(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[&[Rule]],
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
    take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
) -> Result<(), Failure> {
    let mut at = 0;
    each(graph, offer, rules, shares, |weight, _| {
        at += 1;
        take(at - 1, weight)
    })
}

/// Each weight in graph order, cut to `shares` where possible.
pub fn each(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[&[Rule]],
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
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
            (Stored::Form(_), None) => match shares(at as u32) {
                Some((ax, first, count)) => {
                    Weight::Packed(f, offer.packed_axis(&c.name, ax, (first, count))?)
                }
                None => Weight::Packed(f, offer.packed(&c.name)?),
            },
            (Stored::Bf16, None) => match offer.filed(&c.name) {
                Some(file) => Weight::Filed(BF16_FORM, file),
                None => halves(offer, slice, None)?,
            },
            (Stored::Bf16, rows) => halves(offer, slice, rows)?,
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

/// The architecture and graph sizes `dir/config.json` gives.
pub(crate) fn config(
    dir: &Path,
    [slots, context]: [u64; 2],
    overrides: &Overrides,
) -> Result<(String, Config), Box<dyn Error + Send + Sync>> {
    overrides.config(dir, |k| {
        (k.need(&["model_type"]), Config::read(k, slots, context))
    })
}

/// The file whose presence marks an exlr8 release.
const EXLR8: &str = "dense/exlr8-vocab.safetensors";

/// Whether `dir` holds an exlr8 release.
pub(crate) fn is_exlr8(dir: &Path) -> bool {
    dir.join(EXLR8).is_file()
}

/// `dir`'s weights: an exlr8 release at `manifest`, or shards.
pub(crate) fn offer(
    dir: &Path,
    manifest: Option<&str>,
) -> Result<Box<dyn Offer>, Box<dyn Error + Send + Sync>> {
    Ok(match is_exlr8(dir) {
        true => Box::new(release(dir, manifest)?),
        false => Box::new(Shards::open(dir)?),
    })
}

/// Reads a model directory: `slots` sequences of `context` rows.
pub fn load(dir: &Path, slots: u64, context: u64) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    load_named(
        dir,
        [slots, context],
        &[],
        Steps::Plain,
        &Overrides::default(),
        None,
    )
}

/// User rules go last. `manifest` picks exlr8 widths.
pub fn load_named(
    dir: &Path,
    sizes: [u64; 2],
    user: &[Rule],
    steps: Steps,
    overrides: &Overrides,
    manifest: Option<&str>,
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    load_tapped(dir, sizes, user, steps, overrides, manifest, &[])
}

/// As `load_named`, with 1-based layer `taps` on stepping.
pub fn load_tapped(
    dir: &Path,
    sizes: [u64; 2],
    user: &[Rule],
    steps: Steps,
    overrides: &Overrides,
    manifest: Option<&str>,
    taps: &[u64],
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    if dir.is_file() {
        if !taps.is_empty() {
            return Err("gguf checkpoints do not support step taps yet".into());
        }
        return gguf::load(dir, sizes[0], sizes[1], user, steps);
    }
    let (json, arch) = arch::read(dir, overrides)?;
    let step = arch
        .step
        .ok_or_else(|| format!("{} has no step graph yet", arch.name))?;
    let ask = arch::Ask {
        dir,
        sizes,
        user,
        steps,
        overrides,
        manifest,
        taps,
        json: &json,
    };
    let (graph, weights, pins) = step(arch, &ask)?;
    let ple = match arch.name {
        "qwen4_exp" => qwen4_exp::ple_config(dir, &ask)?,
        _ => None,
    };
    let mut bundle = dressed(dir, graph, weights, pins, Some(json))?;
    bundle.ple = ple;
    Ok(bundle)
}

/// `graph`'s weights held from `ask`'s folder, `rules` first.
pub(crate) fn held(
    ask: &arch::Ask,
    graph: Graph,
    layers: u64,
    rules: &[Rule],
) -> Result<arch::Built, Box<dyn Error + Send + Sync>> {
    let offer = offer(ask.dir, ask.manifest)?;
    let (weights, pins) = pull(&graph, offer.as_ref(), layers, &[rules, ask.user])?;
    Ok((graph, Weights::Held(weights), pins))
}

/// A module's step graph over `dir`'s weights, if built.
pub fn load_module(
    dir: &Path,
    sizes: [u64; 2],
    user: &[Rule],
    overrides: &Overrides,
    module: &Module,
) -> Result<Option<Bundle>, Box<dyn Error + Send + Sync>> {
    if !module.builds() || dir.is_file() {
        return Ok(None);
    }
    let json = overrides.merged("config.json", &dir.join("config.json"))?;
    let Some(graph) = module.graph(&json, sizes[0], Some(sizes[1]))? else {
        return Ok(None);
    };
    let layers = graph.dims.iter().find(|d| d.name == DimName::Layers);
    let layers = layers.map_or(0, |d| d.size.max());
    let ask = arch::Ask {
        user,
        ..arch::Ask::plain(dir, sizes, overrides, &json)
    };
    let (graph, weights, pins) = held(&ask, graph, layers, &[])?;
    dressed(dir, graph, weights, pins, Some(json)).map(Some)
}

/// `dir`'s whole-prompt graph, its weights and their forms.
pub fn forward(
    dir: &Path,
    sizes: [u64; 2],
    overrides: &Overrides,
) -> Result<(Graph, Weights, Pins), Failure> {
    let (json, arch) = arch::read(dir, overrides)?;
    let forward = arch
        .forward
        .ok_or_else(|| format!("{} has no forward graph yet", arch.name))?;
    forward(arch, &arch::Ask::plain(dir, sizes, overrides, &json))
}

/// `dir`'s exlr8 release at `name`, or its only manifest.
pub(crate) fn release(
    dir: &Path,
    name: Option<&str>,
) -> Result<exlr8::Release, Box<dyn Error + Send + Sync>> {
    let release = exlr8::Release::open(dir);
    let all = release.manifests();
    let name = match (name, all.as_slice()) {
        (Some(name), _) => name.to_string(),
        (None, []) => return Ok(release),
        (None, [one]) => one.clone(),
        (None, many) => {
            return Err(format!("the release has manifests {}: pick one", many.join(", ")).into());
        }
    };
    Ok(release.manifest(&name)?)
}

/// The graph and weights with `dir`'s tokenizer, template, stops.
fn dressed(
    dir: &Path,
    graph: Graph,
    weights: Weights,
    pins: Pins,
    config: Option<String>,
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
        config,
        ple: None,
    })
}
