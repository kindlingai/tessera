//! A Hugging Face model directory: graph, weights, tokenizer, template.

mod arch;
mod decoder;
pub mod dflash;
#[cfg(test)]
mod dflash_test;
pub mod encoder;
#[cfg(test)]
mod exlr8_test;
pub mod flux2;
mod gemma4;
mod gguf;
#[cfg(test)]
mod gguf_test;
pub mod hy4;
mod inkling2;
pub mod mla;
pub mod mtp;
#[cfg(test)]
mod mtp_test;
pub mod names;
#[cfg(test)]
mod names_test;
pub mod parts;
pub mod qwen35;
mod qwen3_vl;
pub mod qwen4_exp;
pub(crate) mod repack;
#[cfg(test)]
mod repack_test;
mod response;
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

pub use response::{Markers, ResponseEvent, ResponseParser, ResponseStream};
pub use template::{FunctionCall, Message, RenderOptions, Template, ToolCall};

use names::Rule;
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Stored};
use tessera_engine::decoder::Config;
use tessera_engine::form::{F32_FORM, Load, named};
use tessera_engine::hf::MODEL_TYPE;
use tessera_engine::keys::{At, Keys, Offered};
use tessera_engine::progress::{Progress, Unit};
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

    /// The integer tensor `name`, read whole.
    fn i64s(&self, name: &str) -> Result<Vec<i64>, Failure> {
        Err(format!("{name}: integer tensors need an open release").into())
    }
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

    /// As `each_share`, as a step that counts `count` weights.
    pub fn each_counted<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        progress: &Progress,
        count: u32,
        shares: impl Fn(u32) -> Option<(u32, usize, usize)>,
        mut take: impl FnMut(u32, &Weight) -> Result<(), E>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let step = progress.step("weights", Unit::Items, Some(count.into()));
        // Sizing runners only count bytes, so skip the reads.
        if tessera_engine::sizer::counting() {
            step.finish();
            return Ok(());
        }
        self.each_share(shares, |at, w| {
            take(at, w)?;
            step.add(1);
            Ok::<_, E>(())
        })?;
        step.finish();
        Ok(())
    }
}

pub struct Bundle {
    pub graph: Graph,
    pub weights: Weights,
    /// Forms packed in the file, when any are.
    pub pins: Option<Pins>,
    pub tokenizer: Hf,
    pub template: Option<Template>,
    /// Response parser from `tokenizer_config.json`, if declared.
    pub response: Option<ResponseParser>,
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

    /// Sets `key` in `file`'s table, over any value there.
    pub fn set(&mut self, file: &str, key: &str, value: serde_json::Value) {
        let table = self.0.entry(file).or_insert_with(|| serde_json::json!({}));
        if let Some(table) = table.as_object_mut() {
            table.insert(key.into(), value);
        }
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
    pull_shared(graph, offer, layers, rules, &|_| None)
}

/// Send-safe loaded weight before wrapping in a `Tensor`.
pub(crate) struct Loaded {
    form: &'static dyn Load,
    data: RawData,
}

enum RawData {
    Floats(Vec<usize>, Vec<f32>),
    Packed(Vec<u8>),
    Filed(std::sync::Arc<dyn tessera_engine::Filed>),
}

type Slot = std::sync::Mutex<Option<Result<Loaded, Failure>>>;

impl Loaded {
    pub(crate) fn into_weight(self) -> Weight {
        match self.data {
            RawData::Floats(shape, v) => Weight::Floats(Tensor::f32(shape, v)),
            RawData::Packed(b) => Weight::Packed(self.form, b),
            RawData::Filed(file) => Weight::Filed(self.form, file),
        }
    }
}

/// Reads one claimed weight from `offer`.
fn load_claim(
    graph: &Graph,
    offer: &dyn Offer,
    slice: &tessera_checkpoint::Slice,
    c: &names::Claim,
    at: usize,
    share: Option<(u32, usize, usize)>,
) -> Result<Loaded, Failure> {
    let w = &graph.weights[at];
    let axis = |e: &tessera_engine::DimExpr| e.value(&graph.dims, &|_| None).map(|n| n as usize);
    let shape: Option<Vec<usize>> = w.ty.shape.0.iter().map(axis).collect();
    let form = form(slice.stored, !c.parts.is_empty())?;
    let packed = matches!(slice.stored, Stored::Form(src) if src.name() != "f32");
    let data = match (packed, c.rows) {
        _ if !c.parts.is_empty() => stacked_raw(offer, slice.stored, &c.parts, shape)?,
        (true, Some(rows)) => RawData::Packed(offer.packed_rows(&c.name, rows)?),
        (true, None) => match (share, offer.filed(&c.name)) {
            (Some((ax, first, count)), None) => {
                RawData::Packed(offer.packed_axis(&c.name, ax, (first, count))?)
            }
            (_, Some(file)) => RawData::Filed(file),
            (None, None) => RawData::Packed(offer.packed(&c.name)?),
        },
        (false, _) => floats_raw(offer, slice, c.rows, shape)?,
    };
    Ok(Loaded { form, data })
}

/// Send-safe variant of `pull` for concurrent multi-graph loading.
pub(crate) fn pull_raw(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
) -> Result<(Vec<Loaded>, Pins), Failure> {
    let slices = offer.slices();
    let claims = names::resolve(graph, &slices, rules)?;
    let by_name: std::collections::HashMap<&str, &tessera_checkpoint::Slice> =
        slices.iter().map(|s| (s.name.as_str(), s)).collect();
    let count = claims.len();
    let wanted: Vec<_> = (0..count).map(|i| shares(i as u32)).collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, 8)
        .min(count);
    let mut loaded = Vec::with_capacity(count);
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(count);
    if threads <= 1 || count < 8 {
        for (at, c) in claims.iter().enumerate() {
            let item = load_claim(graph, offer, by_name[c.name.as_str()], c, at, wanted[at])?;
            forms.push(item.form);
            loaded.push(item);
        }
        return Ok((loaded, gguf::pins(graph, &forms, layers)));
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<Slot> = (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (claims, by_name, wanted, next, results) =
                (&claims, &by_name, &wanted, &next, &results);
            scope.spawn(move || {
                loop {
                    let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if at >= count {
                        break;
                    }
                    let c = &claims[at];
                    let res = load_claim(graph, offer, by_name[c.name.as_str()], c, at, wanted[at]);
                    *results[at].lock().expect("weight slot") = Some(res);
                }
            });
        }
    });
    for slot in results {
        let item = slot.into_inner().expect("weight slot").expect("loaded")?;
        forms.push(item.form);
        loaded.push(item);
    }
    Ok((loaded, gguf::pins(graph, &forms, layers)))
}

/// As `pull`, reading only each rank's slice when `shares` says.
pub(crate) fn pull_shared(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
) -> Result<(Vec<Weight>, Pins), Box<dyn Error + Send + Sync>> {
    let (loaded, pins) = pull_raw(graph, offer, layers, rules, shares)?;
    let weights = loaded.into_iter().map(Loaded::into_weight).collect();
    Ok((weights, pins))
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
    let by_name: std::collections::HashMap<&str, &tessera_checkpoint::Slice> =
        slices.iter().map(|s| (s.name.as_str(), s)).collect();
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for c in &claims {
        let slice = by_name[c.name.as_str()];
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
        Stored::Form(f) if f.name() == "f32" => Ok(F32_FORM),
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
    let by_name: std::collections::HashMap<&str, &tessera_checkpoint::Slice> =
        slices.iter().map(|s| (s.name.as_str(), s)).collect();
    for (at, c) in claims.iter().enumerate() {
        let slice = by_name[c.name.as_str()];
        let item = load_claim(graph, offer, slice, c, at, shares(at as u32))?;
        let f = item.form;
        take(item.into_weight(), f)?;
    }
    Ok(())
}

/// A set's matrices, one slice each, joined in order.
fn stacked_raw(
    offer: &dyn Offer,
    stored: Stored,
    parts: &[String],
    shape: Option<Vec<usize>>,
) -> Result<RawData, Box<dyn Error + Send + Sync>> {
    match stored {
        Stored::Form(src) if src.name() != "f32" => {
            let mut all = Vec::new();
            for p in parts {
                all.extend(offer.packed(p)?);
            }
            Ok(RawData::Packed(all))
        }
        _ => {
            let mut floats = Vec::new();
            for p in parts {
                floats.extend(offer.floats(p)?);
            }
            let shape = shape.unwrap_or_else(|| vec![floats.len()]);
            Ok(RawData::Floats(shape, floats))
        }
    }
}

/// A slice's floats, or its `rows`, in the graph's shape.
fn floats_raw(
    offer: &dyn Offer,
    slice: &tessera_checkpoint::Slice,
    rows: Option<(usize, usize)>,
    shape: Option<Vec<usize>>,
) -> Result<RawData, tessera_checkpoint::Error> {
    let mut v = offer.floats(&slice.name)?;
    if let Some((first, count)) = rows {
        let stride = slice.elements() / slice.shape[0];
        v = v[first * stride..(first + count) * stride].to_vec();
    }
    let shape = shape.unwrap_or_else(|| vec![v.len()]);
    Ok(RawData::Floats(shape, v))
}

/// The architecture and graph sizes `dir/config.json` gives.
pub(crate) fn config(
    dir: &Path,
    [slots, context]: [u64; 2],
    overrides: &Overrides,
) -> Result<(String, Config), Box<dyn Error + Send + Sync>> {
    overrides.config(dir, |k| {
        (
            k.read(&At::top(), MODEL_TYPE),
            Config::read(k, slots, context),
        )
    })
}

/// Opens `dir`'s weights through the checkpoint format registry.
pub(crate) fn offer(
    dir: &Path,
    manifest: Option<&str>,
) -> Result<Box<dyn Offer>, Box<dyn Error + Send + Sync>> {
    Ok(tessera_checkpoint::open(dir, manifest)?)
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
    let built = step(arch, &ask)?;
    let ple = match arch.ple {
        Some(ple) => ple(&ask, &built.1)?,
        None => None,
    };
    dressed(dir, built, Some(json), ple, overrides)
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

/// Builds the weight request used by `load_module`.
pub(crate) fn module_ask<'a>(
    dir: &'a Path,
    sizes: [u64; 2],
    user: &'a [Rule],
    overrides: &'a Overrides,
    json: &'a str,
    manifest: Option<&'a str>,
) -> arch::Ask<'a> {
    arch::Ask {
        user,
        manifest,
        ..arch::Ask::plain(dir, sizes, overrides, json)
    }
}

/// A module's step graph over `dir`'s weights, if built.
pub fn load_module(
    dir: &Path,
    sizes: [u64; 2],
    user: &[Rule],
    overrides: &Overrides,
    manifest: Option<&str>,
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
    let ask = module_ask(dir, sizes, user, overrides, &json, manifest);
    let built = held(&ask, graph, layers, &[])?;
    dressed(dir, built, Some(json), None, overrides).map(Some)
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

/// `own`, else the built-in format `template` writes.
pub(crate) fn respond(
    template: &mut Option<Template>,
    own: Option<ResponseParser>,
) -> Result<Option<ResponseParser>, Box<dyn Error + Send + Sync>> {
    let response = match (own, template.as_ref()) {
        (None, Some(chat)) => ResponseParser::builtin(chat)?,
        (own, _) => own,
    };
    if let (Some(chat), Some(prompt)) = (template, response.as_ref().and_then(|r| r.tool_prompt()))
    {
        chat.with_tool_prompt(prompt)?;
    }
    Ok(response)
}

/// The graph and weights with `dir`'s tokenizer, template, stops.
fn dressed(
    dir: &Path,
    (graph, weights, pins): arch::Built,
    config: Option<String>,
    ple: Option<PleConfig>,
    overrides: &Overrides,
) -> Result<Bundle, Box<dyn Error + Send + Sync>> {
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    let tok_cfg = overrides.merged("tokenizer_config.json", &dir.join("tokenizer_config.json"))?;
    let mut template = Template::from_config(dir, &tok_cfg)?;
    let own = ResponseParser::from_config(&tok_cfg)?;
    let response = respond(&mut template, own)?;
    let mut stop = match std::fs::read_to_string(dir.join("generation_config.json")) {
        Ok(text) => match serde_json::from_str::<Generation>(&text)?.eos_token_id {
            Some(Ids::One(id)) => vec![id],
            Some(Ids::Many(ids)) => ids,
            None => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    // Without `generation_config.json`, stop on the config's end token.
    if stop.is_empty()
        && let Some(id) = template.as_ref().and_then(|t| tokenizer.id(t.eos_token()?))
    {
        stop.push(id);
    }
    Ok(Bundle {
        graph,
        weights,
        pins: Some(pins),
        tokenizer,
        template,
        response,
        stop,
        config,
        ple,
    })
}
