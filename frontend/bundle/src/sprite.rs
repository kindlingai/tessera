//! The sprite denoiser and its CLIP text tower.

use crate::Overrides;
use crate::tokenize::hf::Hf;
use sensai_checkpoint::{File, Offer, Torch};
use sensai_engine::wasm::Module;
use sensai_engine::{Graph, PathSeg, Tensor, Weight, bind, clip, fold, sprite};
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Named tables of floats.
type Tables = Vec<(String, Vec<f32>)>;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Sprite {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    /// RGB per palette id. Id 0 is transparent.
    pub palette: Vec<[u8; 3]>,
}

pub struct Clip {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

/// Weights by name, each cut to its first `len(name)` values.
fn tensors(
    graph: &Graph,
    get: impl Fn(&str) -> Option<Vec<f32>>,
    name: impl Fn(&[PathSeg]) -> String,
    len: impl Fn(&str) -> Option<usize>,
) -> Result<Vec<Tensor>> {
    let find = |p: &[PathSeg]| {
        let name = name(p);
        let mut v = get(&name)?;
        if let Some(n) = len(&name) {
            v.truncate(n);
        }
        Some(Tensor::f32(vec![v.len()], v))
    };
    Ok(bind(graph, find)?)
}

/// The first `.pt` file in `dir`.
fn checkpoint(dir: &Path) -> Result<PathBuf> {
    let mut found: Vec<_> = (std::fs::read_dir(dir)?.flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "pt"))
        .collect();
    found.sort();
    let none = || format!("no .pt checkpoint in {}", dir.display());
    Ok(found.into_iter().next().ok_or_else(none)?)
}

/// The graph and tables, from the module if present.
fn graph(dir: &Path, seqs: u64, module: Option<&Module>) -> Result<(Graph, Tables)> {
    let text = std::fs::read_to_string(dir.join("model_config.json"))?;
    if let Some(m) = module
        && let Some(graph) = m.graph(&text, seqs)?
    {
        return Ok((graph, m.tables(&text)?));
    }
    let config = sprite::Config::upstream(&text, seqs)?;
    let tables = sprite::constants(&config).into_iter();
    Ok((
        sprite::program(&config),
        tables.map(|c| (c.path, c.data)).collect(),
    ))
}

/// The denoiser from its Hugging Face files, folded once here.
pub fn denoiser(dir: &Path, seqs: u64, module: Option<&Module>) -> Result<Sprite> {
    let (graph, tables) = graph(dir, seqs, module)?;
    let file = Torch::open(&checkpoint(dir)?)?;
    let tables: HashMap<String, Vec<f32>> = tables.into_iter().collect();
    let get = |name: &str| match tables.get(name) {
        Some(table) => Some(table.clone()),
        None => file.floats(&format!("ema.{name}")).ok(),
    };
    let raw = tensors(&graph, get, PathSeg::dotted, |_| None)?;
    let (graph, folded) = fold(&graph, &raw)?;
    let palette = (file.floats("ema.palette_rgb")?.chunks(3))
        .map(|c| [c[0] as u8, c[1] as u8, c[2] as u8])
        .collect();
    Ok(Sprite {
        graph,
        weights: folded.into_iter().map(Weight::Floats).collect(),
        palette,
    })
}

/// CLIP's weight file in `dir`, safetensors first.
fn clip_weights(dir: &Path) -> Option<PathBuf> {
    let found = ["model.safetensors", "pytorch_model.bin"].map(|n| dir.join(n));
    found.into_iter().find(|p| p.is_file())
}

/// A Hugging Face name for a CLIP text weight path.
fn hf_name(path: &[PathSeg]) -> String {
    let dotted = PathSeg::dotted(path);
    let mut parts: Vec<&str> = dotted.split('.').collect();
    fn field(f: &str) -> &str {
        match f {
            "ln1" => "layer_norm1",
            "ln2" => "layer_norm2",
            "q" => "self_attn.q_proj",
            "k" => "self_attn.k_proj",
            "v" => "self_attn.v_proj",
            "o" => "self_attn.out_proj",
            "fc1" => "mlp.fc1",
            "fc2" => "mlp.fc2",
            other => other,
        }
    }
    let tail = match parts.as_slice() {
        ["embed"] => "embeddings.token_embedding.weight".into(),
        ["pos"] => "embeddings.position_embedding.weight".into(),
        ["norm", end] => format!("final_layer_norm.{end}"),
        ["layers", i, f, end] => format!("encoder.layers.{i}.{}.{end}", field(f)),
        _ => {
            parts.insert(0, "?");
            parts.join(".")
        }
    };
    format!("text_model.{tail}")
}

/// CLIP's text tower in `dir`, cut to `context` tokens.
pub fn text_tower(dir: &Path, [seqs, context]: [u64; 2], config: &Overrides) -> Result<Clip> {
    let path = dir.join("config.json");
    let config = config.read("clip/config.json", Some(&path), |k| {
        clip::Config::read(k, seqs, context)
    })?;
    let graph = clip::program(&config);
    let path = clip_weights(dir).ok_or_else(|| format!("no CLIP weights in {}", dir.display()))?;
    let file: Box<dyn Offer> = match path.extension().is_some_and(|x| x == "bin") {
        true => Box::new(Torch::open(&path)?),
        false => Box::new(File::open(&path)?),
    };
    let table = (context * config.stack.hidden) as usize;
    let len = |name: &str| name.ends_with("position_embedding.weight").then_some(table);
    let weights = tensors(&graph, |n| file.floats(n).ok(), hf_name, len)?;
    let weights = weights.into_iter().map(Weight::Floats).collect();
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    Ok(Clip {
        graph,
        weights,
        tokenizer,
    })
}
