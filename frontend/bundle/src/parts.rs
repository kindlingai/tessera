//! The parts of a model directory, by name.

use crate::{Overrides, config, encoder, flux2, pull, qwen35, stepper, vae};
use std::error::Error;
use std::path::{Path, PathBuf};
use tessera_checkpoint::Shards;
use tessera_engine::DimName::{self, Context, Rows, Seqs};
use tessera_engine::{Graph, Linear, Pins, Weight, eg2};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// The name of a directory that holds one model.
const WHOLE: &str = "model";

/// A model kind, from its folder's `config.json`.
enum Kind {
    Flux2,
    Vae,
    Eg2,
    Qwen35,
    Decoder,
}

/// Where a part's weights come from.
enum Source {
    Flux2(PathBuf),
    Vae(PathBuf, f32),
    Eg2(PathBuf, u64),
    Qwen35(PathBuf, u64),
    Decoder(PathBuf, u64),
}

/// A graph whose weights load later, for any cut.
pub struct Part {
    pub graph: Graph,
    source: Source,
}

fn kind(dir: &Path) -> Option<Kind> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let class = v["_class_name"].as_str();
    Some(match (class, v["architectures"][0].as_str()) {
        (Some("Flux2Transformer2DModel"), _) => Kind::Flux2,
        (Some("AutoencoderKLFlux2"), _) => Kind::Vae,
        (_, Some("EmbeddingGemma2Model")) => Kind::Eg2,
        (_, Some("Qwen3_5ForConditionalGeneration")) => Kind::Qwen35,
        _ => v["model_type"].as_str().map(|_| Kind::Decoder)?,
    })
}

/// Each part's name: a pipeline's folders, or `WHOLE`.
pub fn names(dir: &Path) -> Result<Vec<String>> {
    let index = dir.join("model_index.json");
    let Ok(text) = std::fs::read_to_string(&index) else {
        return match kind(dir) {
            Some(_) => Ok(vec![WHOLE.into()]),
            None => Err(format!("no model the bundle loads in {}", dir.display()).into()),
        };
    };
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)?;
    let mut names = Vec::new();
    for key in map.keys() {
        if kind(&dir.join(key)).is_some() {
            names.push(key.clone());
        }
    }
    Ok(names)
}

/// Part `name` of `dir`, at the bounds `size` gives.
pub fn open(dir: &Path, name: &str, size: &dyn Fn(DimName) -> Option<u64>) -> Result<Part> {
    let whole = name == WHOLE && !dir.join("model_index.json").exists();
    let dir = if whole { dir.into() } else { dir.join(name) };
    let kind = kind(&dir).ok_or_else(|| format!("no model the bundle loads in {name}"))?;
    // Diffusers loaders find their own folder in the pipeline.
    let root = dir.parent().unwrap_or(&dir).to_path_buf();
    let (at, none) = (|n, or| size(n).unwrap_or(or), &Overrides::default());
    let seqs = at(Seqs, 1);
    let (graph, source) = match kind {
        Kind::Flux2 => {
            let graph = flux2::graph(&root, [seqs, at(Context, 512), at(Rows, 1024)], none)?;
            (graph, Source::Flux2(root))
        }
        Kind::Vae => {
            // Rows count latent tokens, the transformer's output rows.
            let rows = at(Rows, 256);
            let side = rows.isqrt();
            if side * side != rows {
                return Err(format!("{rows} rows do not make a square").into());
            }
            let c = vae::config(&root, [side, seqs], none)?;
            (
                tessera_engine::vae::program(&c),
                Source::Vae(root, c.bn_eps),
            )
        }
        Kind::Eg2 => {
            let c = encoder::config(&dir, [seqs, at(Context, 512)], none)?;
            let layers = c.stack.layers;
            (eg2::program::<Linear>(&c), Source::Eg2(dir, layers))
        }
        Kind::Qwen35 => {
            let c = qwen35::config(&dir, [seqs, at(Context, 256)], none)?;
            let layers = c.stack.layers;
            (
                tessera_engine::qwen35::program(&c),
                Source::Qwen35(dir, layers),
            )
        }
        Kind::Decoder => {
            let (arch, c) = config(&dir, [seqs, at(Context, 256)], none)?;
            let graph = stepper(&arch, &c)
                .ok_or_else(|| format!("{arch} is not a model this engine traces"))?;
            (graph, Source::Decoder(dir, c.stack.layers))
        }
    };
    Ok(Part { graph, source })
}

impl Part {
    /// The weights `graph` reads, and their forms.
    pub fn weights(&self, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
        match &self.source {
            Source::Flux2(dir) => flux2::weights(dir, graph),
            Source::Vae(dir, eps) => Ok((vae::weights(dir, *eps, graph)?, Pins::default())),
            Source::Eg2(dir, layers) => {
                Ok((encoder::weights(dir, *layers, graph)?, Pins::default()))
            }
            Source::Qwen35(dir, layers) => qwen35::weights(dir, *layers, graph),
            Source::Decoder(dir, layers) => pull(graph, &Shards::open(dir)?, *layers, &[]),
        }
    }
}
