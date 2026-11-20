//! A FLUX.2 transformer from a diffusers directory.

use crate::names::repo;
use crate::{Overrides, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::File;
use tessera_engine::flux2::{Config, program};
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Transformer {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
}

/// `dir/transformer`, with `text` rows and up to `rows` image rows.
pub fn transformer(dir: &Path, sizes: [u64; 3], overrides: &Overrides) -> Result<Transformer> {
    let graph = graph(dir, sizes, overrides)?;
    let (weights, pins) = weights(dir, &graph)?;
    Ok(Transformer {
        graph,
        weights,
        pins,
    })
}

/// The transformer `dir/transformer/config.json` describes.
pub fn graph(dir: &Path, [seqs, text, rows]: [u64; 3], overrides: &Overrides) -> Result<Graph> {
    const FILE: &str = "transformer/config.json";
    let config = overrides.read(FILE, Some(&dir.join(FILE)), |k| {
        Config::read(k, seqs, text, rows)
    })?;
    Ok(program(&config))
}

/// The weights `graph` reads from `dir/transformer`, and their forms.
pub fn weights(dir: &Path, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
    let file = File::open(&dir.join("transformer/diffusion_pytorch_model.safetensors"))?;
    let rules = repo("flux2").unwrap_or_default();
    pull(graph, &file, 0, &[&rules])
}
