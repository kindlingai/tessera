//! A FLUX.2 transformer from a diffusers directory.

use crate::names::repo;
use crate::{Overrides, pull};
use sensai_checkpoint::File;
use sensai_engine::flux2::{Config, program};
use sensai_engine::{Graph, Pins, Weight};
use std::error::Error;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Transformer {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
}

/// `dir/transformer`, with `text` rows and up to `rows` image rows.
pub fn transformer(
    dir: &Path,
    [seqs, text, rows]: [u64; 3],
    config: &Overrides,
) -> Result<Transformer> {
    const FILE: &str = "transformer/config.json";
    let config = config.read(FILE, Some(&dir.join(FILE)), |k| {
        Config::read(k, seqs, text, rows)
    })?;
    let graph = program(&config);
    let file = File::open(&dir.join("transformer/diffusion_pytorch_model.safetensors"))?;
    let rules = repo("flux2").unwrap_or_default();
    let (weights, pins) = pull(&graph, &file, 0, &[&rules])?;
    Ok(Transformer {
        graph,
        weights,
        pins,
    })
}
