//! An EmbeddingGemma 2 exlr8 release: graph, weights, tokenizer.

use crate::names::repo;
use crate::tokenize::hf::Hf;
use crate::{Overrides, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::exlr8::Release;
use tessera_engine::eg2::{Config, batched, program};
use tessera_engine::{Graph, Linear, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Encoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

/// The text tower in `dir`, plain or padded.
pub fn encoder(
    dir: &Path,
    sizes: [u64; 2],
    padded: bool,
    overrides: &Overrides,
) -> Result<Encoder> {
    let config = config(dir, sizes, overrides)?;
    let graph = match padded {
        true => batched::<Linear>(&config),
        false => program::<Linear>(&config),
    };
    let weights = weights(dir, config.stack.layers, &graph)?;
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    Ok(Encoder {
        graph,
        weights,
        tokenizer,
    })
}

/// The text tower `dir/config.json` describes.
pub fn config(dir: &Path, [seqs, context]: [u64; 2], overrides: &Overrides) -> Result<Config> {
    overrides.config(dir, |k| Config::read(k, seqs, context))
}

/// The weights `graph` reads from the release in `dir`.
pub fn weights(dir: &Path, layers: u64, graph: &Graph) -> Result<Vec<Weight>> {
    let rules = repo("eg2").unwrap_or_default();
    Ok(pull(graph, &Release::open(dir), layers, &[&rules])?.0)
}
