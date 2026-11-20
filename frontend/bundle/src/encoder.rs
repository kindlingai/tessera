//! An EmbeddingGemma 2 exlr8 release: graph, weights, tokenizer.

use crate::names::repo;
use crate::tokenize::hf::Hf;
use crate::{configured, pull};
use sensai_checkpoint::exlr8::Release;
use sensai_engine::eg2::{Config, batched, program};
use sensai_engine::{Graph, Linear, Weight};
use std::error::Error;
use std::path::Path;

pub struct Encoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

/// The text tower in `dir`, plain or padded.
pub fn encoder(
    dir: &Path,
    [seqs, context]: [u64; 2],
    padded: bool,
) -> Result<Encoder, Box<dyn Error + Send + Sync>> {
    let config = configured(&dir.join("config.json"), |k| Config::read(k, seqs, context))?;
    let graph = match padded {
        true => batched::<Linear>(&config),
        false => program::<Linear>(&config),
    };
    let rules = repo("eg2").unwrap_or_default();
    let release = Release::open(dir);
    let (weights, _) = pull(&graph, &release, config.stack.layers, &[&rules])?;
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    Ok(Encoder {
        graph,
        weights,
        tokenizer,
    })
}
