//! An EmbeddingGemma 2 exlr8 release: graph, weights, tokenizer.

use crate::arch::{Arch, Spot, arch};
use crate::names::repo;
use crate::parts::{Cut, Part};
use crate::tokenize::hf::Hf;
use crate::{Overrides, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::exlr8::Release;
use tessera_engine::eg2::{Config, batched, program};
use tessera_engine::{Graph, Linear, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn part(_: &Arch, cut: &Cut) -> Result<Part> {
    let c = config(&cut.dir, cut.sized(512), &Overrides::default())?;
    let (dir, layers) = (cut.dir.clone(), c.stack.layers);
    Ok(Part::new(program::<Linear>(&c), move |g| {
        Ok((weights(&dir, layers, g)?, Pins::default()))
    }))
}

arch!(
    "eg2",
    [Spot::Class("EmbeddingGemma2Model")],
    rules: include_str!("../names/eg2.json"),
    part: part,
);

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
