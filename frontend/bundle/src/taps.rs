//! A decoder's residual rows after chosen layers, as text features.

use crate::arch::family;
use crate::tokenize::hf::Hf;
use crate::{Overrides, Template, config, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Shards;
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Tapped {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
    pub tokenizer: Hf,
    pub template: Template,
    /// The id prompts are padded with.
    pub pad: u32,
}

/// The tokenizer config's pad token, as an id.
fn pad(dir: &Path, tokenizer: &Hf) -> Result<u32> {
    let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
    let config: serde_json::Value = serde_json::from_str(&text)?;
    let token = config["pad_token"].as_str().ok_or("no pad token")?;
    Ok(tokenizer.id(token).ok_or("the pad token has no id")?)
}

/// The model in `dir`, cut after its deepest tap.
pub fn tapped(
    [dir, words]: [&Path; 2],
    [seqs, context]: [u64; 2],
    layers: &[u64],
    overrides: &Overrides,
) -> Result<Tapped> {
    let (arch, mut config) = config(dir, [seqs, context], overrides)?;
    let deepest = layers.iter().copied().max().ok_or("no layers to tap")?;
    if deepest > config.stack.layers {
        return Err(format!("layer {deepest} of {}", config.stack.layers).into());
    }
    config.stack.layers = deepest;
    let graph = (family(&arch)?.taps)(&config, layers);
    let (weights, pins) = pull(&graph, &Shards::open(dir)?, deepest, &[])?;
    let tokenizer = Hf::open(&words.join("tokenizer.json"))?;
    let template = Template::load(words)?.ok_or("no chat template")?;
    Ok(Tapped {
        graph,
        weights,
        pins,
        pad: pad(words, &tokenizer)?,
        tokenizer,
        template,
    })
}
