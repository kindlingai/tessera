//! A Qwen3.5 Hugging Face directory: its text model and weights.

use crate::names::{Rule, repo};
use crate::{Overrides, each, pins, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Shards;
use tessera_engine::qwen35::{Config, program};
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// The Hugging Face class of a Qwen3.5 directory.
pub(crate) const CLASS: &str = "Qwen3_5ForConditionalGeneration";

fn rules() -> Vec<Rule> {
    repo("qwen35").expect("the repository has Qwen3.5 rules")
}

/// The text model's logits graph, its forms and its files.
pub struct Text {
    pub graph: Graph,
    pub pins: Pins,
    shards: Shards,
    rules: Vec<Rule>,
}

impl Text {
    /// Each weight in graph order, one read at a time.
    pub fn each(&self, mut take: impl FnMut(u32, Weight) -> Result<()>) -> Result<()> {
        let mut at = 0;
        each(
            &self.graph,
            &self.shards,
            &[&self.rules],
            &|_| None,
            |w, _| {
                take(at, w)?;
                at += 1;
                Ok(())
            },
        )
    }
}

/// The text model in `dir`: `seqs` sequences of `context` rows.
pub fn open(dir: &Path, sizes: [u64; 2], overrides: &Overrides) -> Result<Text> {
    let c = config(dir, sizes, overrides)?;
    let graph = program(&c);
    let (shards, rules) = (Shards::open(dir)?, rules());
    let pins = pins(&graph, &shards, c.stack.layers, &[&rules])?;
    Ok(Text {
        graph,
        pins,
        shards,
        rules,
    })
}

/// The text model `dir/config.json` describes.
pub fn config(dir: &Path, [seqs, context]: [u64; 2], overrides: &Overrides) -> Result<Config> {
    overrides.config(dir, |k| Config::read(k, seqs, context))
}

/// Every weight `graph` reads from `dir`, held, and their forms.
pub fn weights(dir: &Path, layers: u64, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
    pull(graph, &Shards::open(dir)?, layers, &[&rules()])
}
