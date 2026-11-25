//! A Qwen3.5 Hugging Face directory: its text model and weights.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::names::{Rule, repo};
use crate::parts::{Cut, Part};
use crate::{Overrides, Stream, Weights, numbered, pins, pull};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Shards;
use tessera_engine::qwen35::{Config, program};
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// The Hugging Face class of a Qwen3.5 directory.
const CLASS: &str = "Qwen3_5ForConditionalGeneration";

fn rules() -> Vec<Rule> {
    repo("qwen35").expect("the repository has Qwen3.5 rules")
}

fn step(_: &Arch, _: &Ask) -> Result<Built> {
    Err(format!("{CLASS} has no step graph yet. tessera-eval --forward runs it").into())
}

fn forward(_: &Arch, ask: &Ask) -> Result<Built> {
    let text = open(ask.dir, ask.sizes, ask.overrides)?;
    let (graph, pins) = (text.graph.clone(), text.pins.clone());
    Ok((graph, Weights::Streamed(Box::new(text)), pins))
}

fn part(_: &Arch, cut: &Cut) -> Result<Part> {
    let c = config(&cut.dir, cut.sized(256), &Overrides::default())?;
    let (dir, layers) = (cut.dir.clone(), c.stack.layers);
    Ok(Part::new(program(&c), move |g| weights(&dir, layers, g)))
}

arch!(
    "qwen35",
    [Spot::Class(CLASS), Spot::Type("qwen3_5")],
    rules: include_str!("../names/qwen35.json"),
    step: step,
    forward: forward,
    part: part,
);

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
        self.stream(&|_| None, &mut take)
    }
}

impl Stream for Text {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        numbered(&self.graph, &self.shards, &[&self.rules], shares, take)
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
