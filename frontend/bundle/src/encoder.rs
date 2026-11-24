//! Text embedding models, and the EmbeddingGemma 2 exlr8 release.

use crate::arch::{Arch, Spot, arch, read};
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
    embed: Embeds {
        shape: Shape {
            max_tokens: 1024,
            seqs: 16,
            rows: 4096,
        },
        open: encoder,
    },
);

/// The launch sizes an embedder plans for.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Tokens per text, `<bos>` and `<eos>` included.
    pub max_tokens: usize,
    /// The most texts in one launch.
    pub seqs: usize,
    /// The most padded rows in one launch.
    pub rows: usize,
}

/// An embedding model's launch sizes and text tower.
pub(crate) struct Embeds {
    pub shape: Shape,
    /// The tower at `[seqs, max_tokens]`, plain or padded.
    pub open: fn(&Path, [u64; 2], bool, &Overrides) -> Result<Encoder>,
}

pub struct Encoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub tokenizer: Hf,
}

/// The embedding model in `dir` at its own launch sizes.
pub fn open(dir: &Path, padded: bool, overrides: &Overrides) -> Result<(Shape, Encoder)> {
    let (_, arch) = read(dir, overrides)?;
    let embeds = arch.embed.as_ref();
    let embeds = embeds.ok_or_else(|| format!("{} is not an embedding model", arch.name))?;
    let shape = embeds.shape;
    let at = [shape.seqs as u64, shape.max_tokens as u64];
    Ok((shape, (embeds.open)(dir, at, padded, overrides)?))
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
