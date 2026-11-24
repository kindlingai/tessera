//! The parts of a model directory, by name.

use crate::arch::{self, Arch};
use std::error::Error;
use std::path::{Path, PathBuf};
use tessera_engine::DimName::{self, Context, Seqs};
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// The name of a directory that holds one model.
const WHOLE: &str = "model";

/// Reads the weights a graph cut from a part needs.
type Load = Box<dyn Fn(&Graph) -> Result<(Vec<Weight>, Pins)> + Send + Sync>;

/// A graph whose weights load later, for any cut.
pub struct Part {
    pub graph: Graph,
    load: Load,
}

impl Part {
    pub(crate) fn new(
        graph: Graph,
        load: impl Fn(&Graph) -> Result<(Vec<Weight>, Pins)> + Send + Sync + 'static,
    ) -> Self {
        Self {
            graph,
            load: Box::new(load),
        }
    }

    /// The weights `graph` reads, and their forms.
    pub fn weights(&self, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
        (self.load)(graph)
    }
}

/// A part's folder and its graph's bounds.
pub(crate) struct Cut<'a> {
    pub dir: PathBuf,
    size: &'a dyn Fn(DimName) -> Option<u64>,
}

impl Cut<'_> {
    /// The pipeline folder. Diffusers loaders find their own folder.
    pub fn root(&self) -> PathBuf {
        self.dir.parent().unwrap_or(&self.dir).to_path_buf()
    }

    /// Bound `name`, or `or` when the caller leaves it open.
    pub fn at(&self, name: DimName, or: u64) -> u64 {
        (self.size)(name).unwrap_or(or)
    }

    /// One sequence unless set, of `context` rows unless set.
    pub fn sized(&self, context: u64) -> [u64; 2] {
        [self.at(Seqs, 1), self.at(Context, context)]
    }
}

/// The architecture in `dir`, when one has part graphs.
fn kind(dir: &Path) -> Option<&'static Arch> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    arch::spot(&text).ok().filter(|a| a.part.is_some())
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
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|_| format!("no model the bundle loads in {name}"))?;
    let arch = arch::spot(&text)?;
    let build = arch
        .part
        .ok_or_else(|| format!("{} has no part graph", arch.name))?;
    build(arch, &Cut { dir, size })
}
