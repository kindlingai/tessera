//! The parts of a model directory, by name.

use crate::Overrides;
use crate::arch::{self, Arch};
use crate::names::Rule;
use std::error::Error;
use std::path::{Path, PathBuf};
use tessera_checkpoint::Offer;
use tessera_engine::DimName::{self, Context, Layers, Seqs};
use tessera_engine::wasm::Module;
use tessera_engine::{Graph, Pins, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// The name of a directory that holds one model.
const WHOLE: &str = "model";

/// Rank slice selector for one weight index.
pub(crate) type Shares<'a> = &'a dyn Fn(u32) -> Option<(u32, usize, usize)>;

/// Reads the weights a graph cut from a part needs.
type Load = Box<dyn Fn(&Graph, Shares<'_>) -> Result<(Vec<Weight>, Pins)> + Send + Sync>;

/// A graph whose weights load later, for any cut.
pub struct Part {
    pub graph: Graph,
    /// Header-only weight forms, when known ahead of loading.
    pub pins: Option<Pins>,
    /// Merged `config.json` with recipe overrides applied.
    pub config: Option<String>,
    load: Load,
}

impl Part {
    pub(crate) fn new(
        graph: Graph,
        load: impl Fn(&Graph) -> Result<(Vec<Weight>, Pins)> + Send + Sync + 'static,
    ) -> Self {
        Self {
            graph,
            pins: None,
            config: None,
            load: Box::new(move |g, _| load(g)),
        }
    }

    pub(crate) fn with_meta(
        graph: Graph,
        pins: Option<Pins>,
        config: Option<String>,
        load: impl Fn(&Graph, Shares<'_>) -> Result<(Vec<Weight>, Pins)> + Send + Sync + 'static,
    ) -> Self {
        Self {
            graph,
            pins,
            config,
            load: Box::new(load),
        }
    }

    /// The weights `graph` reads, and their forms.
    pub fn weights(&self, graph: &Graph) -> Result<(Vec<Weight>, Pins)> {
        self.weights_shared(graph, &|_| None)
    }

    /// The weights `graph` reads, sliced by `shares` where split.
    pub fn weights_shared(
        &self,
        graph: &Graph,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
    ) -> Result<(Vec<Weight>, Pins)> {
        (self.load)(graph, shares)
    }
}

/// Builds a `Part` backed by a checkpoint `Offer`.
pub(crate) fn from_offer(
    graph: Graph,
    dir: PathBuf,
    manifest: Option<&str>,
    layers: u64,
    rules: Vec<Rule>,
    user: Vec<Rule>,
    config: String,
) -> Result<Part> {
    let offer = crate::offer(&dir, manifest)?;
    let pins = crate::pins(&graph, offer.as_ref(), layers, &[&rules, &user])?;
    let manifest = manifest.map(str::to_string);
    Ok(Part::with_meta(
        graph,
        Some(pins),
        Some(config),
        move |g, shares| {
            let offer = crate::offer(&dir, manifest.as_deref())?;
            crate::pull_shared(g, offer.as_ref(), layers, &[&rules, &user], shares)
        },
    ))
}

/// Builds a `Part` for an architecture with optional rotated experts.
pub(crate) fn from_streamed<C>(
    cut: &Cut,
    cfg: C,
    layers: u64,
    make_rules: impl FnOnce(&C, bool) -> Vec<Rule>,
    traces: [fn(&C) -> Graph; 2],
) -> Result<Part> {
    from_opened(cut, cfg, layers, make_rules, traces, crate::offer)
}

/// How a part opens its checkpoint.
pub(crate) type Open = fn(&Path, Option<&str>) -> Result<Box<dyn Offer>>;

/// `from_streamed`, its checkpoint opened by `open`.
pub(crate) fn from_opened<C>(
    cut: &Cut,
    cfg: C,
    layers: u64,
    make_rules: impl FnOnce(&C, bool) -> Vec<Rule>,
    traces: [fn(&C) -> Graph; 2],
    open: Open,
) -> Result<Part> {
    let offer = open(&cut.dir, cut.manifest)?;
    let rotated = crate::mla::has_rotated_experts(&offer.slices());
    let rules = make_rules(&cfg, rotated);
    let graph = traces[usize::from(rotated)](&cfg);
    let pins = crate::pins(&graph, offer.as_ref(), layers, &[&rules, cut.user])?;
    let (dir, manifest, user) = (
        cut.dir.clone(),
        cut.manifest.map(str::to_string),
        cut.user.to_vec(),
    );
    Ok(Part::with_meta(
        graph,
        Some(pins),
        Some(cut.json.to_string()),
        move |g, shares| {
            let offer = open(&dir, manifest.as_deref())?;
            crate::pull_shared(g, offer.as_ref(), layers, &[&rules, &user], shares)
        },
    ))
}

/// Options for opening one model part with recipe settings.
pub struct Spec<'a> {
    pub overrides: &'a Overrides,
    pub user: &'a [Rule],
    pub manifest: Option<&'a str>,
    pub module: Option<&'a Module>,
    pub size: &'a dyn Fn(DimName) -> Option<u64>,
}

/// A part's folder, recipe overrides and graph bounds.
pub(crate) struct Cut<'a> {
    pub dir: PathBuf,
    pub overrides: &'a Overrides,
    pub user: &'a [Rule],
    pub manifest: Option<&'a str>,
    pub json: &'a str,
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
fn kind(dir: &Path, overrides: &Overrides) -> Option<&'static Arch> {
    let text = overrides
        .merged("config.json", &dir.join("config.json"))
        .ok()?;
    arch::spot(&text).ok().filter(|a| a.part.is_some())
}

/// As `names`, honoring recipe config overrides and graph modules.
pub fn names_with(
    dir: &Path,
    overrides: &Overrides,
    module: Option<&Module>,
) -> Result<Vec<String>> {
    let index = dir.join("model_index.json");
    let Ok(text) = std::fs::read_to_string(&index) else {
        if module.is_some_and(Module::builds) && dir.join("config.json").exists() {
            return Ok(vec![WHOLE.into()]);
        }
        return match kind(dir, overrides) {
            Some(_) => Ok(vec![WHOLE.into()]),
            None => Err(format!("no model the bundle loads in {}", dir.display()).into()),
        };
    };
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&text)?;
    let mut names = Vec::new();
    for key in map.keys() {
        let sub = overrides.under(key);
        if kind(&dir.join(key), &sub).is_some() {
            names.push(key.clone());
        }
    }
    Ok(names)
}

/// Part `name` of `dir`, honoring recipe overrides and modules.
pub fn open_with(dir: &Path, name: &str, spec: &Spec<'_>) -> Result<Part> {
    let whole = name == WHOLE && !dir.join("model_index.json").exists();
    let part_dir = if whole { dir.into() } else { dir.join(name) };
    let scoped = (!whole).then(|| spec.overrides.under(name));
    let overrides = scoped.as_ref().unwrap_or(spec.overrides);
    let json = overrides
        .merged("config.json", &part_dir.join("config.json"))
        .map_err(|_| format!("no model the bundle loads in {name}"))?;
    if whole
        && let Some(module) = spec.module
        && module.builds()
    {
        let seqs = (spec.size)(Seqs).unwrap_or(1);
        let context = (spec.size)(Context).unwrap_or(256);
        if let Some(graph) = module.graph(&json, seqs, Some(context))? {
            let layers = graph
                .dims
                .iter()
                .find(|d| d.name == Layers)
                .map_or(0, |d| d.size.max());
            return from_offer(
                graph,
                part_dir,
                spec.manifest,
                layers,
                Vec::new(),
                spec.user.to_vec(),
                json,
            );
        }
    }
    let arch = arch::spot(&json)?;
    let build = arch
        .part
        .ok_or_else(|| format!("{} has no part graph", arch.name))?;
    build(
        arch,
        &Cut {
            dir: part_dir,
            overrides,
            user: spec.user,
            manifest: spec.manifest,
            json: &json,
            size: spec.size,
        },
    )
}
