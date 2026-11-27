//! The choices lowering takes, as data a search can edit.

use crate::form::{Load, Pins, Site, matches, named};
use crate::rule::{Rule, gather};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::Path;
use tessera_codegen::kept::{self, Layout};
use tessera_graph::{DType, Graph, NodeId, Op, PathSeg, Region, layers, sums_ranks};

/// Operand precision a matrix unit may round to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compute {
    /// Keep f32 precision through multi-pass splits.
    #[default]
    Exact,
    /// Native 1-pass bf16 MMA with f32 accumulation.
    Bf16,
}

/// Choices to favor, bar and pin when lowering a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// The form each weight, activation, kept row and output uses.
    #[serde(default)]
    pub pins: Pins,
    /// The dtype each cross-rank sum's parts cross as.
    #[serde(default)]
    pub sums: Wires,
    /// Operand precision for matrix-unit kernels.
    #[serde(default)]
    pub compute: Compute,
    /// Dtype overrides by spot. `settle` folds them in.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dtype: Vec<Dtype>,
    /// Kernel shape values by `kernel:knob`.
    #[serde(default, skip_serializing_if = "Knobs::is_empty")]
    pub shape: Knobs,
    /// A rule to pick wherever it applies, whatever it costs.
    #[serde(default)]
    pub favor: Option<String>,
    /// Fail lowering unless `favor` covers at least one step.
    #[serde(default)]
    pub require_favored: bool,
    /// Kernels never to choose, by name.
    #[serde(default)]
    pub barred: Vec<String>,
    /// Choices for the nodes of some regions only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scoped: Vec<Scoped>,
    /// How launches go to the device.
    #[serde(default)]
    pub launch: Launching,
    /// Where kept rows lie, by slot and position.
    #[serde(default = "dense", with = "by_name")]
    pub kept: &'static dyn Layout,
    /// Shared paged KV row capacity across all slots.
    #[serde(default)]
    pub kept_rows: Option<usize>,
    /// Split `(kv_vram_rows, index_vram_rows)` for sparse MLA models.
    #[serde(default)]
    pub index_rows: Option<(usize, usize)>,
    /// Rules made at run time, beside the linked ones.
    #[serde(skip)]
    pub rules: Vec<&'static Rule>,
    /// Model config JSON passed to runtime rules.
    #[serde(skip)]
    pub config: Option<String>,
}

/// Kernel choices for the nodes of the regions `at` names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scoped {
    /// A pattern over labels such as `layer.3.attention`. `*` is any run.
    pub at: String,
    /// A rule that wins here wherever it applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub favor: Option<String>,
    /// Rules and kernels never chosen here, by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub barred: Vec<String>,
    /// A kernel whose way wins here wherever it is offered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pick: Option<String>,
    /// Matrix-unit operand precision here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<Compute>,
}

/// Each node's label: `head`, `layer.<k>[.<kind>]` or `tail`.
pub fn labels(g: &Graph) -> Vec<String> {
    let layers = layers(g);
    let first = layers
        .iter()
        .map(|(_, i)| g.regions[*i].start.0 as usize)
        .min()
        .unwrap_or(0);
    let mut out: Vec<String> = (0..g.nodes.len())
        .map(|i| if i < first { "head" } else { "tail" }.into())
        .collect();
    let mut set = |r: &Region, label: String| {
        for at in &mut out[r.start.0 as usize..r.end.0 as usize] {
            at.clone_from(&label);
        }
    };
    let kind = |r: &Region| format!("{:?}", r.kind).to_lowercase();
    let inside = |r: &Region, o: &Region| o.start <= r.start && r.end <= o.end && r != o;
    for (k, i) in &layers {
        let layer = &g.regions[*i];
        set(layer, format!("layer.{k}"));
        for r in g.regions.iter().filter(|r| inside(r, layer)) {
            set(r, format!("layer.{k}.{}", kind(r)));
        }
    }
    // A trimmed picking layer keeps only its blocks.
    for r in g.regions.iter().filter(|r| r.kind.block()) {
        let held = layers.iter().any(|(_, i)| inside(r, &g.regions[*i]));
        let before = layers.iter().filter(|(_, i)| g.regions[*i].end <= r.start);
        let last = before
            .max_by_key(|(_, i)| g.regions[*i].end)
            .map(|(k, _)| k + 1);
        if let Some(k) = last.filter(|_| !held) {
            set(r, format!("layer.{k}.{}", kind(r)));
        }
    }
    out
}

/// Wire dtypes for cross-rank sums, below model pins.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wires {
    /// The wire of sums no pattern names, else setup's.
    #[serde(default)]
    pub all: Option<DType>,
    /// Patterns over each sum's weight path. The last match wins.
    #[serde(default)]
    pub named: Vec<(String, DType)>,
}

impl Wires {
    /// Pins the sums of `g` the model left unpinned.
    pub fn pin(&self, mut g: Graph) -> Graph {
        for at in 0..g.nodes.len() {
            let n = &g.nodes[at];
            if !sums_ranks(&n.op) || n.numerics.is_some_and(|m| m.wire.is_some()) {
                continue;
            }
            let name = source(&g, n.args[0]).map(PathSeg::dotted);
            let named =
                name.and_then(|name| self.named.iter().rev().find(|(p, _)| matches(p, &name)));
            let wire = named.map(|(_, w)| *w).or(self.all);
            if let (Some(wire), Some(num)) = (wire, &mut g.nodes[at].numerics) {
                num.wire = Some(wire);
            }
        }
        g
    }
}

/// The weight path of the first matmul below `id`.
fn source(g: &Graph, mut id: NodeId) -> Option<&[PathSeg]> {
    loop {
        let n = g.node(id);
        id = match &n.op {
            Op::Weight(w) => return Some(&g.weights[*w as usize].path),
            Op::Matmul => n.args[1],
            _ => *n.args.first()?,
        };
    }
}

fn dense() -> &'static dyn Layout {
    kept::DENSE
}

/// A layout in a recipe is its name.
mod by_name {
    use super::*;

    pub fn serialize<S: Serializer>(l: &&'static dyn Layout, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(l.name())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<&'static dyn Layout, D::Error> {
        let name = String::deserialize(d)?;
        kept::named(&name).ok_or_else(|| serde::de::Error::custom(format!("no layout {name}")))
    }
}

/// Launch-time choices, each taken where the device allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Launching {
    /// Record a whole plan once and replay it.
    pub graphs: bool,
    /// Let each generated kernel start before the last one ends.
    pub pdl: bool,
}

impl Default for Launching {
    fn default() -> Self {
        Self {
            graphs: true,
            pdl: true,
        }
    }
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            pins: Pins::default(),
            sums: Wires::default(),
            compute: Compute::default(),
            dtype: Vec::new(),
            shape: Knobs::default(),
            favor: None,
            require_favored: false,
            barred: Vec::new(),
            scoped: Vec::new(),
            launch: Launching::default(),
            kept: dense(),
            kept_rows: None,
            index_rows: None,
            rules: Vec::new(),
            config: None,
        }
    }
}

impl Recipe {
    /// Shares a paged KV pool of `rows` rows across slots.
    pub fn share_kept(&mut self, rows: usize) {
        self.kept_rows = Some(rows);
        if self.kept.page().is_none() {
            self.kept = kept::named("paged").expect("paged layout");
        }
    }

    /// Sizes sparse KV and dense index VRAM pools separately.
    pub fn share_index(&mut self, kv_rows: usize, index_rows: usize) {
        self.index_rows = Some((kv_rows, index_rows));
    }

    /// Folds the dtype rules into the pins, sums and compute.
    pub fn settle(&mut self) {
        for rule in std::mem::take(&mut self.dtype) {
            rule.apply(self);
        }
    }

    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        let mut recipe: Self = serde_json::from_str(&text).map_err(std::io::Error::other)?;
        recipe.settle();
        Ok(recipe)
    }
}

const SITES: [(&str, Site); 7] = [
    ("norm", Site::Norm),
    ("softmax", Site::Softmax),
    ("logits", Site::Logits),
    ("router", Site::Router),
    ("residual", Site::Residual),
    ("attention", Site::Attention),
    ("matmul", Site::Matmul),
];

/// The setting one rule changes.
#[derive(Clone, Debug, PartialEq)]
enum Set {
    Activations(&'static dyn Load),
    Site(Site, &'static dyn Load),
    Kept(&'static dyn Load),
    Outputs(&'static dyn Load),
    Wire(DType),
    Operand(Compute),
}

/// One `[path][@spot][:property]=dtype` override.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "(String, String)", into = "(String, String)")]
pub struct Dtype {
    spot: String,
    value: String,
    /// A pattern over weight paths. Empty fits every spot.
    path: String,
    set: Set,
}

fn store(value: &str) -> Result<&'static dyn Load, String> {
    match named(value) {
        Some(f) if f.stores() && f.dtype().is_float() => Ok(f),
        Some(_) => Err(format!("form {value} is not a writable float")),
        None => Err(format!("no form called {value}")),
    }
}

fn wire(value: &str) -> Result<DType, String> {
    match value {
        "f32" => Ok(DType::F32),
        "bf16" => Ok(DType::Bf16),
        "f16" => Ok(DType::F16),
        "fp8" => Ok(DType::Fp8),
        _ => Err(format!("no wire dtype called {value}")),
    }
}

fn operand(value: &str) -> Result<Compute, String> {
    match value {
        "f32" => Ok(Compute::Exact),
        "bf16" => Ok(Compute::Bf16),
        _ => Err(format!("no operand dtype called {value}")),
    }
}

impl Dtype {
    /// Parses a spot and its dtype.
    pub fn new(spot: &str, value: &str) -> Result<Self, String> {
        let (rest, property) = match spot.rsplit_once(':') {
            Some((rest, property)) => (rest, Some(property)),
            None => (spot, None),
        };
        let (path, kind) = match rest.rsplit_once('@') {
            Some((path, kind)) => (path, Some(kind)),
            None => (rest, None),
        };
        let site = |kind: &str| SITES.iter().find(|(name, _)| *name == kind).map(|s| s.1);
        let set = match (kind, property) {
            (Some("sum") | None, Some("wire")) | (Some("sum"), None) => Set::Wire(wire(value)?),
            (Some("matmul") | None, Some("operand")) => Set::Operand(operand(value)?),
            (Some("activation"), Some("store") | None) | (None, Some("store")) => {
                Set::Activations(store(value)?)
            }
            (Some("kept"), Some("store") | None) => Set::Kept(store(value)?),
            (Some("output"), Some("store") | None) => Set::Outputs(store(value)?),
            (Some(kind), Some("store") | None) => match site(kind) {
                Some(site) => Set::Site(site, store(value)?),
                None => return Err(format!("no spot called {kind} in `{spot}`")),
            },
            _ => return Err(format!("`{spot}` names no dtype a rule can set")),
        };
        if !path.is_empty() && !matches!(set, Set::Wire(_)) {
            return Err(format!("`{spot}`: only sum wires take a path"));
        }
        Ok(Self {
            spot: spot.into(),
            value: value.into(),
            path: path.into(),
            set,
        })
    }

    /// Parses `spot=dtype`, the flag form.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text.rsplit_once('=') {
            Some((spot, value)) => Self::new(spot, value),
            None => Err(format!("`{text}` is not spot=dtype")),
        }
    }

    /// Sets this rule's dtype in `recipe`.
    fn apply(&self, recipe: &mut Recipe) {
        let pins = std::mem::take(&mut recipe.pins);
        recipe.pins = match self.set {
            Set::Activations(form) => pins.activations(form),
            Set::Site(site, form) => pins.site(site, form),
            Set::Kept(form) => pins.kept(form),
            Set::Outputs(form) => pins.outputs(form),
            Set::Wire(_) | Set::Operand(_) => pins,
        };
        match self.set {
            Set::Wire(wire) if self.path.is_empty() => {
                recipe.sums.all = Some(wire);
                recipe.sums.named.push(("*".into(), wire));
            }
            Set::Wire(wire) => recipe.sums.named.push((self.path.clone(), wire)),
            Set::Operand(compute) => recipe.compute = compute,
            _ => {}
        }
    }
}

impl TryFrom<(String, String)> for Dtype {
    type Error = String;

    fn try_from((spot, value): (String, String)) -> Result<Self, String> {
        Self::new(&spot, &value)
    }
}

impl From<Dtype> for (String, String) {
    fn from(d: Dtype) -> Self {
        (d.spot, d.value)
    }
}

/// One shape number a kernel reads, with its legal range.
pub struct Knob {
    pub kernel: &'static str,
    pub name: &'static str,
    pub least: u64,
    pub most: u64,
}

gather! {
    #[gather]
    pub static KNOBS: scattered_collect::ScatteredSlice<Knob>;
}

/// Declares a knob that shape rules may set.
#[macro_export]
macro_rules! knob {
    ($kernel:literal, $name:literal, $least:expr, $most:expr) => {
        const _: () = {
            $crate::rule::scatter! {
                #[scatter($crate::KNOBS)]
                static KNOB: $crate::Knob = $crate::Knob {
                    kernel: $kernel,
                    name: $name,
                    least: $least,
                    most: $most,
                };
            }
        };
    };
}

/// Kernel shape values by `kernel:knob`. The last match wins.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<(String, u64)>", into = "Vec<(String, u64)>")]
pub struct Knobs(Vec<(String, u64)>);

impl Knobs {
    /// Sets `spot`, a `kernel:knob`, to `value`.
    pub fn set(&mut self, spot: &str, value: u64) -> Result<(), String> {
        let named = |k: &&Knob| spot.split_once(':') == Some((k.kernel, k.name));
        match KNOBS.iter().find(named) {
            Some(k) if (k.least..=k.most).contains(&value) => {
                self.0.push((spot.into(), value));
                Ok(())
            }
            Some(k) => Err(format!("`{spot}` takes {} to {}", k.least, k.most)),
            None => Err(format!("no kernel knob called `{spot}`")),
        }
    }

    /// Parses `kernel:knob=value`, the flag form.
    pub fn parse(&mut self, text: &str) -> Result<(), String> {
        let pair = text.rsplit_once('=');
        match pair.and_then(|(spot, v)| Some((spot, v.parse().ok()?))) {
            Some((spot, value)) => self.set(spot, value),
            None => Err(format!("`{text}` is not kernel:knob=value")),
        }
    }

    /// The value set for `knob` of `kernel`, if any.
    pub fn get(&self, kernel: &str, knob: &str) -> Option<usize> {
        let named = |(spot, _): &&(String, u64)| spot.split_once(':') == Some((kernel, knob));
        self.0.iter().rev().find(named).map(|(_, v)| *v as usize)
    }

    /// These values, then `later`'s.
    pub fn then(&self, later: &Knobs) -> Knobs {
        Knobs(self.0.iter().chain(&later.0).cloned().collect())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl TryFrom<Vec<(String, u64)>> for Knobs {
    type Error = String;

    fn try_from(pairs: Vec<(String, u64)>) -> Result<Self, String> {
        let mut knobs = Knobs::default();
        for (spot, value) in pairs {
            knobs.set(&spot, value)?;
        }
        Ok(knobs)
    }
}

impl From<Knobs> for Vec<(String, u64)> {
    fn from(k: Knobs) -> Self {
        k.0
    }
}
