//! The choices lowering takes, as data a search can edit.

use crate::form::{Pins, matches};
use crate::rule::Rule;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::Path;
use tessera_codegen::kept::{self, Layout};
use tessera_graph::{DType, Graph, NodeId, Op, PathSeg, sums_ranks};

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
    /// A rule to pick wherever it applies, whatever it costs.
    #[serde(default)]
    pub favor: Option<String>,
    /// Fail lowering unless `favor` covers at least one step.
    #[serde(default)]
    pub require_favored: bool,
    /// Kernels never to choose, by name.
    #[serde(default)]
    pub barred: Vec<String>,
    /// How launches go to the device.
    #[serde(default)]
    pub launch: Launching,
    /// Where kept rows lie, by slot and position.
    #[serde(default = "dense", with = "by_name")]
    pub kept: &'static dyn Layout,
    /// Rules made at run time, beside the linked ones.
    #[serde(skip)]
    pub rules: Vec<&'static Rule>,
    /// Model config JSON passed to runtime rules.
    #[serde(skip)]
    pub config: Option<String>,
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
            favor: None,
            require_favored: false,
            barred: Vec::new(),
            launch: Launching::default(),
            kept: dense(),
            rules: Vec::new(),
            config: None,
        }
    }
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(std::io::Error::other)
    }
}
