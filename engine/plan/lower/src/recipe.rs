//! The choices lowering takes, as data a search can edit.

use crate::form::Pins;
use codegen::kept::{self, Layout};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::Path;

/// What to favor, bar and pin when lowering a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// The form each weight is stored in.
    #[serde(default)]
    pub pins: Pins,
    /// A rule to pick wherever it applies, whatever it costs.
    #[serde(default)]
    pub favor: Option<String>,
    /// Kernels never to choose, by name.
    #[serde(default)]
    pub barred: Vec<String>,
    /// How launches go to the device.
    #[serde(default)]
    pub launch: Launching,
    /// Where kept rows lie, by slot and position.
    #[serde(default = "dense", with = "by_name")]
    pub kept: &'static dyn Layout,
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
            favor: None,
            barred: Vec::new(),
            launch: Launching::default(),
            kept: dense(),
        }
    }
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(std::io::Error::other)
    }
}
