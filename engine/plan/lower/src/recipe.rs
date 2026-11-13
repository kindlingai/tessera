//! The choices lowering takes, as data a search can edit.

use crate::form::Pins;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// What to favor, bar and pin when lowering a model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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

impl Recipe {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(std::io::Error::other)
    }
}
