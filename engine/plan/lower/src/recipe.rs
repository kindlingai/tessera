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
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(std::io::Error::other)
    }
}
