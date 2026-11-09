//! The graph's JSON form, its version and its hash.

use crate::node::Graph;
use serde::Deserialize;
use std::fmt;

/// Bumped whenever the JSON form changes meaning.
pub const VERSION: u32 = 3;

#[derive(Debug)]
pub enum FormatError {
    Json(serde_json::Error),
    Version { found: u32 },
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::Json(e) => write!(f, "graph JSON: {e}"),
            FormatError::Version { found } => {
                write!(
                    f,
                    "graph format version {found}, this build reads {VERSION}"
                )
            }
        }
    }
}

impl std::error::Error for FormatError {}

#[derive(Deserialize)]
struct Header {
    version: u32,
}

impl Graph {
    /// Field order is fixed, so equal graphs give equal text.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("graphs always serialize")
    }

    /// Checks the version before reading anything else.
    pub fn from_json(text: &str) -> Result<Graph, FormatError> {
        let header: Header = serde_json::from_str(text).map_err(FormatError::Json)?;
        if header.version != VERSION {
            return Err(FormatError::Version {
                found: header.version,
            });
        }
        serde_json::from_str(text).map_err(FormatError::Json)
    }

    /// FNV-1a over the compact JSON form.
    pub fn hash(&self) -> u64 {
        let text = serde_json::to_vec(self).expect("graphs always serialize");
        text.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    }
}
