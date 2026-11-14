//! The report each check writes as JSON, and tessera-check reads.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// One tool's findings, or why it did not run.
#[derive(Serialize, Deserialize, Default)]
pub struct Report {
    /// The tool's short name, such as `lint`.
    pub tool: String,
    pub title: String,
    /// Each rule's name and fix, by rule id.
    #[serde(default)]
    pub rules: BTreeMap<String, Rule>,
    #[serde(default)]
    pub findings: Vec<Finding>,
    /// Set when the tool did not run, with the reason.
    #[serde(default)]
    pub skipped: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    pub fix: String,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Fails the check.
    Fail,
    /// Advice. The check still passes.
    Note,
}

#[derive(Serialize, Deserialize)]
pub struct Finding {
    pub rule: String,
    pub level: Level,
    pub path: String,
    pub line: usize,
    pub message: String,
    /// Code to show beside the message.
    #[serde(default)]
    pub snippets: Vec<Snippet>,
    /// A change that clears it, like a line to add.
    #[serde(default)]
    pub fix: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Snippet {
    pub path: String,
    pub line: usize,
    pub text: String,
}

impl Report {
    pub fn new(tool: &str, title: &str) -> Self {
        Self {
            tool: tool.into(),
            title: title.into(),
            ..Self::default()
        }
    }

    pub fn fails(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.level == Level::Fail)
            .count()
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}
