//! Rules from `rules/<id>/`: rule.yaml, rule.md and fixes.txt.

use crate::yaml;
use std::path::{Path, PathBuf};

pub struct Rule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub stage: u32,
    pub detectors: Vec<String>,
    /// Detectors that read Markdown and other prose files only.
    pub prose_detectors: Vec<String>,
    /// False when a `style-ok` stamp cannot clear a hit.
    pub stampable: bool,
    /// Drops hits on a comment that documents a test.
    pub ignore_test_docs: bool,
    /// Checked by sentence structure, where the tagger is available.
    pub parse: bool,
    /// Its examples' file type, if the rule sets one.
    #[allow(dead_code, reason = "the example tests read it")]
    pub examples: Option<String>,
    pub dir: PathBuf,
}

impl Rule {
    /// The first paragraph of rule.md: how to fix a hit.
    pub fn guidance(&self) -> String {
        let text = std::fs::read_to_string(self.dir.join("rule.md")).unwrap_or_default();
        let first = text
            .split("\n\n")
            .map(str::trim)
            .find(|p| !p.is_empty() && !p.starts_with('#'));
        first
            .map(|p| p.split_whitespace().collect::<Vec<_>>().join(" "))
            .unwrap_or_default()
    }

    /// The first `bad => good` pair in fixes.txt.
    pub fn example(&self) -> Option<(String, String)> {
        let text = std::fs::read_to_string(self.dir.join("fixes.txt")).ok()?;
        let block = text.split("\n---").next()?;
        let (bad, good) = block.split_once("\n=>")?;
        Some((bad.trim().to_string(), good.trim().to_string()))
    }
}

/// Every rule under `root`, by stage and then id.
pub fn load(root: &Path) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let Ok(text) = std::fs::read_to_string(dir.join("rule.yaml")) else {
            continue;
        };
        let y = yaml::parse(&text);
        let text_of = |k: &str| {
            y.get(k)
                .and_then(yaml::Yaml::str)
                .unwrap_or_default()
                .to_string()
        };
        let list = |k: &str| y.get(k).map(|v| v.list().to_vec()).unwrap_or_default();
        let jev = |k: &str| {
            y.get("jev")
                .and_then(|j| j.get(k))
                .and_then(yaml::Yaml::str)
                == Some("true")
        };
        rules.push(Rule {
            id: entry.file_name().to_string_lossy().into_owned(),
            name: text_of("name"),
            description: text_of("description"),
            stage: text_of("stage").parse().unwrap_or(99),
            detectors: list("detectors"),
            prose_detectors: list("prose_detectors"),
            stampable: text_of("stampable") != "false",
            ignore_test_docs: jev("ignore_test_docs"),
            parse: text_of("parse") == "true",
            examples: Some(text_of("examples")).filter(|e| !e.is_empty()),
            dir,
        });
    }
    rules.sort_by(|a, b| (a.stage, &a.id).cmp(&(b.stage, &b.id)));
    Ok(rules)
}
