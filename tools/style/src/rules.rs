//! Rules from `rules/<id>/`: rule.yaml, rule.md and fixes.txt.

use std::path::{Path, PathBuf};

pub struct Rule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub dir: PathBuf,
}

/// The single-quoted value of a top-level `key: '...'` line.
fn field(yaml: &str, key: &str) -> String {
    let prefix = format!("{key}: '");
    let line = yaml
        .lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_default();
    line.strip_suffix('\'').unwrap_or(line).replace("''", "'")
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

/// Every rule under `root`, by id.
pub fn load(root: &Path) -> Result<Vec<Rule>, String> {
    let mut rules = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let Ok(yaml) = std::fs::read_to_string(dir.join("rule.yaml")) else {
            continue;
        };
        rules.push(Rule {
            id: entry.file_name().to_string_lossy().into_owned(),
            name: field(&yaml, "name"),
            description: field(&yaml, "description"),
            dir,
        });
    }
    rules.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rules)
}
