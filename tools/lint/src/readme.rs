//! README.md tables of the crates below each folder.

use std::collections::BTreeMap;
use std::path::Path;

use crate::Violation;

pub const MARKER: &str = "<!-- crate table -->";

/// A crate's directory, with its name and description.
pub struct Crate {
    pub dir: String,
    pub name: String,
    pub description: String,
}

#[derive(serde::Deserialize)]
struct Manifest {
    package: Package,
}

#[derive(serde::Deserialize)]
struct Package {
    name: String,
    description: Option<String>,
}

fn read_crate(root: &Path, dir: &str) -> Result<Crate, String> {
    let path = format!("{dir}Cargo.toml");
    let text = std::fs::read_to_string(root.join(&path)).map_err(|e| format!("{path}: {e}"))?;
    let m: Manifest = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok(Crate {
        dir: dir.to_string(),
        name: m.package.name,
        description: m.package.description.unwrap_or_default(),
    })
}

/// Each folder above a crate, with the crates below it.
pub fn folders(crates: &[Crate]) -> BTreeMap<String, Vec<&Crate>> {
    let mut out: BTreeMap<String, Vec<&Crate>> = BTreeMap::new();
    for c in crates {
        let parts: Vec<&str> = c.dir.trim_end_matches('/').split('/').collect();
        for depth in 0..parts.len() {
            let folder: String = parts[..depth].iter().map(|p| format!("{p}/")).collect();
            out.entry(folder).or_default().push(c);
        }
    }
    for list in out.values_mut() {
        list.sort_by_key(|c| c.dir.trim_end_matches('/').to_string());
    }
    out
}

/// The table for `folder`, with links relative to it.
pub fn table(folder: &str, crates: &[&Crate]) -> String {
    let mut out = String::from("Crate | Description\n--- | ---\n");
    for c in crates {
        let link = c.dir[folder.len()..].trim_end_matches('/');
        let desc = c.description.replace('|', "\\|");
        out.push_str(&format!("[{}]({link}) | {desc}\n", c.name));
    }
    out
}

/// `text` with the table after its marker replaced.
pub fn splice(text: &str, table: &str) -> Option<String> {
    let at = text.find(MARKER)?;
    let body = at + MARKER.len();
    let body = body + text[body..].find('\n').map_or(text.len() - body, |n| n + 1);
    let old: usize = text[body..]
        .split_inclusive('\n')
        .take_while(|l| l.contains('|'))
        .map(str::len)
        .sum();
    Some(format!("{}{table}{}", &text[..body], &text[body + old..]))
}

fn crates(root: &Path, dirs: &[String]) -> Result<Vec<Crate>, String> {
    dirs.iter().map(|d| read_crate(root, d)).collect()
}

pub fn check(root: &Path, dirs: &[String], out: &mut Vec<Violation>) -> Result<(), String> {
    let all = crates(root, dirs)?;
    for (folder, list) in folders(&all) {
        let path = format!("{folder}README.md");
        let found = match std::fs::read_to_string(root.join(&path)) {
            Ok(text) => splice(&text, &table(&folder, &list)).map(|new| new == text),
            Err(_) => None,
        };
        let message = match found {
            Some(true) => continue,
            Some(false) => "crate table is out of date",
            None => "needs a crate table after `<!-- crate table -->`",
        };
        out.push(Violation {
            path,
            line: 1,
            rule: "crate-table",
            message: format!("{message}: run `cargo run -p tessera-lint -- tables`"),
        });
    }
    Ok(())
}

/// Writes every table. Adds missing markers and READMEs.
pub fn write(root: &Path, dirs: &[String]) -> Result<(), String> {
    let all = crates(root, dirs)?;
    for (folder, list) in folders(&all) {
        let path = root.join(&folder).join("README.md");
        let table = table(&folder, &list);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => splice(&text, &table).unwrap_or(format!("{text}\n{MARKER}\n{table}")),
            Err(_) => format!("# {}\n\n{MARKER}\n{table}", folder.trim_end_matches('/')),
        };
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}
