//! README.md tables of the crates below each folder.

use std::collections::BTreeMap;
use std::path::Path;

use crate::comments::{SEMICOLON, prose_semicolon, words};
use crate::{LIMITS, Violation};

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

/// `&`, `<` and `>` as HTML entities.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The table's opening and header row.
const HEAD: &str = "<table>\n<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>\n";

/// One indent step per folder level.
fn indent(depth: usize) -> String {
    "&emsp;".repeat(depth)
}

/// `folder`'s crates as a folder tree, then name and description.
pub fn table(folder: &str, crates: &[&Crate]) -> String {
    let mut sorted: Vec<(Vec<&str>, &Crate)> = Vec::new();
    for c in crates {
        let rel = c.dir[folder.len()..].trim_end_matches('/');
        sorted.push((rel.split('/').collect(), c));
    }
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = String::from(HEAD);
    let mut open: Vec<&str> = Vec::new();
    for (parts, c) in &sorted {
        let (dirs, _) = parts.split_at(parts.len() - 1);
        let shared = open.iter().zip(dirs).take_while(|(a, b)| a == b).count();
        open.truncate(shared);
        for (depth, name) in dirs.iter().enumerate().skip(shared) {
            open.push(name);
            let link = open.join("/");
            let pad = indent(depth);
            out.push_str(&format!(
                "<tr><td colspan=\"3\">{pad}<a href=\"{link}\"><b>{name}/</b></a></td></tr>\n"
            ));
        }
        let (link, pad, desc) = (parts.join("/"), indent(dirs.len()), escape(&c.description));
        let (dir, name) = (parts[parts.len() - 1], escape(&c.name));
        out.push_str(&format!(
            "<tr><td>{pad}<a href=\"{link}\">{dir}/</a></td><td>{name}</td><td>{desc}</td></tr>\n"
        ));
    }
    out.push_str("</table>\n");
    out
}

/// The table's length after the marker: HTML or Markdown.
fn table_len(text: &str) -> usize {
    let mut lines = text.split_inclusive('\n');
    if !text.starts_with("<table") {
        return lines.take_while(|l| l.contains('|')).map(str::len).sum();
    }
    let mut len = 0;
    for l in lines.by_ref() {
        len += l.len();
        if l.trim_end() == "</table>" {
            break;
        }
    }
    len
}

/// `text` with the table after its marker replaced.
pub fn splice(text: &str, table: &str) -> Option<String> {
    let at = text.find(MARKER)?;
    let body = at + MARKER.len();
    let body = body + text[body..].find('\n').map_or(text.len() - body, |n| n + 1);
    let old = table_len(&text[body..]);
    Some(format!("{}{table}{}", &text[..body], &text[body + old..]))
}

fn crates(root: &Path, dirs: &[String]) -> Result<Vec<Crate>, String> {
    dirs.iter().map(|d| read_crate(root, d)).collect()
}

pub fn check(root: &Path, dirs: &[String], out: &mut Vec<Violation>) -> Result<(), String> {
    let all = crates(root, dirs)?;
    for c in &all {
        described(root, c, out);
    }
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

/// A description follows the comment rules, with a longer limit.
fn described(root: &Path, c: &Crate, out: &mut Vec<Violation>) {
    let path = format!("{}Cargo.toml", c.dir);
    let text = std::fs::read_to_string(root.join(&path)).unwrap_or_default();
    let at = text.lines().position(|l| l.starts_with("description"));
    let line = at.map_or(1, |i| i + 1);
    let mut push = |rule, message: String| {
        out.push(Violation {
            path: path.clone(),
            line,
            rule,
            message,
        })
    };
    let (n, max) = (words(&c.description), LIMITS.max_description_words);
    if n > max {
        push(
            "description-words",
            format!("description of {n} words (max {max})"),
        );
    }
    if prose_semicolon(&c.description) {
        push("comment-semicolon", SEMICOLON.into());
    }
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
