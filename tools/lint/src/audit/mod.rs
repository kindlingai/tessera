//! Rules that read the workspace's crates and what they name.

#[cfg(test)]
mod audit_test;
mod deps;
mod layers;
#[cfg(test)]
mod layers_test;
mod repeats;
#[cfg(test)]
mod repeats_test;
mod suite;
mod unused;
#[cfg(test)]
mod unused_test;
mod workspace;

use std::collections::BTreeMap;
use std::path::Path;

use sensai_findings::Level;
use syn::Item;

use crate::files::{Source, owner};
use crate::rust::{is_test, is_test_file};
use crate::{LIMITS, Violation};
use unused::exported;

pub use deps::Thin;
pub use layers::{Allow, Facade};
pub use repeats::Exprs;

/// Rules in `warn_rules` only advise.
pub fn level(rule: &str, warn: &[String]) -> Level {
    if warn.iter().any(|w| w == rule) {
        Level::Note
    } else {
        Level::Fail
    }
}

/// Crate directories that `tessera-publish` merges, by published crate.
fn merged(root: &Path) -> Result<Vec<Vec<String>>, String> {
    let groups = tessera_publish::merged(root).map_err(|e| format!("publish plan: {e}"))?;
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut out = Vec::new();
    for group in groups {
        let mut dirs = Vec::new();
        for dir in group {
            if let Ok(rel) = dir.strip_prefix(&root) {
                dirs.push(format!("{}/", rel.display()));
            }
        }
        out.push(dirs);
    }
    Ok(out)
}

pub fn check(
    root: &Path,
    dirs: &[String],
    sources: &[Source],
    out: &mut Vec<Violation>,
) -> Result<(), String> {
    let pkgs = workspace::load(root)?;
    layers::check(&pkgs, &LIMITS.layers, &LIMITS.layer_allow, out);
    layers::facade(&pkgs, &LIMITS.facade, out);
    layers::placement(&pkgs, &LIMITS.layers, LIMITS.placement_gap, out);
    deps::check(&pkgs, sources, &LIMITS.layers[0], &LIMITS.thin_edge, out);
    unused::check(dirs, sources, out);
    clashes(dirs, &merged(root)?, sources, out);
    suite::check(&pkgs, sources, out);
    repeats::check(sources, out);
    repeats::exprs(sources, &LIMITS.repeat_exprs, out);
    Ok(())
}

/// Exported `macro_rules!` names in `items`, with their lines.
fn macros(items: &[Item], out: &mut Vec<(String, usize)>) {
    for item in items {
        match item {
            Item::Macro(m) if exported(&m.attrs) && m.mac.path.is_ident("macro_rules") => {
                if let Some(ident) = &m.ident {
                    out.push((ident.to_string(), ident.span().start().line));
                }
            }
            Item::Mod(m) if !m.attrs.iter().any(is_test) => {
                if let Some((_, inner)) = &m.content {
                    macros(inner, out);
                }
            }
            _ => {}
        }
    }
}

/// `groups` holds the crate directories each published crate merges.
pub fn clashes(
    dirs: &[String],
    groups: &[Vec<String>],
    sources: &[Source],
    out: &mut Vec<Violation>,
) {
    for group in groups {
        let mut found: BTreeMap<String, Vec<(&str, &str, usize)>> = BTreeMap::new();
        for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
            let Some(dir) = owner(dirs, &src.path).filter(|d| group.iter().any(|g| g == d)) else {
                continue;
            };
            let rel = &src.path[dir.len()..];
            if !rel.starts_with("src/") || rel.starts_with("src/bin/") || is_test_file(rel) {
                continue;
            }
            let Ok(file) = syn::parse_file(&src.text) else {
                continue;
            };
            let mut names = Vec::new();
            macros(&file.items, &mut names);
            for (name, line) in names {
                found.entry(name).or_default().push((dir, &src.path, line));
            }
        }
        for (name, places) in &found {
            for &(dir, path, line) in places {
                let Some(&(_, other, at)) = places.iter().find(|p| p.0 != dir) else {
                    continue;
                };
                out.push(Violation {
                    path: path.to_string(),
                    line,
                    rule: "macro-clash",
                    message: format!(
                        "`{name}!` is also exported at {other}:{at}, and `tessera-publish` \
                         merges both crates into one: rename one of them"
                    ),
                });
            }
        }
    }
}
