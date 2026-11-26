//! Joins the imported names of edited `use` lines.

use std::collections::{BTreeMap, BTreeSet};

/// Attribute lines, visibility and the path before the name.
type Key = (String, String, String);
type Names = BTreeMap<Key, BTreeSet<String>>;

/// One side's `use` statements, or its group items.
pub struct Side<'s, 'a> {
    pub entries: &'s [&'s [&'a str]],
    pub items: bool,
}

/// The joined lines when both sides keep each base name.
pub fn join(ours: &Side, base: &Side, theirs: &Side) -> Option<String> {
    let first = ours.entries.first().or(theirs.entries.first())?.first()?;
    let indent = &first[..first.len() - first.trim_start().len()];
    let (mut all, base, theirs) = (names(ours)?, names(base)?, names(theirs)?);
    for (key, kept) in &base {
        let held = |side: &Names| side.get(key).is_some_and(|names| names.is_superset(kept));
        if !held(&all) || !held(&theirs) {
            return None;
        }
    }
    for (key, added) in theirs {
        all.entry(key).or_default().extend(added);
    }
    let mut out = String::new();
    for ((attrs, vis, path), mut names) in all {
        let mut uses = Vec::new();
        if ours.items {
            uses.push(format!("{},", join_names(&names)));
        } else if path.is_empty() {
            uses.extend(names.iter().map(|name| format!("{vis}use {name};")));
        } else {
            if names.remove("*") {
                uses.push(format!("{vis}use {path}::*;"));
            }
            let one = names
                .first()
                .filter(|name| names.len() == 1 && *name != "self");
            match one {
                Some(name) => uses.push(format!("{vis}use {path}::{name};")),
                None if names.is_empty() => {}
                None => uses.push(format!("{vis}use {path}::{{{}}};", join_names(&names))),
            }
        }
        for line in uses {
            out.push_str(&attrs);
            out.push_str(indent);
            out.push_str(&line);
            out.push('\n');
        }
    }
    Some(out)
}

fn join_names(names: &BTreeSet<String>) -> String {
    names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Each name a side imports, by statement shape.
fn names(side: &Side) -> Option<Names> {
    let mut out = Names::new();
    for entry in side.entries {
        let start = entry
            .iter()
            .position(|line| !crate::lists::prefix(line.trim(), false))?;
        let attrs = entry[..start].concat();
        if entry[start..].iter().any(|line| line.contains("//")) {
            return None;
        }
        let text = entry[start..]
            .iter()
            .flat_map(|line| line.split_whitespace());
        let text = text.collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            return None;
        }
        if side.items {
            if !attrs.is_empty() {
                return None;
            }
            for name in split_top(text.strip_suffix(',')?) {
                let key = Key::default();
                out.entry(key).or_default().insert(name.to_string());
            }
            continue;
        }
        let rest = crate::lists::after_vis(&text);
        let vis = text[..text.len() - rest.len()].to_string();
        let mut leaves = Vec::new();
        leaves_of(
            rest.strip_prefix("use ")?.strip_suffix(';')?,
            "",
            &mut leaves,
        )?;
        for leaf in leaves {
            let path = leaf.split(" as ").next()?;
            let (path, name) = match path.rfind("::") {
                Some(at) => (&leaf[..at], &leaf[at + 2..]),
                None => ("", leaf.as_str()),
            };
            let key = (attrs.clone(), vis.clone(), path.to_string());
            out.entry(key).or_default().insert(name.to_string());
        }
    }
    Some(out)
}

/// The parts of a comma list, outside nested braces.
fn split_top(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (at, c) in text.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.push(text[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    out.push(text[start..].trim());
    out.retain(|part| !part.is_empty());
    out
}

/// Adds each full path of a `use` tree to `out`.
fn leaves_of(tree: &str, head: &str, out: &mut Vec<String>) -> Option<()> {
    let tree = tree.trim();
    let Some(open) = tree.find('{') else {
        out.push(format!("{head}{tree}"));
        return Some(());
    };
    let inner = tree[open + 1..].strip_suffix('}')?;
    let head = format!("{head}{}", &tree[..open]);
    if !(head.is_empty() || head.ends_with("::")) {
        return None;
    }
    for part in split_top(inner) {
        leaves_of(part, &head, out)?;
    }
    Some(())
}
