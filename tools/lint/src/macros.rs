//! Rust `macros/` files: only `macro_rules!`, each called twice or more.

use std::collections::BTreeMap;

use proc_macro2::{TokenStream, TokenTree};
use syn::Item;

use crate::files::{Source, macro_dir, owner};
use crate::rust::lines_of;
use crate::{LIMITS, Violation};

/// Counts `name!` invocations in a token stream, by name.
pub fn calls(tokens: TokenStream, counts: &mut BTreeMap<String, usize>) {
    let mut prev: Option<String> = None;
    for t in tokens {
        match t {
            TokenTree::Punct(p) if p.as_char() == '!' => {
                if let Some(name) = prev.take() {
                    *counts.entry(name).or_default() += 1;
                }
            }
            TokenTree::Ident(i) => prev = Some(i.to_string()),
            TokenTree::Group(g) => {
                prev = None;
                calls(g.stream(), counts);
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => prev = None,
        }
    }
}

struct Def {
    path: String,
    line: usize,
    name: String,
    /// Calls inside its own body, which do not count.
    own: usize,
}

/// Definitions in one macros file. Anything else is a violation.
fn definitions(src: &Source, out: &mut Vec<Violation>) -> Vec<Def> {
    let file = match syn::parse_file(&src.text) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut defs = Vec::new();
    for item in &file.items {
        let (line, _) = lines_of(item);
        match item {
            Item::Macro(m) if m.mac.path.is_ident("macro_rules") && m.ident.is_some() => {
                let name = m
                    .ident
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                let mut own = BTreeMap::new();
                calls(m.mac.tokens.clone(), &mut own);
                let own = own.get(&name).copied().unwrap_or(0);
                let path = src.path.clone();
                defs.push(Def {
                    path,
                    line,
                    name,
                    own,
                });
            }
            Item::Mod(m) if m.content.is_none() => {}
            _ => out.push(src.violation(
                line,
                "macro-content",
                "a macros/ file may hold only macro_rules! and mod declarations".into(),
            )),
        }
    }
    defs
}

pub fn check(dirs: &[String], sources: &[Source], out: &mut Vec<Violation>) {
    let mut defs: BTreeMap<&str, Vec<Def>> = BTreeMap::new();
    let mut counts: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        let Some(dir) = owner(dirs, &src.path) else {
            continue;
        };
        if macro_dir(dir, &src.path).is_some() {
            defs.entry(dir).or_default().extend(definitions(src, out));
        }
        if let Ok(tokens) = src.text.parse::<TokenStream>() {
            calls(tokens, counts.entry(dir).or_default());
        }
    }
    for (dir, list) in defs {
        let crate_counts = counts.get(dir);
        for d in list {
            let all = crate_counts
                .and_then(|c| c.get(&d.name))
                .copied()
                .unwrap_or(0);
            let n = all.saturating_sub(d.own);
            let min = LIMITS.min_macro_calls;
            if n < min {
                out.push(Violation {
                    path: d.path,
                    line: d.line,
                    rule: "macro-calls",
                    message: format!("macro `{}` has {n} callsites (min {min})", d.name),
                });
            }
        }
    }
}
