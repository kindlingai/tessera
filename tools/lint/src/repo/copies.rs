//! Types with the same members in two crates.

use std::collections::BTreeMap;

use syn::visit::Visit;

use crate::files::{Source, owner};
use crate::{LIMITS, Violation};

/// Fields or variants a type needs before a copy counts.
const MIN_MEMBERS: usize = 3;

/// A struct's named fields or an enum's variants, sorted.
struct Def {
    name: String,
    line: usize,
    members: Vec<String>,
    kind: &'static str,
}

#[derive(Default)]
struct Defs(Vec<Def>);

/// Definitions by kind and members, with their files.
type Groups<'a> = BTreeMap<(&'static str, Vec<String>), Vec<(&'a Source, Def)>>;

impl Defs {
    fn add(&mut self, ident: &syn::Ident, kind: &'static str, mut members: Vec<String>) {
        members.sort();
        if members.len() >= MIN_MEMBERS {
            let (name, line) = (ident.to_string(), ident.span().start().line);
            self.0.push(Def {
                name,
                line,
                members,
                kind,
            });
        }
    }
}

impl Visit<'_> for Defs {
    fn visit_item_struct(&mut self, s: &syn::ItemStruct) {
        if let syn::Fields::Named(f) = &s.fields {
            let names = f.named.iter().filter_map(|f| f.ident.as_ref());
            self.add(&s.ident, "fields", names.map(|i| i.to_string()).collect());
        }
    }

    fn visit_item_enum(&mut self, e: &syn::ItemEnum) {
        let names = e.variants.iter().map(|v| v.ident.to_string());
        self.add(&e.ident, "variants", names.collect());
    }
}

/// `path::Name`, as `lint.toml` lists allowed copies.
fn id(src: &Source, def: &Def) -> String {
    format!("{}::{}", src.path, def.name)
}

/// A group every member of which one allowed set names.
fn allowed(ids: &[String]) -> bool {
    let sets = &LIMITS.allowed_type_copies;
    sets.iter().any(|set| ids.iter().all(|id| set.contains(id)))
}

pub fn check(dirs: &[String], sources: &[Source], out: &mut Vec<Violation>) {
    let mut groups = Groups::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        let Ok(file) = syn::parse_file(&src.text) else {
            continue;
        };
        let mut defs = Defs::default();
        defs.visit_file(&file);
        for def in defs.0 {
            let key = (def.kind, def.members.clone());
            groups.entry(key).or_default().push((src, def));
        }
    }
    for list in groups.values() {
        let mut crates: Vec<Option<&str>> =
            list.iter().map(|(s, _)| owner(dirs, &s.path)).collect();
        crates.sort();
        crates.dedup();
        let ids: Vec<String> = list.iter().map(|(s, d)| id(s, d)).collect();
        if crates.len() < 2 || allowed(&ids) {
            continue;
        }
        for (src, def) in list {
            let others: Vec<&String> = ids.iter().filter(|i| **i != id(src, def)).collect();
            let others = others.iter().map(|i| format!("`{i}`")).collect::<Vec<_>>();
            let message = format!(
                "`{}` has the same {} as {}: share one type",
                def.name,
                def.kind,
                others.join(", ")
            );
            out.push(src.violation(def.line, "type-copy", message));
        }
    }
}
