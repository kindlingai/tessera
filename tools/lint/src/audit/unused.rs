//! Public items that no other crate names.

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use syn::{Attribute, ImplItem, Item as Syn, Type, UseTree, Visibility};

use crate::Violation;
use crate::files::{Source, owner};
use crate::rust::{is_test, is_test_file};

/// A public item, and the names its declaration mentions.
struct Item {
    path: String,
    line: usize,
    name: String,
    what: &'static str,
    refs: BTreeSet<String>,
}

/// One crate's items and the names its files use.
#[derive(Default)]
struct Crate {
    items: Vec<Item>,
    /// Names in the signatures of a type's methods, by type.
    impls: BTreeMap<String, BTreeSet<String>>,
    /// Names in each `macro_rules!` body.
    defs: BTreeMap<String, BTreeSet<String>>,
    /// Item macro calls, by macro name.
    calls: Vec<(String, BTreeSet<String>)>,
    /// Names used anywhere in the crate.
    all: BTreeSet<String>,
    /// Names used by tests, benches, examples and binaries.
    targets: BTreeSet<String>,
    has_lib: bool,
}

fn idents(tokens: TokenStream, out: &mut BTreeSet<String>) {
    for t in tokens {
        match t {
            TokenTree::Ident(i) => {
                let name = i.to_string();
                out.insert(name.strip_prefix("r#").unwrap_or(&name).to_string());
            }
            TokenTree::Group(g) => idents(g.stream(), out),
            _ => {}
        }
    }
}

fn names(tokens: TokenStream) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    idents(tokens, &mut out);
    out
}

fn is_pub(v: &Visibility) -> bool {
    matches!(v, Visibility::Public(_))
}

/// Test items, and items called by name from outside.
fn skipped(attrs: &[Attribute]) -> bool {
    let called = ["proc_macro", "no_mangle", "export_name"];
    attrs.iter().any(|a| {
        let text = a.to_token_stream().to_string();
        is_test(a) || called.iter().any(|n| text.contains(n))
    })
}

fn exported(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("macro_export"))
}

/// A public item's name, kind and type tokens.
fn declared(item: &Syn) -> Option<(&syn::Ident, &'static str, TokenStream)> {
    let all = item.to_token_stream();
    match item {
        Syn::Fn(f) if is_pub(&f.vis) && !skipped(&f.attrs) => {
            Some((&f.sig.ident, "fn", f.sig.to_token_stream()))
        }
        Syn::Struct(s) if is_pub(&s.vis) && !skipped(&s.attrs) => Some((&s.ident, "struct", all)),
        Syn::Enum(e) if is_pub(&e.vis) && !skipped(&e.attrs) => Some((&e.ident, "enum", all)),
        Syn::Trait(t) if is_pub(&t.vis) && !skipped(&t.attrs) => Some((&t.ident, "trait", all)),
        Syn::Type(t) if is_pub(&t.vis) && !skipped(&t.attrs) => Some((&t.ident, "type", all)),
        Syn::Const(c) if is_pub(&c.vis) && !skipped(&c.attrs) => {
            Some((&c.ident, "const", c.ty.to_token_stream()))
        }
        Syn::Static(s) if is_pub(&s.vis) && !skipped(&s.attrs) => {
            Some((&s.ident, "static", s.ty.to_token_stream()))
        }
        Syn::Macro(m) if exported(&m.attrs) && m.mac.path.is_ident("macro_rules") => {
            let ident = m.ident.as_ref()?;
            Some((ident, "macro", m.mac.tokens.clone()))
        }
        _ => None,
    }
}

/// The names a `pub use` brings out. Globs name nothing.
fn leaves<'a>(tree: &'a UseTree, out: &mut Vec<&'a syn::Ident>) {
    match tree {
        UseTree::Path(p) => leaves(&p.tree, out),
        UseTree::Name(n) if n.ident != "self" => out.push(&n.ident),
        UseTree::Rename(r) => out.push(&r.rename),
        UseTree::Group(g) => g.items.iter().for_each(|t| leaves(t, out)),
        _ => {}
    }
}

fn self_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) => Some(p.path.segments.last()?.ident.to_string()),
        _ => None,
    }
}

impl Crate {
    fn push(&mut self, path: &str, ident: &syn::Ident, what: &'static str, refs: TokenStream) {
        self.items.push(Item {
            path: path.to_string(),
            line: ident.span().start().line,
            name: ident.to_string(),
            what,
            refs: names(refs),
        });
    }

    fn collect(&mut self, path: &str, list: &[Syn]) {
        for item in list {
            if let Some((ident, what, refs)) = declared(item) {
                self.push(path, ident, what, refs);
            }
            match item {
                Syn::Use(u) if is_pub(&u.vis) => {
                    let mut found = Vec::new();
                    leaves(&u.tree, &mut found);
                    for ident in found {
                        self.push(path, ident, "re-export", TokenStream::new());
                    }
                }
                Syn::Mod(m) if !skipped(&m.attrs) => {
                    let inner = m.content.iter().flat_map(|(_, i)| i.iter());
                    self.collect(path, &inner.cloned().collect::<Vec<_>>());
                }
                Syn::Impl(i) => self.methods(i),
                Syn::Macro(m) => self.invoked(&m.mac, m.ident.as_ref()),
                _ => {}
            }
        }
    }

    fn invoked(&mut self, mac: &syn::Macro, defined: Option<&syn::Ident>) {
        let called = names(mac.tokens.clone());
        if let Some(name) = defined {
            self.defs.insert(name.to_string(), called);
        } else if let Some(name) = mac.path.get_ident() {
            self.calls.push((name.to_string(), called));
        }
    }

    /// Each call's names with the body of its macro.
    fn groups(&self) -> Vec<BTreeSet<&str>> {
        let mut groups = Vec::new();
        for (name, called) in &self.calls {
            let body = self.defs.get(name).into_iter().flatten();
            groups.push(called.iter().chain(body).map(String::as_str).collect());
        }
        groups
    }

    fn methods(&mut self, i: &syn::ItemImpl) {
        let Some(name) = self_name(&i.self_ty) else {
            return;
        };
        let bounds = names(i.generics.to_token_stream());
        self.impls.entry(name.clone()).or_default().extend(bounds);
        for item in &i.items {
            if let ImplItem::Fn(f) = item
                && (i.trait_.is_some() || is_pub(&f.vis))
            {
                let sig = names(f.sig.to_token_stream());
                self.impls.entry(name.clone()).or_default().extend(sig);
            }
        }
    }
}

/// Files that build their own targets, and so need `pub`.
fn is_target(rel: &str) -> bool {
    let dirs = ["tests/", "benches/", "examples/", "src/bin/"];
    dirs.iter().any(|d| rel.starts_with(d)) || rel == "src/main.rs"
}

fn read<'a>(dirs: &'a [String], sources: &'a [Source]) -> BTreeMap<&'a str, Crate> {
    let mut crates: BTreeMap<&str, Crate> = BTreeMap::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        let Some(dir) = owner(dirs, &src.path) else {
            continue;
        };
        let Ok(tokens) = src.text.parse::<TokenStream>() else {
            continue;
        };
        let rel = &src.path[dir.len()..];
        let c = crates.entry(dir).or_default();
        c.has_lib |= rel == "src/lib.rs";
        idents(tokens.clone(), &mut c.all);
        if is_target(rel) {
            idents(tokens, &mut c.targets);
        } else if rel.starts_with("src/")
            && !is_test_file(rel)
            && let Ok(file) = syn::parse_file(&src.text)
        {
            c.collect(&src.path, &file.items);
        }
    }
    crates
}

/// Names reached from the items other crates use.
fn reached<'a>(c: &'a Crate, outside: &BTreeSet<&str>) -> BTreeSet<&'a str> {
    let mut by_name: BTreeMap<&str, Vec<&Item>> = BTreeMap::new();
    for item in &c.items {
        by_name.entry(&item.name).or_default().push(item);
    }
    let groups = c.groups();
    let called = groups.iter().flatten().copied();
    let mut todo: Vec<&str> = by_name
        .keys()
        .copied()
        .chain(called)
        .filter(|n| outside.contains(n))
        .collect();
    let mut used = BTreeSet::new();
    while let Some(name) = todo.pop() {
        if !used.insert(name) {
            continue;
        }
        for group in groups.iter().filter(|g| g.contains(name)) {
            todo.extend(group.iter().copied());
        }
        for item in by_name.get(name).into_iter().flatten() {
            todo.extend(item.refs.iter().map(String::as_str));
        }
        todo.extend(c.impls.get(name).into_iter().flatten().map(String::as_str));
    }
    used
}

pub fn check(dirs: &[String], sources: &[Source], out: &mut Vec<Violation>) {
    let crates = read(dirs, sources);
    let mut users: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for (dir, c) in &crates {
        for name in &c.all {
            users.entry(name).or_default().insert(dir);
        }
    }
    for (dir, c) in &crates {
        if !c.has_lib {
            continue;
        }
        let mut outside = BTreeSet::new();
        for (name, who) in &users {
            if who.iter().any(|d| d != dir) || c.targets.contains(*name) {
                outside.insert(*name);
            }
        }
        let used = reached(c, &outside);
        for item in c.items.iter().filter(|i| !used.contains(i.name.as_str())) {
            out.push(Violation {
                path: item.path.clone(),
                line: item.line,
                rule: "unused-pub",
                message: format!(
                    "{} `{}` is public but no other crate names it: make it private or remove it",
                    item.what, item.name
                ),
            });
        }
    }
}
