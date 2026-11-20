//! Re-exports of other crates that no other crate uses.

use std::collections::BTreeSet;

use syn::{Item, UseTree, Visibility};

use super::File;
use crate::Violation;
use crate::audit::unused::{is_target, leaves};
use crate::audit::workspace::Pkg;
use crate::rust::{attrs_of, is_test};

/// The first segment of a `use` tree.
fn root(tree: &UseTree) -> Option<String> {
    match tree {
        UseTree::Path(p) => Some(p.ident.to_string()),
        _ => None,
    }
}

/// A `pub use` name of another crate, and its module.
struct Export {
    name: String,
    line: usize,
    module: Vec<String>,
}

/// The module path of a file in a crate.
fn module_of(rel: &str) -> Vec<String> {
    let path = rel.strip_prefix("src/").unwrap_or(rel);
    let path = path.strip_suffix(".rs").unwrap_or(path);
    let mut segs: Vec<String> = path.split('/').map(String::from).collect();
    if segs.last().is_some_and(|s| s == "mod" || s == "lib") {
        segs.pop();
    }
    segs
}

/// The `pub use` lines of `crates`, inside module `at`.
fn reexports(items: &[Item], crates: &BTreeSet<String>, at: &[String], out: &mut Vec<Export>) {
    for item in items {
        if attrs_of(item).iter().any(is_test) {
            continue;
        }
        match item {
            Item::Use(u) if matches!(u.vis, Visibility::Public(_)) => {
                if root(&u.tree).is_some_and(|r| crates.contains(&r)) {
                    let mut found = Vec::new();
                    leaves(&u.tree, &mut found);
                    for i in found {
                        out.push(Export {
                            name: i.to_string(),
                            line: i.span().start().line,
                            module: at.to_vec(),
                        });
                    }
                }
            }
            Item::Mod(m) if m.ident != "testing" => {
                let inner = m.content.iter().flat_map(|(_, i)| i.iter());
                let mut path = at.to_vec();
                path.push(m.ident.to_string());
                reexports(&inner.cloned().collect::<Vec<_>>(), crates, &path, out);
            }
            _ => {}
        }
    }
}

/// Names reached through `me`, and the modules globs reach.
#[derive(Default)]
struct Reach<'a> {
    names: BTreeSet<&'a str>,
    globs: BTreeSet<&'a [String]>,
}

impl Reach<'_> {
    fn has(&self, e: &Export) -> bool {
        self.names.contains(e.name.as_str()) || self.globs.contains(e.module.as_slice())
    }
}

fn reached<'a>(files: &'a [File], me: &Pkg) -> Reach<'a> {
    let ident = me.name.replace('-', "_");
    let mut found = Reach::default();
    for f in files {
        let outside = f.pkg.name != me.name || is_target(f.rel);
        let via = if outside { ident.as_str() } else { "$crate" };
        for n in f.code.iter().chain(&f.test).filter(|n| n.segs[0] == via) {
            let mut segs = &n.segs[1..];
            if let Some((last, module)) = segs.split_last()
                && last == "*"
            {
                found.globs.insert(module);
                segs = module;
            }
            found.names.extend(segs.iter().map(String::as_str));
        }
    }
    found
}

pub fn check(pkgs: &[Pkg], files: &[File], out: &mut Vec<Violation>) {
    let crates: BTreeSet<String> = pkgs.iter().map(|p| p.name.replace('-', "_")).collect();
    for p in pkgs {
        let used = reached(files, p);
        let mut others = crates.clone();
        others.remove(&p.name.replace('-', "_"));
        for f in files
            .iter()
            .filter(|f| f.pkg.name == p.name && !f.code.is_empty() && !is_target(f.rel))
        {
            let Ok(file) = syn::parse_file(&f.src.text) else {
                continue;
            };
            let mut found = Vec::new();
            reexports(&file.items, &others, &module_of(f.rel), &mut found);
            for Export { name, line, .. } in found.into_iter().filter(|e| !used.has(e)) {
                out.push(Violation {
                    path: f.src.path.clone(),
                    line,
                    rule: "reexport",
                    message: format!(
                        "`{name}` is re-exported but no other crate names it through `{}`: make it private or remove it",
                        p.name
                    ),
                });
            }
        }
    }
}
