//! Workspace files, crates, modules and the items they name.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use syn::spanned::Spanned;

use crate::calls::{self, Scan};

pub struct Source {
    pub text: String,
    scan: OnceCell<Scan>,
    tree: OnceCell<Option<syn::File>>,
}

pub struct Krate {
    pub ident: String,
    pub dir: String,
    pub root: Option<String>,
    pub bin: bool,
    mods: OnceCell<Vec<Module>>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    File,
    Mod,
    Item,
    Trait,
    Macro,
    Derive,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub name: String,
    pub kind: Kind,
    pub file: String,
    pub line: usize,
    pub end: usize,
}

pub struct Module {
    pub path: Vec<String>,
    pub file: String,
    items: Vec<Item>,
    uses: Vec<(String, Vec<String>)>,
    globs: Vec<Vec<String>>,
    kids: BTreeMap<String, usize>,
}

pub struct Repo {
    pub files: Vec<String>,
    sources: BTreeMap<String, Source>,
    pub crates: BTreeMap<String, Krate>,
}

/// Tracked and untracked, unignored paths.
pub fn listed(root: &Path) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("git ls-files: {e}"))?;
    if !out.status.success() {
        return Err("git ls-files failed".into());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let paths = text.split('\0').filter(|p| !p.is_empty());
    Ok(paths.map(String::from).collect())
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

/// Where a module file's child modules live.
fn kid_dir(file: &str) -> String {
    let dir = parent(file);
    let stem = file
        .rsplit('/')
        .next()
        .unwrap_or(file)
        .trim_end_matches(".rs");
    let last = dir.rsplit('/').next().unwrap_or("");
    let crate_root = ["tests", "bin", "examples", "benches"].contains(&last);
    match ["lib", "main", "mod", "build"].contains(&stem) || crate_root {
        true => format!("{dir}/"),
        false => format!("{dir}/{stem}/"),
    }
}

fn path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|a| match &a.meta {
        syn::Meta::NameValue(nv) if nv.path.is_ident("path") => match &nv.value {
            syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => Some(s.value()),
            _ => None,
        },
        _ => None,
    })
}

fn kind_of(item: &syn::Item) -> Option<(&syn::Ident, Kind)> {
    Some(match item {
        syn::Item::Fn(f) => (&f.sig.ident, Kind::Item),
        syn::Item::Struct(s) => (&s.ident, Kind::Item),
        syn::Item::Enum(e) => (&e.ident, Kind::Item),
        syn::Item::Union(u) => (&u.ident, Kind::Item),
        syn::Item::Type(t) => (&t.ident, Kind::Item),
        syn::Item::Const(c) => (&c.ident, Kind::Item),
        syn::Item::Static(s) => (&s.ident, Kind::Item),
        syn::Item::Trait(t) => (&t.ident, Kind::Trait),
        _ => return None,
    })
}

/// A proc macro's name and kind, from its attribute.
fn proc_macro(f: &syn::ItemFn) -> Option<(String, Kind)> {
    for a in &f.attrs {
        let name = a.path().get_ident()?.to_string();
        match name.as_str() {
            "proc_macro_derive" => {
                let id: syn::Ident = a
                    .parse_args_with(|p: syn::parse::ParseStream| {
                        let id = p.parse()?;
                        let _: proc_macro2::TokenStream = p.parse()?;
                        Ok(id)
                    })
                    .ok()?;
                return Some((id.to_string(), Kind::Derive));
            }
            "proc_macro" => return Some((f.sig.ident.to_string(), Kind::Macro)),
            "proc_macro_attribute" => return Some((f.sig.ident.to_string(), Kind::Derive)),
            _ => {}
        }
    }
    None
}

/// `use` trees as (local name, written path). `*` names a glob.
pub fn flatten(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<(String, Vec<String>)>,
) {
    let mut push = |local: String, last: Option<String>| {
        let mut p = prefix.clone();
        p.extend(last);
        out.push((local, p));
    };
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            flatten(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) if n.ident == "self" => {
            push(prefix.last().cloned().unwrap_or_default(), None)
        }
        syn::UseTree::Name(n) => push(n.ident.to_string(), Some(n.ident.to_string())),
        syn::UseTree::Rename(r) => push(r.rename.to_string(), Some(r.ident.to_string())),
        syn::UseTree::Glob(_) => push("*".into(), None),
        syn::UseTree::Group(g) => {
            for t in &g.items {
                flatten(t, prefix, out);
            }
        }
    }
}

pub fn item(name: String, kind: Kind, file: &str, span: proc_macro2::Span) -> Item {
    let (line, end) = (span.start().line, span.end().line);
    Item {
        name,
        kind,
        file: file.into(),
        line,
        end,
    }
}

/// `macro_rules!` names, and statics declared inside macro calls.
fn macro_items(mods: &mut [Module], at: usize, m: &syn::ItemMacro, file: &str) {
    let span = m.span();
    if let Some(id) = &m.ident {
        let mut found = item(id.to_string(), Kind::Macro, file, span);
        found.line = id.span().start().line;
        let export = m.attrs.iter().any(|a| a.path().is_ident("macro_export"));
        if export && at != 0 {
            mods[0].items.push(found.clone());
        }
        mods[at].items.push(found);
        return;
    }
    if let Some(name) = calls::static_name(&m.mac.tokens).filter(|n| !n.starts_with('$')) {
        mods[at].items.push(item(name, Kind::Item, file, span));
    }
}

impl Repo {
    pub fn open(root: &Path) -> Result<Repo, String> {
        let mut files = listed(root)?;
        files.retain(|p| root.join(p).exists());
        let mut sources = BTreeMap::new();
        let mut crates = BTreeMap::new();
        for path in &files {
            if path.ends_with(".rs") {
                let text = std::fs::read_to_string(root.join(path)).unwrap_or_default();
                let (scan, tree) = (OnceCell::new(), OnceCell::new());
                sources.insert(path.clone(), Source { text, scan, tree });
            }
            if let Some(k) = path.strip_suffix("Cargo.toml") {
                let text = std::fs::read_to_string(root.join(path)).unwrap_or_default();
                let Ok(manifest) = text.parse::<toml::Table>() else {
                    continue;
                };
                let Some(name) = manifest.get("package").and_then(|p| p.get("name")) else {
                    continue;
                };
                let lib = manifest.get("lib").and_then(|l| l.get("path"));
                let lib = lib.and_then(|p| p.as_str()).map(|p| format!("{k}{p}"));
                let found = |f: &str| files.iter().any(|x| x == f);
                let main = format!("{k}src/main.rs");
                let root_file = lib
                    .or_else(|| Some(format!("{k}src/lib.rs")).filter(|f| found(f)))
                    .or_else(|| Some(main.clone()).filter(|f| found(f)));
                let bin = found(&main)
                    || manifest.contains_key("bin")
                    || files.iter().any(|f| f.starts_with(&format!("{k}src/bin/")));
                let ident = name.as_str().unwrap_or_default().replace('-', "_");
                let mods = OnceCell::new();
                let dir = k.to_string();
                crates.insert(
                    ident.clone(),
                    Krate {
                        ident,
                        dir,
                        root: root_file,
                        bin,
                        mods,
                    },
                );
            }
        }
        Ok(Repo {
            files,
            sources,
            crates,
        })
    }

    pub fn text(&self, file: &str) -> Option<&str> {
        self.sources.get(file).map(|s| s.text.as_str())
    }

    /// Rust files whose text holds `needle`.
    pub fn holding<'a>(&'a self, needle: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.sources
            .iter()
            .filter(move |(_, s)| s.text.contains(needle))
            .map(|(p, _)| p.as_str())
    }

    pub fn scan(&self, file: &str) -> Option<&Scan> {
        let s = self.sources.get(file)?;
        Some(s.scan.get_or_init(|| calls::scan(&s.text)))
    }

    pub fn tree(&self, file: &str) -> Option<&syn::File> {
        let s = self.sources.get(file)?;
        s.tree
            .get_or_init(|| syn::parse_file(&s.text).ok())
            .as_ref()
    }

    /// The deepest crate holding `file`.
    pub fn owner(&self, file: &str) -> Option<&Krate> {
        self.crates
            .values()
            .filter(|k| file.starts_with(&k.dir))
            .max_by_key(|k| k.dir.len())
    }

    pub fn package(&self, name: &str) -> Option<&Krate> {
        self.crates.get(&name.replace('-', "_"))
    }

    /// The file a `mod name;` in `file` loads.
    pub fn kid_file(&self, file: &str, name: &str, attrs: &[syn::Attribute]) -> Option<String> {
        let tries = match path_attr(attrs) {
            Some(p) => vec![format!("{}/{p}", parent(file))],
            None => {
                let dir = kid_dir(file);
                vec![format!("{dir}{name}.rs"), format!("{dir}{name}/mod.rs")]
            }
        };
        tries.into_iter().find(|f| self.sources.contains_key(f))
    }

    pub fn modules<'a>(&'a self, k: &'a Krate) -> &'a [Module] {
        k.mods.get_or_init(|| {
            let mut mods = Vec::new();
            if let Some(root) = &k.root
                && let Some(tree) = self.tree(root)
            {
                mods.push(Module::new(Vec::new(), root));
                self.fill(k, &mut mods, 0, &tree.items, root);
            }
            mods
        })
    }

    fn fill(&self, k: &Krate, mods: &mut Vec<Module>, at: usize, items: &[syn::Item], file: &str) {
        let here = mods[at].path.clone();
        let kids: Vec<String> = items
            .iter()
            .filter_map(|i| match i {
                syn::Item::Mod(m) => Some(m.ident.to_string()),
                _ => None,
            })
            .collect();
        for it in items {
            match it {
                syn::Item::Mod(m) => {
                    let name = m.ident.to_string();
                    let mut path = here.clone();
                    path.push(name.clone());
                    let (kid_file, inner) = match &m.content {
                        Some((_, inner)) => (file.to_string(), inner.clone()),
                        None => {
                            let Some(f) = self.kid_file(file, &name, &m.attrs) else {
                                continue;
                            };
                            let Some(tree) = self.tree(&f) else {
                                continue;
                            };
                            (f, tree.items.clone())
                        }
                    };
                    let kid = mods.len();
                    mods[at].kids.insert(name.clone(), kid);
                    mods[at]
                        .items
                        .push(item(name, Kind::Mod, &kid_file, m.ident.span()));
                    mods.push(Module::new(path, &kid_file));
                    self.fill(k, mods, kid, &inner, &kid_file);
                }
                syn::Item::Use(u) if !matches!(u.vis, syn::Visibility::Inherited) => {
                    let mut leaves = Vec::new();
                    flatten(&u.tree, &mut Vec::new(), &mut leaves);
                    for (local, raw) in leaves {
                        let abs = self.absolute(&k.ident, &here, &kids, raw);
                        match local.as_str() {
                            "*" => mods[at].globs.push(abs),
                            _ => mods[at].uses.push((local, abs)),
                        }
                    }
                }
                syn::Item::Macro(m) => macro_items(mods, at, m, file),
                other => {
                    if let syn::Item::Fn(f) = other
                        && let Some((name, kind)) = proc_macro(f)
                    {
                        mods[at]
                            .items
                            .push(item(name, kind, file, f.sig.ident.span()));
                    }
                    if let Some((id, kind)) = kind_of(other) {
                        let mut found = item(id.to_string(), kind, file, other.span());
                        found.line = id.span().start().line;
                        mods[at].items.push(found);
                    }
                }
            }
        }
    }

    /// A written `use` path made absolute, crate ident first.
    fn absolute(
        &self,
        ident: &str,
        here: &[String],
        kids: &[String],
        raw: Vec<String>,
    ) -> Vec<String> {
        let mut base: Vec<String> = std::iter::once(ident.to_string())
            .chain(here.iter().cloned())
            .collect();
        let mut rest = raw.as_slice();
        match rest.first().map(String::as_str) {
            Some("crate") => {
                base.truncate(1);
                rest = &rest[1..];
            }
            Some("self") => rest = &rest[1..],
            Some(first) if self.crates.contains_key(first) && !kids.iter().any(|k| k == first) => {
                return raw;
            }
            _ => {}
        }
        while rest.first().is_some_and(|s| s == "super") {
            if base.len() > 1 {
                base.pop();
            }
            rest = &rest[1..];
        }
        base.extend(rest.iter().cloned());
        base
    }

    /// Every item a symbol path names: `krate::module::name`, `!` for macros.
    pub fn resolve(&self, path: &str) -> Vec<Item> {
        let (body, mac) = match path.strip_suffix('!') {
            Some(b) => (b, true),
            None => (path, false),
        };
        let segs: Vec<String> = body.split("::").map(String::from).collect();
        let mut out = Vec::new();
        self.lookup(&segs, mac, 0, &mut out);
        out
    }

    fn lookup(&self, segs: &[String], mac: bool, depth: usize, out: &mut Vec<Item>) {
        let Some(k) = self.crates.get(&segs[0]) else {
            return;
        };
        let mods = self.modules(k);
        if depth > 24 || mods.is_empty() || segs.len() < 2 {
            return;
        }
        let mut at = 0;
        for (i, seg) in segs[1..].iter().enumerate() {
            let m = &mods[at];
            let last = i + 2 == segs.len();
            if last {
                let fits = |it: &&Item| match mac {
                    true => it.kind == Kind::Macro,
                    false => it.kind != Kind::Macro,
                };
                out.extend(
                    m.items
                        .iter()
                        .filter(|it| it.name == *seg)
                        .filter(fits)
                        .cloned(),
                );
            } else if let Some(&kid) = m.kids.get(seg) {
                at = kid;
                continue;
            }
            let rest = &segs[i + 2..];
            let mut targets: Vec<Vec<String>> = m
                .uses
                .iter()
                .filter(|(local, _)| local == seg)
                .map(|(_, t)| t.clone())
                .collect();
            for g in &m.globs {
                let mut p = g.clone();
                p.push(seg.clone());
                targets.push(p);
            }
            for mut t in targets {
                t.extend(rest.iter().cloned());
                if t != segs {
                    self.lookup(&t, mac, depth + 1, out);
                }
            }
            return;
        }
    }

    /// The module path of `file`, crate ident first.
    pub fn module_of(&self, file: &str) -> Option<Vec<String>> {
        let k = self.owner(file)?;
        let m = self.modules(k).iter().find(|m| m.file == file)?;
        Some(
            std::iter::once(k.ident.clone())
                .chain(m.path.iter().cloned())
                .collect(),
        )
    }

    /// Every `use` in a file, at any visibility, as written.
    pub fn uses_in(&self, file: &str) -> Vec<(String, Vec<String>)> {
        let mut out = Vec::new();
        let Some(tree) = self.tree(file) else {
            return out;
        };
        let mut stack: Vec<&syn::Item> = tree.items.iter().collect();
        while let Some(it) = stack.pop() {
            match it {
                syn::Item::Use(u) => flatten(&u.tree, &mut Vec::new(), &mut out),
                syn::Item::Mod(m) => stack.extend(m.content.iter().flat_map(|c| c.1.iter())),
                syn::Item::Fn(f) => {
                    for s in &f.block.stmts {
                        if let syn::Stmt::Item(i) = s {
                            stack.push(i);
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// Whether the crate has tests: a tests directory or `#[test]`.
    pub fn has_tests(&self, k: &Krate) -> bool {
        self.sources.iter().any(|(p, s)| {
            p.starts_with(&k.dir)
                && self.owner(p).is_some_and(|o| o.dir == k.dir)
                && (p.starts_with(&format!("{}tests/", k.dir)) || s.text.contains("#[test]"))
        })
    }
}

impl Module {
    fn new(path: Vec<String>, file: &str) -> Module {
        Module {
            path,
            file: file.into(),
            items: Vec::new(),
            uses: Vec::new(),
            globs: Vec::new(),
            kids: BTreeMap::new(),
        }
    }
}
