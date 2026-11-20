//! Rules that read the `a::b` paths each Rust file names.

#[cfg(test)]
mod deps_test;
mod edges;
mod reexport;

use std::collections::BTreeSet;

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};
use quote::ToTokens;
use syn::Item;

use super::workspace::Pkg;

use crate::Violation;
use crate::files::Source;
use crate::rust::{attrs_of, is_test, is_test_file};
pub use edges::Thin;

/// A path with its segments, `use` groups expanded.
pub struct Named {
    pub segs: Vec<String>,
    pub line: usize,
}

/// A Rust file and the paths it names.
pub struct File<'a> {
    pub pkg: &'a Pkg,
    pub src: &'a Source,
    pub rel: &'a str,
    /// Paths outside tests and `testing` modules.
    pub code: Vec<Named>,
    pub test: Vec<Named>,
}

fn colons(t: &[TokenTree], i: usize) -> bool {
    let punct = |j: usize| match t.get(j) {
        Some(TokenTree::Punct(p)) if p.as_char() == ':' => Some(p.spacing()),
        _ => None,
    };
    punct(i) == Some(Spacing::Joint) && punct(i + 1).is_some()
}

fn is_punct(t: &TokenTree, c: char) -> bool {
    matches!(t, TokenTree::Punct(p) if p.as_char() == c)
}

/// Reads segments from `i` and returns the next index.
fn walk(
    t: &[TokenTree],
    mut i: usize,
    mut segs: Vec<String>,
    line: usize,
    out: &mut Vec<Named>,
) -> usize {
    let start = segs.len();
    loop {
        match t.get(i) {
            Some(TokenTree::Ident(id)) => {
                segs.push(id.to_string());
                i += 1;
                if colons(t, i) {
                    i += 2;
                    continue;
                }
            }
            Some(p) if is_punct(p, '*') => {
                segs.push("*".into());
                i += 1;
            }
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => {
                let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                for part in inner.split(|x| is_punct(x, ',')) {
                    walk(part, 0, segs.clone(), line, out);
                }
                return i + 1;
            }
            _ => {}
        }
        break;
    }
    if segs.len() > start {
        out.push(Named { segs, line });
    }
    i
}

/// `use a as b` gives the path `a::_`.
fn link(t: &[TokenTree], i: usize) -> Option<Named> {
    let word = |j: usize| match t.get(j) {
        Some(TokenTree::Ident(id)) => Some(id),
        _ => None,
    };
    let (kw, name, op) = (word(i)?, word(i + 1)?, word(i + 2)?);
    let alias = matches!(t.get(i + 3), Some(TokenTree::Ident(_)));
    (kw == "use" && op == "as" && alias).then(|| Named {
        segs: vec![name.to_string(), "_".into()],
        line: name.span().start().line,
    })
}

/// Paths in `tokens`. A `$crate` root keeps its `$`.
pub fn scan(tokens: TokenStream, out: &mut Vec<Named>) {
    let t: Vec<TokenTree> = tokens.into_iter().collect();
    let mut i = 0;
    while i < t.len() {
        if let Some(n) = link(&t, i) {
            out.push(n);
            i += 4;
            continue;
        }
        if let TokenTree::Ident(id) = &t[i]
            && colons(&t, i + 1)
            && !(i >= 2 && colons(&t, i - 2))
        {
            let dollar = i > 0 && is_punct(&t[i - 1], '$');
            let root = format!("{}{id}", if dollar { "$" } else { "" });
            i = walk(&t, i + 3, vec![root], id.span().start().line, out);
            continue;
        }
        if let TokenTree::Group(g) = &t[i] {
            scan(g.stream(), out);
        }
        i += 1;
    }
}

/// Splits items into test and other code.
fn split(items: &[Item], code: &mut Vec<Named>, test: &mut Vec<Named>) {
    for item in items {
        let testing = matches!(item, Item::Mod(m) if m.ident == "testing");
        if testing || attrs_of(item).iter().any(is_test) {
            scan(item.to_token_stream(), test);
        } else if let Item::Mod(m) = item
            && let Some((_, inner)) = &m.content
        {
            split(inner, code, test);
        } else {
            scan(item.to_token_stream(), code);
        }
    }
}

/// Tests, benches, examples and `testing` modules.
fn is_test_path(rel: &str) -> bool {
    let dirs = ["tests/", "benches/", "examples/"];
    let testing = rel.contains("/testing/") || rel.ends_with("/testing.rs");
    dirs.iter().any(|d| rel.starts_with(d)) || is_test_file(rel) || testing
}

fn files<'a>(pkgs: &'a [Pkg], sources: &'a [Source]) -> Vec<File<'a>> {
    let mut found = Vec::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        let home = pkgs.iter().filter(|p| src.path.starts_with(&p.dir));
        let Some(pkg) = home.max_by_key(|p| p.dir.len()) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&src.text) else {
            continue;
        };
        let rel = &src.path[pkg.dir.len()..];
        let (mut code, mut test) = (Vec::new(), Vec::new());
        if is_test_path(rel) {
            scan(file.to_token_stream(), &mut test);
        } else {
            split(&file.items, &mut code, &mut test);
        }
        found.push(File {
            pkg,
            src,
            rel,
            code,
            test,
        });
    }
    found
}

/// A test harness: a library without dependents or binaries.
fn harness(pkgs: &[Pkg], files: &[File], pkg: &Pkg) -> bool {
    let used = pkgs
        .iter()
        .flat_map(|p| &p.deps)
        .any(|d| d.name == pkg.name);
    let bin = |f: &File| f.rel == "src/main.rs" || f.rel.starts_with("src/bin/");
    !used && !files.iter().any(|f| f.pkg.name == pkg.name && bin(f))
}

/// Only test code names `testing` modules.
fn testing(pkgs: &[Pkg], files: &[File], out: &mut Vec<Violation>) {
    for f in files.iter().filter(|f| !harness(pkgs, files, f.pkg)) {
        let mut lines = BTreeSet::new();
        for n in f
            .code
            .iter()
            .filter(|n| n.segs.iter().any(|s| s == "testing"))
        {
            if lines.insert(n.line) {
                out.push(Violation {
                    path: f.src.path.clone(),
                    line: n.line,
                    rule: "testing-path",
                    message: format!(
                        "`{}` outside test code: only tests may use `testing` modules",
                        n.segs.join("::")
                    ),
                });
            }
        }
    }
}

pub fn check(
    pkgs: &[Pkg],
    sources: &[Source],
    base: &[String],
    thin: &Thin,
    out: &mut Vec<Violation>,
) {
    let files = files(pkgs, sources);
    edges::check(pkgs, &files, base, thin, out);
    testing(pkgs, &files, out);
    reexport::check(pkgs, &files, out);
}
