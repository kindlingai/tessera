//! Rules about the tests of each crate.

#[cfg(test)]
mod suite_test;

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::{Spacing, TokenStream, TokenTree};
use syn::visit::Visit;

use super::workspace::Pkg;
use crate::files::Source;
use crate::rust::{is_test, is_test_file};
use crate::{LIMITS, Violation};

/// A token, reduced to what path roots need.
#[derive(PartialEq)]
enum Tok {
    Name(String),
    Colons,
    Other,
}

fn flatten(tokens: TokenStream, out: &mut Vec<Tok>) {
    let mut lone_colon = false;
    for t in tokens {
        match t {
            TokenTree::Punct(p) if p.as_char() == ':' => {
                if lone_colon {
                    out.push(Tok::Colons);
                }
                lone_colon = !lone_colon && p.spacing() == Spacing::Joint;
                continue;
            }
            TokenTree::Ident(i) => out.push(Tok::Name(i.to_string())),
            TokenTree::Group(g) => {
                out.push(Tok::Other);
                flatten(g.stream(), out);
                out.push(Tok::Other);
            }
            _ => out.push(Tok::Other),
        }
        lone_colon = false;
    }
}

/// Names that start a path or a `use a as _`.
fn roots(tokens: TokenStream) -> BTreeSet<String> {
    let mut flat = Vec::new();
    flatten(tokens, &mut flat);
    let mut out = BTreeSet::new();
    for (i, t) in flat.iter().enumerate() {
        let Tok::Name(name) = t else {
            continue;
        };
        let next = flat.get(i + 1);
        let after = next == Some(&Tok::Colons) || next == Some(&Tok::Name("as".into()));
        let before = i > 0 && flat[i - 1] == Tok::Colons;
        if after && !before {
            out.insert(name.clone());
        }
    }
    out
}

/// Clock types that tests may not name.
const CLOCKS: [&str; 2] = ["Instant", "SystemTime"];

/// The first clock a file names, with its line.
fn clock(tokens: TokenStream) -> Option<(usize, String)> {
    for t in tokens {
        match t {
            TokenTree::Ident(i) if CLOCKS.iter().any(|c| i == c) => {
                return Some((i.span().start().line, i.to_string()));
            }
            TokenTree::Group(g) => {
                if let Some(found) = clock(g.stream()) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

/// Lines of `#[test]` functions.
#[derive(Default)]
struct Tests(Vec<usize>);

impl<'ast> Visit<'ast> for Tests {
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        if f.attrs
            .iter()
            .any(|a| is_test(a) && !a.path().is_ident("cfg"))
        {
            self.0.push(f.sig.ident.span().start().line);
        }
        syn::visit::visit_item_fn(self, f);
    }
}

/// The package holding `path`, and the path inside it.
fn home<'a>(pkgs: &'a [Pkg], path: &'a str) -> Option<(&'a Pkg, &'a str)> {
    let best = pkgs
        .iter()
        .filter(|p| path.starts_with(&p.dir))
        .max_by_key(|p| p.dir.len())?;
    Some((best, &path[best.dir.len()..]))
}

/// A `tests/` file alone, or a folder of them.
fn group(pkg: &Pkg, rel: &str) -> String {
    let below = rel.strip_prefix("tests/").unwrap_or(rel);
    match below.split_once('/') {
        Some((dir, _)) if rel.starts_with("tests/") => format!("{}tests/{dir}/", pkg.dir),
        _ => format!("{}{rel}", pkg.dir),
    }
}

/// A Rust file: its names, tests and clocks.
struct Entry<'a> {
    src: &'a Source,
    pkg: &'a Pkg,
    group: String,
    /// In `tests/`, or a `*_test.rs` file of `src/`.
    suite: bool,
    inside: bool,
    used: BTreeSet<String>,
    tests: Vec<usize>,
    timed: Option<(usize, String)>,
}

fn entries<'a>(pkgs: &'a [Pkg], sources: &'a [Source]) -> Vec<Entry<'a>> {
    let mut found = Vec::new();
    for src in sources.iter().filter(|s| s.path.ends_with(".rs")) {
        let Some((pkg, rel)) = home(pkgs, &src.path) else {
            continue;
        };
        let inside = rel.starts_with("src/") && is_test_file(rel);
        let suite = inside || rel.starts_with("tests/");
        let (Ok(tokens), Ok(file)) = (src.text.parse::<TokenStream>(), syn::parse_file(&src.text))
        else {
            continue;
        };
        let mut tests = Tests::default();
        tests.visit_file(&file);
        found.push(Entry {
            src,
            pkg,
            group: group(pkg, rel),
            suite,
            inside,
            used: if suite {
                roots(tokens.clone())
            } else {
                BTreeSet::new()
            },
            tests: tests.0,
            timed: if suite { clock(tokens) } else { None },
        });
    }
    found
}

/// The names each group of test files uses.
fn group_names<'a>(all: &'a [Entry]) -> BTreeMap<&'a str, BTreeSet<&'a str>> {
    let mut groups: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for e in all {
        let names = groups.entry(&e.group).or_default();
        names.extend(e.used.iter().map(String::as_str));
    }
    groups
}

fn homes(all: &[Entry], pkgs: &[Pkg], out: &mut Vec<Violation>) {
    let crates: BTreeSet<String> = pkgs.iter().map(|p| p.name.replace('-', "_")).collect();
    let groups = group_names(all);
    for e in all.iter().filter(|e| e.suite) {
        let Some(&line) = e.tests.first() else {
            continue;
        };
        let used = if e.inside {
            e.used.iter().map(String::as_str).collect()
        } else {
            groups[e.group.as_str()].clone()
        };
        let own = e.pkg.name.replace('-', "_");
        let local = ["crate", "super", "self"];
        if used.contains(own.as_str()) || (e.inside && local.iter().any(|l| used.contains(l))) {
            continue;
        }
        let named: Vec<&str> = crates
            .iter()
            .filter(|c| used.contains(c.as_str()))
            .map(String::as_str)
            .collect();
        let what = if named.is_empty() {
            "no workspace crate".to_string()
        } else {
            named.join(", ")
        };
        out.push(Violation {
            path: e.src.path.clone(),
            line,
            rule: "test-home",
            message: format!(
                "tests only {what}: move them to the crate they test, not `{}`",
                e.pkg.name
            ),
        });
    }
}

fn timing(all: &[Entry], out: &mut Vec<Violation>) {
    for e in all.iter().filter(|e| e.suite) {
        let exempt = LIMITS.timing_exempt.iter();
        if exempt
            .into_iter()
            .any(|d| e.src.path.starts_with(d.as_str()))
        {
            continue;
        }
        if let Some((line, name)) = &e.timed {
            out.push(Violation {
                path: e.src.path.clone(),
                line: *line,
                rule: "test-timing",
                message: format!(
                    "`{name}` in a test file: time code in a bench or tools/calibrate"
                ),
            });
        }
    }
}

pub fn check(pkgs: &[Pkg], sources: &[Source], out: &mut Vec<Violation>) {
    let all = entries(pkgs, sources);
    homes(&all, pkgs, out);
    timing(&all, out);
}
