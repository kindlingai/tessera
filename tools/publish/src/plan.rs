//! Which workspace crates go into which published crate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::Result;
use crate::meta::{Meta, Package};

pub(crate) const ENGINE: &str = "tessera-engine";
pub(crate) const FRONTEND: &str = "tessera-frontend";
pub(crate) const CLI: &str = "tessera-cli";
/// The command the CLI crate installs.
pub(crate) const BIN: &str = "tesserax";
/// The npm launcher. Platform packages add `-<os>-<cpu>`.
pub(crate) const NPM: &str = "@kindlingai/tessera";
const CLI_SOURCE: &str = "tessera-cli";

#[derive(PartialEq)]
pub(crate) enum Kind {
    Lib,
    Macro,
    Bin,
    /// [`crate::lint`]: a binary with every crate as a private module.
    Lint,
}

/// The kind of a member's root file.
#[derive(PartialEq)]
pub(crate) enum Role {
    Lib,
    /// A binary or example, whose `main` is a `dead_code` root.
    Main,
    /// An integration test, built only with `cfg(test)`.
    Test,
}

/// One crate to publish.
pub(crate) struct Out {
    pub name: String,
    pub kind: Kind,
    pub description: String,
    pub members: Vec<Member>,
    /// Replacement paths, by crate name.
    pub names: BTreeMap<String, String>,
    /// A member whose items the root re-exports.
    pub glob: Option<String>,
}

/// A workspace crate inside a published crate.
pub(crate) struct Member {
    pub pkg: usize,
    /// Its module, or empty when it is the whole crate.
    pub module: String,
    /// Its directory in the published crate.
    pub to: PathBuf,
    /// Its root file, relative to its package.
    pub root: PathBuf,
    pub role: Role,
}

pub(crate) fn plan(meta: &Meta) -> Result<Vec<Out>> {
    let find = |name: &str| meta.packages.iter().position(|p| p.name == name);
    let engine = find(ENGINE).ok_or("no tessera-engine package")?;
    let cli = find(CLI_SOURCE).ok_or("no tessera-cli package")?;
    let below = closure(meta, engine);
    let macros: Vec<usize> = below
        .iter()
        .copied()
        .filter(|&p| meta.packages[p].is_macro())
        .collect();
    let core: Vec<usize> = below
        .iter()
        .copied()
        .filter(|p| !macros.contains(p))
        .collect();
    let (front_macros, front): (Vec<usize>, Vec<usize>) = (0..meta.packages.len())
        .filter(|&p| under(meta, p, "frontend") && p != cli && meta.packages[p].lib().is_some())
        .partition(|&p| meta.packages[p].is_macro());
    let mut outs = Vec::new();
    for &m in &macros {
        let names = paths(meta, &core, |module| format!("tessera_engine::{module}"));
        outs.push(single(meta, m, Kind::Macro, names));
    }
    for &m in &front_macros {
        let names = paths(meta, &front, |module| format!("tessera_frontend::{module}"));
        outs.push(single(meta, m, Kind::Macro, names));
    }
    outs.push(merged(meta, ENGINE, &core, "engine")?);
    outs.push(merged(meta, FRONTEND, &front, "frontend")?);
    let mut bin = single(
        meta,
        cli,
        Kind::Bin,
        paths(meta, &front, |m| {
            format!("{}::{m}", FRONTEND.replace('-', "_"))
        }),
    );
    bin.name = CLI.into();
    outs.push(bin);
    Ok(outs)
}

/// `start` and the workspace crates it needs to build.
fn closure(meta: &Meta, start: usize) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([start]);
    let mut todo = vec![start];
    while let Some(p) = todo.pop() {
        for d in meta.packages[p].normal() {
            let found = meta.packages.iter().position(|q| q.name == d.name);
            if let Some(q) = found.filter(|q| seen.insert(*q)) {
                todo.push(q);
            }
        }
    }
    seen
}

fn under(meta: &Meta, p: usize, top: &str) -> bool {
    meta.packages[p]
        .dir()
        .starts_with(meta.workspace_root.join(top))
}

pub(crate) fn lib_name(pkg: &Package) -> String {
    pkg.lib()
        .map_or_else(|| pkg.name.replace('-', "_"), |t| t.name.clone())
}

pub(crate) fn module(pkg: &Package) -> String {
    let lib = lib_name(pkg);
    lib.strip_prefix("tessera_").unwrap_or(&lib).to_string()
}

fn paths(meta: &Meta, pkgs: &[usize], to: impl Fn(&str) -> String) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for &p in pkgs {
        let pkg = &meta.packages[p];
        if pkg.name != ENGINE {
            out.insert(lib_name(pkg), to(&module(pkg)));
        }
    }
    out
}

pub(crate) fn single(meta: &Meta, p: usize, kind: Kind, names: BTreeMap<String, String>) -> Out {
    let pkg = &meta.packages[p];
    let member = Member {
        pkg: p,
        module: String::new(),
        to: PathBuf::new(),
        root: PathBuf::new(),
        role: Role::Lib,
    };
    let description = pkg.description.clone().unwrap_or_default();
    Out {
        name: pkg.name.clone(),
        kind,
        description,
        members: vec![member],
        names,
        glob: None,
    }
}

/// Crates under `top/` as modules of one library.
fn merged(meta: &Meta, name: &str, pkgs: &[usize], top: &str) -> Result<Out> {
    let base = meta.workspace_root.join(top);
    let mut members = Vec::new();
    for &p in pkgs {
        let pkg = &meta.packages[p];
        let rel = pkg
            .dir()
            .strip_prefix(&base)
            .map_err(|_| format!("{} is not under {top}/", pkg.name))?;
        let lib = pkg.lib().ok_or(format!("{} has no library", pkg.name))?;
        members.push(Member {
            pkg: p,
            module: module(pkg),
            to: PathBuf::from("src").join(rel),
            root: lib.src_path.strip_prefix(pkg.dir())?.to_path_buf(),
            role: Role::Lib,
        });
    }
    let mut names = paths(meta, pkgs, |m| format!("crate::{m}"));
    let root = pkgs.iter().find(|&&p| meta.packages[p].name == name);
    if let Some(&p) = root {
        names.insert(
            lib_name(&meta.packages[p]),
            format!("crate::{}", module(&meta.packages[p])),
        );
    }
    let description = match root {
        Some(&p) => meta.packages[p].description.clone().unwrap_or_default(),
        None => "Model files, tasks and serving over tessera-engine".into(),
    };
    let glob = root.map(|&p| module(&meta.packages[p]));
    Ok(Out {
        name: name.into(),
        kind: Kind::Lib,
        description,
        members,
        names,
        glob,
    })
}
