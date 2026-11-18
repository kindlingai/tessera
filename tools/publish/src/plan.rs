//! Which workspace crates go into which published crate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::Result;
use crate::meta::{Meta, Package};

pub const ENGINE: &str = "sensai-engine";
pub const FRONTEND: &str = "sensai";
pub const CLI: &str = "sensaix";
/// The second name the binary answers to.
pub const CLI_ALIAS: &str = "sensai";
const CLI_SOURCE: &str = "sensai-cli";

#[derive(PartialEq)]
pub enum Kind {
    Lib,
    Macro,
    Bin,
}

/// One crate to publish.
pub struct Out {
    pub name: String,
    pub kind: Kind,
    pub description: String,
    pub members: Vec<Member>,
    /// Crate names and the paths that replace them.
    pub names: BTreeMap<String, String>,
    /// A member whose items the root re-exports.
    pub glob: Option<String>,
}

/// A workspace crate inside a published crate.
pub struct Member {
    pub pkg: usize,
    /// Its module, or empty when it is the whole crate.
    pub module: String,
    /// Its directory in the published crate.
    pub to: PathBuf,
}

pub fn plan(meta: &Meta) -> Result<Vec<Out>> {
    let find = |name: &str| meta.packages.iter().position(|p| p.name == name);
    let engine = find(ENGINE).ok_or("no sensai-engine package")?;
    let cli = find(CLI_SOURCE).ok_or("no sensai-cli package")?;
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
    let front: Vec<usize> = (0..meta.packages.len())
        .filter(|&p| under(meta, p, "frontend") && p != cli && meta.packages[p].lib().is_some())
        .collect();
    let mut outs = Vec::new();
    for &m in &macros {
        let names = paths(meta, &core, |module| format!("sensai_engine::{module}"));
        outs.push(single(meta, m, Kind::Macro, names));
    }
    outs.push(merged(meta, ENGINE, &core, "engine")?);
    outs.push(merged(meta, FRONTEND, &front, "frontend")?);
    let mut bin = single(
        meta,
        cli,
        Kind::Bin,
        paths(meta, &front, |m| format!("sensai::{m}")),
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

fn lib_name(pkg: &Package) -> String {
    pkg.lib()
        .map_or_else(|| pkg.name.replace('-', "_"), |t| t.name.clone())
}

fn module(pkg: &Package) -> String {
    let lib = lib_name(pkg);
    lib.strip_prefix("sensai_").unwrap_or(&lib).to_string()
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

fn single(meta: &Meta, p: usize, kind: Kind, names: BTreeMap<String, String>) -> Out {
    let pkg = &meta.packages[p];
    let member = Member {
        pkg: p,
        module: String::new(),
        to: PathBuf::new(),
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
        members.push(Member {
            pkg: p,
            module: module(pkg),
            to: PathBuf::from("src").join(rel),
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
        None => "Model files, tasks and serving over sensai-engine".into(),
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
