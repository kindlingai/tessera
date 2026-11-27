//! Merges workspace crates into the crates that go to crates.io.

mod dist;
mod emit;
mod lint;
mod manifest;
mod meta;
mod plan;
mod rewrite;
mod tokens;

#[cfg(test)]
mod rewrite_test;

use std::fs;
use std::path::{Path, PathBuf};

use toml::{Table, Value};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Writes the crates.io crates, or with `--lint` the [`lint`] crate.
pub fn run() -> Result<()> {
    let meta = meta::load(&std::env::current_dir()?)?;
    let a = args()?;
    let (outs, default) = match a.lint {
        true => (lint::plan(&meta)?, "target/publish-lint"),
        false => (plan::plan(&meta)?, "target/publish"),
    };
    let base = a.dir.unwrap_or_else(|| meta.workspace_root.join(default));
    let engine = meta
        .packages
        .iter()
        .find(|p| p.name == plan::ENGINE)
        .ok_or("no engine")?;
    let rel = dist::Release {
        version: a.version.unwrap_or_else(|| engine.version.clone()),
        repository: repository(&meta.workspace_root)?,
    };
    let mut owners = manifest::Owners::new();
    for o in &outs {
        for m in &o.members {
            owners.insert(m.pkg, o.name.as_str());
        }
    }
    if base.exists() {
        fs::remove_dir_all(&base)?;
    }
    for o in &outs {
        emit::emit(&meta, o, &owners, &rel, &base)?;
    }
    if let Some(cli) = outs.iter().find(|o| o.name == plan::CLI) {
        let launcher = meta.workspace_root.join("tools/publish/npm/tesserax.js");
        dist::npm(cli, &rel, &launcher, &base.join("npm"))?;
    }
    let mut workspace = Table::new();
    workspace.insert("resolver".into(), "3".into());
    let members = outs.iter().map(|o| Value::from(o.name.clone())).collect();
    workspace.insert("members".into(), Value::Array(members));
    let mut doc = Table::new();
    doc.insert("workspace".into(), workspace.into());
    fs::write(base.join("Cargo.toml"), toml::to_string(&doc)?)?;
    fs::copy(
        meta.workspace_root.join("Cargo.lock"),
        base.join("Cargo.lock"),
    )?;
    println!(
        "tessera-publish: {} crates in {}",
        outs.len(),
        base.display()
    );
    Ok(())
}

#[derive(Default)]
struct Args {
    version: Option<String>,
    dir: Option<PathBuf>,
    lint: bool,
}

/// `[--version <v>] [--lint] [dir]`
fn args() -> Result<Args> {
    let mut out = Args::default();
    let mut given = std::env::args().skip(1);
    while let Some(a) = given.next() {
        match a.as_str() {
            "--version" => out.version = Some(given.next().ok_or("--version needs a value")?),
            "--lint" => out.lint = true,
            _ if out.dir.is_none() => out.dir = Some(PathBuf::from(a)),
            _ => return Err(format!("unexpected argument {a}").into()),
        }
    }
    Ok(out)
}

/// The workspace's `repository`, if it names one.
fn repository(root: &Path) -> Result<Option<String>> {
    let doc: Table = toml::from_str(&fs::read_to_string(root.join("Cargo.toml"))?)?;
    let repo = doc
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("repository"))
        .and_then(Value::as_str);
    Ok(repo.map(String::from))
}
