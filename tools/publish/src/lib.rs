//! Merges workspace crates into the crates that go to crates.io.

mod dist;
mod emit;
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

/// Each merged library's member directories.
pub fn merged(root: &Path) -> Result<Vec<Vec<PathBuf>>> {
    let meta = meta::load(root)?;
    let mut groups = Vec::new();
    for o in plan::plan(&meta)? {
        if o.members.len() > 1 {
            let dirs = o
                .members
                .iter()
                .map(|m| meta.packages[m.pkg].dir().to_path_buf());
            groups.push(dirs.collect());
        }
    }
    Ok(groups)
}

/// Writes the merged crates to `target/publish` or `[dir]`.
pub fn run() -> Result<()> {
    let meta = meta::load(&std::env::current_dir()?)?;
    let (version, dir) = args()?;
    let base = dir.unwrap_or_else(|| meta.workspace_root.join("target/publish"));
    let outs = plan::plan(&meta)?;
    let engine = meta
        .packages
        .iter()
        .find(|p| p.name == plan::ENGINE)
        .ok_or("no engine")?;
    let rel = dist::Release {
        version: version.unwrap_or_else(|| engine.version.clone()),
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
    let cli = outs.iter().find(|o| o.name == plan::CLI).ok_or("no cli")?;
    let launcher = meta.workspace_root.join("tools/publish/npm/sensaix.js");
    dist::npm(cli, &rel, &launcher, &base.join("npm"))?;
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

/// `[--version <v>] [dir]`
fn args() -> Result<(Option<String>, Option<PathBuf>)> {
    let mut version = None;
    let mut dir = None;
    let mut given = std::env::args().skip(1);
    while let Some(a) = given.next() {
        match a.as_str() {
            "--version" => version = Some(given.next().ok_or("--version needs a value")?),
            _ if dir.is_none() => dir = Some(PathBuf::from(a)),
            _ => return Err(format!("unexpected argument {a}").into()),
        }
    }
    Ok((version, dir))
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
