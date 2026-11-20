//! The parts of `cargo metadata` the merge reads.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::Result;

#[derive(Deserialize)]
pub(crate) struct Meta {
    pub packages: Vec<Package>,
    pub workspace_root: PathBuf,
}

#[derive(Deserialize)]
pub(crate) struct Package {
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub edition: String,
    pub license: Option<String>,
    pub manifest_path: PathBuf,
    pub dependencies: Vec<Dep>,
    pub targets: Vec<Target>,
    pub features: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct Dep {
    pub name: String,
    pub req: String,
    pub kind: Option<String>,
    pub rename: Option<String>,
    pub optional: bool,
    pub uses_default_features: bool,
    pub features: Vec<String>,
    pub target: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct Target {
    pub name: String,
    pub kind: Vec<String>,
    pub src_path: PathBuf,
}

impl Package {
    pub fn dir(&self) -> &Path {
        self.manifest_path
            .parent()
            .expect("a manifest sits in a directory")
    }

    /// The library or proc-macro target.
    pub fn lib(&self) -> Option<&Target> {
        let lib = |t: &&Target| t.kind.iter().any(|k| k == "lib" || k == "proc-macro");
        self.targets.iter().find(lib)
    }

    pub fn is_macro(&self) -> bool {
        self.targets
            .iter()
            .any(|t| t.kind.iter().any(|k| k == "proc-macro"))
    }

    /// Normal dependencies, without dev and build ones.
    pub fn normal(&self) -> impl Iterator<Item = &Dep> {
        self.dependencies.iter().filter(|d| d.kind.is_none())
    }
}

impl Dep {
    /// The name the depending crate uses.
    pub fn key(&self) -> &str {
        self.rename.as_deref().unwrap_or(&self.name)
    }
}

pub(crate) fn load(root: &Path) -> Result<Meta> {
    let out = Command::new("cargo")
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .output()?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into());
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}
