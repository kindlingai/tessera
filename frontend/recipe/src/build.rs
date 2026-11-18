//! A module crate in a recipe, built for wasm.

use crate::{Error, io};
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET: &str = "wasm32-unknown-unknown";

/// The crate's package name, as cargo names its output.
fn package(manifest: &Path) -> Result<String, Error> {
    let text = std::fs::read_to_string(manifest).map_err(io(manifest))?;
    let table: toml::Table = toml::from_str(&text).map_err(|e| Error::Toml(manifest.into(), e))?;
    let name = table.get("package").and_then(|p| p.get("name"));
    let name = name.and_then(|n| n.as_str());
    let name =
        name.ok_or_else(|| Error::Source(manifest.display().to_string(), "no package name"))?;
    Ok(name.replace('-', "_"))
}

/// Builds the crate in `dir`. Returns the module's path.
pub(crate) fn module(dir: &Path, cache: &Path) -> Result<PathBuf, Error> {
    let manifest = dir.join("Cargo.toml");
    let target = cache.join("modules");
    let mut cargo = Command::new("cargo");
    cargo.args(["build", "--release", "--quiet", "--target", TARGET]);
    cargo.arg("--manifest-path").arg(&manifest);
    let out = cargo.arg("--target-dir").arg(&target).output();
    let out = out.map_err(io(dir))?;
    if !out.status.success() {
        let said = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(Error::Command("cargo build".into(), said));
    }
    let name = package(&manifest)?;
    Ok(target
        .join(TARGET)
        .join("release")
        .join(format!("{name}.wasm")))
}
