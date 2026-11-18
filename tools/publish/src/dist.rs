//! Package files for the binary on PyPI and npm.

use std::fs;
use std::path::Path;

use serde_json::{Value as Json, json};
use toml::{Table, Value};

use crate::Result;
use crate::plan::Out;

/// What a release names in every package it writes.
pub struct Release {
    pub version: String,
    pub repository: Option<String>,
}

/// npm platform, CPU and C library for each binary build.
pub const PLATFORMS: [(&str, &str, Option<&str>); 3] = [
    ("linux", "x64", Some("glibc")),
    ("linux", "arm64", Some("glibc")),
    ("darwin", "arm64", None),
];

/// A maturin `pyproject.toml` that ships the binary in a wheel.
pub fn pyproject(out: &Out, rel: &Release, dir: &Path) -> Result<()> {
    let mut project = Table::new();
    project.insert("name".into(), out.name.clone().into());
    project.insert("description".into(), out.description.clone().into());
    project.insert("license".into(), "MIT".into());
    project.insert("requires-python".into(), ">=3.9".into());
    project.insert("dynamic".into(), Value::Array(vec!["version".into()]));
    if let Some(repo) = &rel.repository {
        let mut urls = Table::new();
        urls.insert("Repository".into(), repo.clone().into());
        project.insert("urls".into(), urls.into());
    }
    let mut build = Table::new();
    build.insert(
        "requires".into(),
        Value::Array(vec!["maturin>=1.9,<2".into()]),
    );
    build.insert("build-backend".into(), "maturin".into());
    let mut maturin = Table::new();
    maturin.insert("bindings".into(), "bin".into());
    let mut tool = Table::new();
    tool.insert("maturin".into(), maturin.into());
    let mut doc = Table::new();
    doc.insert("project".into(), project.into());
    doc.insert("build-system".into(), build.into());
    doc.insert("tool".into(), tool.into());
    fs::write(dir.join("pyproject.toml"), toml::to_string(&doc)?)?;
    Ok(())
}

/// npm packages per platform, plus a launcher that picks one.
pub fn npm(out: &Out, rel: &Release, launcher: &Path, base: &Path) -> Result<()> {
    let name = &out.name;
    let mut optional = serde_json::Map::new();
    for (os, cpu, libc) in PLATFORMS {
        let pkg = format!("{name}-{os}-{cpu}");
        let mut doc = common(&pkg, out, rel);
        doc["os"] = json!([os]);
        doc["cpu"] = json!([cpu]);
        doc["files"] = json!([format!("bin/{name}")]);
        if let Some(libc) = libc {
            doc["libc"] = json!([libc]);
        }
        write_json(&base.join(&pkg), &doc)?;
        optional.insert(pkg, rel.version.clone().into());
    }
    let mut doc = common(name, out, rel);
    let launcher_bin = format!("bin/{name}.js");
    doc["bin"] = json!({ name.as_str(): launcher_bin, crate::plan::CLI_ALIAS: launcher_bin });
    doc["files"] = json!(["bin/"]);
    doc["optionalDependencies"] = optional.into();
    let dir = base.join(name);
    write_json(&dir, &doc)?;
    fs::create_dir_all(dir.join("bin"))?;
    fs::copy(launcher, dir.join("bin").join(format!("{name}.js")))?;
    Ok(())
}

fn common(pkg: &str, out: &Out, rel: &Release) -> Json {
    let mut doc = json!({
        "name": pkg,
        "version": rel.version,
        "description": out.description,
        "license": "MIT",
    });
    if let Some(repo) = &rel.repository {
        doc["repository"] = json!({ "type": "git", "url": format!("git+{repo}.git") });
    }
    doc
}

fn write_json(dir: &Path, doc: &Json) -> Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(
        dir.join("package.json"),
        serde_json::to_string_pretty(doc)? + "\n",
    )?;
    Ok(())
}
