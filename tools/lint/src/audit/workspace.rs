//! Workspace crates and their dependencies.

use std::path::Path;
use std::process::Command;

/// Which table of a manifest holds a dependency.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Normal,
    Dev,
    Build,
}

impl Kind {
    pub fn section(self) -> &'static str {
        match self {
            Kind::Normal => "[dependencies]",
            Kind::Dev => "[dev-dependencies]",
            Kind::Build => "[build-dependencies]",
        }
    }

    pub fn verb(self) -> &'static str {
        match self {
            Kind::Normal => "depends on",
            Kind::Dev => "dev-depends on",
            Kind::Build => "build-depends on",
        }
    }
}

/// A dependency on another workspace crate.
pub struct Dep {
    pub name: String,
    pub kind: Kind,
}

pub struct Pkg {
    pub name: String,
    /// The crate's directory below the repository, with a trailing `/`.
    pub dir: String,
    pub manifest: String,
    pub deps: Vec<Dep>,
}

impl Pkg {
    pub fn manifest_path(&self) -> String {
        format!("{}Cargo.toml", self.dir)
    }

    /// The manifest line naming `dep` in a `kind` table.
    pub fn dep_line(&self, dep: &str, kind: Kind) -> usize {
        let mut inside = false;
        for (i, line) in self.manifest.lines().enumerate() {
            let t = line.trim();
            if t.starts_with('[') {
                inside = t == kind.section();
            }
            let named = t
                .strip_prefix(dep)
                .is_some_and(|rest| rest.starts_with(['.', ' ', '=']));
            if inside && named {
                return i + 1;
            }
        }
        1
    }
}

#[derive(serde::Deserialize)]
struct Metadata {
    packages: Vec<Package>,
}

#[derive(serde::Deserialize)]
struct Package {
    name: String,
    manifest_path: String,
    dependencies: Vec<Dependency>,
}

#[derive(serde::Deserialize)]
struct Dependency {
    name: String,
    kind: Option<String>,
    path: Option<String>,
}

fn kind(name: Option<&str>) -> Kind {
    match name {
        Some("dev") => Kind::Dev,
        Some("build") => Kind::Build,
        _ => Kind::Normal,
    }
}

/// Parses `cargo metadata --no-deps` output.
pub fn parse(root: &Path, json: &[u8]) -> Result<Vec<Pkg>, String> {
    let meta: Metadata = serde_json::from_slice(json).map_err(|e| format!("metadata: {e}"))?;
    let mut out = Vec::new();
    for p in meta.packages {
        let manifest = Path::new(&p.manifest_path);
        let dir = manifest.parent().and_then(|d| d.strip_prefix(root).ok());
        let Some(dir) = dir else {
            continue;
        };
        let mut dir = dir.to_string_lossy().into_owned();
        if !dir.is_empty() {
            dir.push('/');
        }
        let text = std::fs::read_to_string(manifest).unwrap_or_default();
        let deps = p
            .dependencies
            .iter()
            .filter(|d| d.path.is_some())
            .map(|d| Dep {
                name: d.name.clone(),
                kind: kind(d.kind.as_deref()),
            })
            .collect();
        out.push(Pkg {
            name: p.name,
            dir,
            manifest: text,
            deps,
        });
    }
    Ok(out)
}

pub fn load(root: &Path) -> Result<Vec<Pkg>, String> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("cargo metadata failed: {}", err.trim()));
    }
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    parse(&root, &out.stdout)
}
