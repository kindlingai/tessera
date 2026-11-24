//! A recipe: weights, lowering choices and serve flags.

mod build;
mod hub;
#[cfg(test)]
mod hub_test;
#[cfg(test)]
mod lib_test;
mod source;
#[cfg(test)]
mod source_test;

pub use hub::{fetch as fetch_hf, fetch_as, safe_path};
pub use source::Source;

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Errors that stop a recipe from opening.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("{0}: {1}")]
    Toml(PathBuf, toml::de::Error),
    #[error("{0}: {1}")]
    Source(String, &'static str),
    #[error("{0}: {1}")]
    Command(String, String),
    #[error("{0}: {1}")]
    Http(String, String),
    #[error("weights need exactly one of path or hf")]
    Weights,
    #[error("no cache directory: set TESSERA_RECIPES")]
    NoCache,
}

/// Where a recipe's weights come from.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Weights {
    /// A local directory, relative to the recipe.
    pub path: Option<PathBuf>,
    /// `owner/name@commit` on Hugging Face.
    pub hf: Option<String>,
    /// More `owner/name@commit` repositories, each passed to serve
    /// as `--<name> <dir>`.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
    /// The expert width an exlr8 release is fetched at.
    pub width: Option<u32>,
}

/// One `recipe.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub name: String,
    pub weights: Weights,
    /// Weight-name rules, relative to the recipe.
    pub names: Option<PathBuf>,
    /// Lowering choices. `None` keeps the model's own.
    pub lower: Option<tessera_engine::Recipe>,
    /// A wasm module of lowering rules, relative to the recipe.
    pub module: Option<PathBuf>,
    /// Serve flags by name, with `_` for `-`.
    #[serde(default)]
    pub serve: toml::Table,
    /// Config values by file, relative to the weights.
    #[serde(default)]
    pub config: toml::Table,
    /// The directory the recipe was read from.
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Recipe {
    /// Reads `source`, cloned into `cache` when remote.
    pub fn open(source: &str, cache: Option<&Path>) -> Result<Self, Error> {
        let dir = Source::parse(source)?.resolve(cache)?;
        Self::read(&dir)
    }

    /// Reads `recipe.toml` in `dir`.
    pub fn read(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("recipe.toml");
        let text = std::fs::read_to_string(&path).map_err(io(&path))?;
        Self::parse(&text, &path)
    }

    /// `text` as the recipe file at `path`.
    pub fn parse(text: &str, path: &Path) -> Result<Self, Error> {
        let mut recipe: Recipe = toml::from_str(text).map_err(|e| Error::Toml(path.into(), e))?;
        recipe.dir = path.parent().unwrap_or(path).to_path_buf();
        Ok(recipe)
    }

    /// The weight-name rules file, if the recipe has one.
    pub fn names(&self) -> Option<PathBuf> {
        self.names.as_ref().map(|n| self.dir.join(n))
    }

    /// The rules module, if the recipe has one.
    pub fn module(&self) -> Option<PathBuf> {
        self.module.as_ref().map(|m| self.dir.join(m))
    }
    /// The module's bytes, built first if it is a crate.
    pub fn module_bytes(&self, cache: Option<&Path>) -> Result<Option<Vec<u8>>, Error> {
        let Some(path) = self.module() else {
            return Ok(None);
        };
        let file = match path.join("Cargo.toml").is_file() {
            true => build::module(&path, cache.ok_or(Error::NoCache)?)?,
            false => path,
        };
        Ok(Some(std::fs::read(&file).map_err(io(&file))?))
    }

    /// The weights directory, fetched into `cache` when remote.
    pub fn weights(&self, cache: Option<&Path>, token: Option<&str>) -> Result<PathBuf, Error> {
        match (&self.weights.path, &self.weights.hf) {
            (Some(path), None) => Ok(self.dir.join(path)),
            (None, Some(hf)) => {
                hub::snapshot(hf, cache.ok_or(Error::NoCache)?, token, self.weights.width)
            }
            _ => Err(Error::Weights),
        }
    }

    /// Each extra repository, fetched into `cache`, as a serve flag.
    pub fn extras(
        &self,
        cache: Option<&Path>,
        token: Option<&str>,
    ) -> Result<Vec<(String, PathBuf)>, Error> {
        let mut flags = Vec::new();
        for (name, spec) in &self.weights.extra {
            let dir = hub::snapshot(spec, cache.ok_or(Error::NoCache)?, token, None)?;
            flags.push((format!("--{}", name.replace('_', "-")), dir));
        }
        Ok(flags)
    }

    /// `[config]` as JSON tables, by file.
    pub fn overrides(&self) -> serde_json::Map<String, serde_json::Value> {
        match serde_json::to_value(&self.config) {
            Ok(serde_json::Value::Object(files)) => files,
            _ => serde_json::Map::new(),
        }
    }

    /// `[serve]` as `--flag` and value. `true` gives a bare flag.
    pub fn flags(&self) -> Vec<(String, Option<String>)> {
        let mut flags = Vec::new();
        for (key, value) in &self.serve {
            let flag = format!("--{}", key.replace('_', "-"));
            match value {
                toml::Value::Boolean(false) => {}
                toml::Value::Boolean(true) => flags.push((flag, None)),
                toml::Value::String(s) => flags.push((flag, Some(s.clone()))),
                other => flags.push((flag, Some(other.to_string()))),
            }
        }
        flags
    }
}

/// Tags an IO failure with its path.
fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |e| Error::Io(path.to_path_buf(), e)
}

/// Whether `text` is a full 40-digit commit hash.
fn pinned(text: &str) -> bool {
    text.len() == 40 && text.bytes().all(|b| b.is_ascii_hexdigit())
}
