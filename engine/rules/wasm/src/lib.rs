//! Rules a wasm module proposes, given sizes as JSON.

mod abi;
mod guest;
mod site;
mod text;

use abi::{ABI, Ask, Entry, Kind, Manifest, Site};
use guest::{BUILD_FUEL, FUEL, Guest};
use sensai_graph::Graph;
use sensai_lower::{Draft, Propose, Rule, T, Tree};
use std::sync::Arc;
use text::Fixed;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("wasm: {0}")]
    Wasm(String),
    #[error("the module has no export {0}")]
    Missing(&'static str),
    #[error("the module's JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the module speaks abi {0}, the host {ABI}")]
    Abi(u32),
    #[error("an earlier call panicked")]
    Poisoned,
    #[error(transparent)]
    Kernel(#[from] sensai_device::Error),
}

impl Error {
    fn wasm(e: impl std::fmt::Display) -> Self {
        Error::Wasm(e.to_string())
    }
}

/// One declared rule, asking the shared module.
struct Asking {
    guest: Arc<Guest>,
    rule: &'static str,
    kind: Kind,
    kernels: &'static [Fixed],
}

impl Asking {
    /// The module's offers, or the reason it gave none.
    fn ask(&self, site: Site) -> Result<Option<Vec<Entry>>, Error> {
        let rule = self.rule.into();
        let input = serde_json::to_vec(&Ask { rule, site })?;
        let output = self.guest.call("propose", Some(&input), FUEL)?;
        Ok(serde_json::from_slice(&output)?)
    }
}

impl Propose for Asking {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let (site, reads) = site::find(tree, at, self.kind)?;
        match self.ask(site) {
            Ok(offers) => site::drafts(offers?, self.kernels, tree, &reads),
            Err(e) => {
                eprintln!("wasm rule {}: {e}", self.rule);
                None
            }
        }
    }

    fn regions(&self) -> bool {
        self.kind != Kind::Matmul
    }
}

pub(crate) fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// A loaded module: its rules, and maybe a graph.
pub struct Module {
    guest: Arc<Guest>,
    pub rules: Vec<&'static Rule>,
    graph: bool,
}

/// The inputs to `graph` and `tables`.
#[derive(serde::Serialize)]
struct Build<'a> {
    config: &'a serde_json::value::RawValue,
    seqs: u64,
}

impl Module {
    /// The module in `bytes`. Its rules are never freed.
    pub fn open(bytes: &[u8]) -> Result<Self, Error> {
        let guest = Arc::new(Guest::new(bytes)?);
        let manifest = guest.call("manifest", None, FUEL)?;
        let manifest: Manifest = serde_json::from_slice(&manifest)?;
        if manifest.abi != ABI {
            return Err(Error::Abi(manifest.abi));
        }
        let mut kernels = Vec::with_capacity(manifest.kernels.len());
        for k in manifest.kernels {
            kernels.push(Fixed::open(k, &manifest.files)?);
        }
        let kernels: &'static [Fixed] = kernels.leak();
        let mut rules = Vec::with_capacity(manifest.rules.len());
        for r in manifest.rules {
            let rule = leak(r.name);
            let asking = Asking {
                guest: guest.clone(),
                rule,
                kind: r.site,
                kernels,
            };
            rules.push(Rule::made(rule, module_path!(), asking));
        }
        let graph = manifest.graph;
        Ok(Module {
            guest,
            rules,
            graph,
        })
    }

    /// Calls `name` with the config. `None` without a graph.
    fn build(&self, name: &'static str, config: &str, seqs: u64) -> Result<Option<Vec<u8>>, Error> {
        if !self.graph {
            return Ok(None);
        }
        let config = serde_json::value::RawValue::from_string(config.into())?;
        let input = serde_json::to_vec(&Build {
            config: &config,
            seqs,
        })?;
        Ok(Some(self.guest.call(name, Some(&input), BUILD_FUEL)?))
    }

    /// The graph for `config` JSON, if the module builds one.
    pub fn graph(&self, config: &str, seqs: u64) -> Result<Option<Graph>, Error> {
        let Some(out) = self.build("graph", config, seqs)? else {
            return Ok(None);
        };
        let text = std::str::from_utf8(&out).map_err(Error::wasm)?;
        Ok(Some(Graph::from_json(text).map_err(Error::wasm)?))
    }

    /// The tables that graph reads, by dotted path.
    pub fn tables(&self, config: &str) -> Result<Vec<(String, Vec<f32>)>, Error> {
        let Some(out) = self.build("tables", config, 1)? else {
            return Ok(Vec::new());
        };
        sensai_guest::decode(&out).ok_or_else(|| Error::wasm("tables: a short or bad buffer"))
    }
}

/// The rules `bytes` declares. They live until the process ends.
pub fn load(bytes: &[u8]) -> Result<Vec<&'static Rule>, Error> {
    Ok(Module::open(bytes)?.rules)
}
