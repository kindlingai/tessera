//! Rules a wasm module proposes, given sizes as JSON.

mod abi;
mod guest;
mod site;
mod text;

use abi::{ABI, Ask, Kind, Manifest, Offer, Site};
use guest::Guest;
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
    fn ask(&self, site: Site) -> Result<Option<Vec<Offer>>, Error> {
        let rule = self.rule;
        let input = serde_json::to_vec(&Ask { rule, site })?;
        let output = self.guest.call("propose", Some(&input))?;
        Ok(serde_json::from_slice(&output)?)
    }
}

impl Propose for Asking {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let (site, nodes) = site::find(tree, at, self.kind)?;
        match self.ask(site) {
            Ok(offers) => site::drafts(offers?, self.kernels, tree, &nodes),
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

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// The rules `bytes` declares. They live until the process ends.
pub fn load(bytes: &[u8]) -> Result<Vec<&'static Rule>, Error> {
    let guest = Arc::new(Guest::new(bytes)?);
    let manifest: Manifest = serde_json::from_slice(&guest.call("manifest", None)?)?;
    if manifest.abi != ABI {
        return Err(Error::Abi(manifest.abi));
    }
    let mut kernels = Vec::with_capacity(manifest.kernels.len());
    for k in manifest.kernels {
        kernels.push(Fixed {
            name: leak(k.name),
            entry: leak(k.entry),
            metal: leak(k.metal),
            cuda: leak(k.cuda),
        });
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
    Ok(rules)
}
