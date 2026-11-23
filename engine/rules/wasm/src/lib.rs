//! Rules a wasm module proposes, given sizes as JSON.

mod abi;
mod guest;
mod site;
mod text;

use abi::{ABI, Answer, Ask, Kind, Manifest};
use guest::{BUILD_FUEL, FUEL, Guest};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tessera_graph::Graph;
use tessera_lower::{Draft, Propose, Rule, T, Tree};
use text::Fixed;

static DUMP: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Site-dump path and favored rule for wasm diagnostics.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub dump: Option<PathBuf>,
    pub favor: Option<String>,
}

/// Sets the process-wide fallback site dump path.
pub fn configure(dump: Option<PathBuf>) {
    *DUMP.lock().unwrap_or_else(|e| e.into_inner()) = dump;
}

/// One JSONL record written when site dumping is enabled.
#[derive(serde::Serialize)]
struct Record<'a> {
    ask: &'a serde_json::value::RawValue,
    response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

struct Sink {
    dump: Option<PathBuf>,
    favor: Option<String>,
    file: Mutex<Option<File>>,
}

impl Sink {
    fn new(trace: Trace) -> Self {
        Self {
            dump: trace.dump,
            favor: trace.favor,
            file: Mutex::new(None),
        }
    }

    fn path(&self) -> Option<PathBuf> {
        self.dump
            .clone()
            .or_else(|| DUMP.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    fn record(&self, input: &[u8], raw: Option<&[u8]>, error: Option<&str>) {
        let Some(target) = self.path() else {
            return;
        };
        let Ok(ask_str) = std::str::from_utf8(input) else {
            return;
        };
        let Ok(ask) = serde_json::value::RawValue::from_string(ask_str.to_string()) else {
            return;
        };
        let response = match raw {
            Some(bytes) => serde_json::from_slice(bytes).unwrap_or_else(|_| {
                serde_json::Value::String(String::from_utf8_lossy(bytes).into_owned())
            }),
            None => serde_json::Value::Null,
        };
        let Ok(line) = serde_json::to_string(&Record {
            ask: &ask,
            response,
            error,
        }) else {
            return;
        };
        let mut guard = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let path = match target.is_dir() {
                true => target.join("sites.jsonl"),
                false => target,
            };
            if let Some(dir) = path.parent()
                && !dir.as_os_str().is_empty()
            {
                let _ = std::fs::create_dir_all(dir);
            }
            *guard = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
                .ok();
        }
        if let Some(file) = guard.as_mut() {
            let _ = writeln!(file, "{line}");
        }
    }
}

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
    #[error("the module could not read the site: {0}")]
    Unread(String),
    #[error("an earlier call panicked")]
    Poisoned,
    #[error(transparent)]
    Kernel(#[from] tessera_device::Error),
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
    sink: Arc<Sink>,
}

impl Asking {
    fn is_favored(&self) -> bool {
        self.sink.favor.as_deref() == Some(self.rule)
    }

    fn warn(&self, msg: &str) {
        match self.is_favored() {
            true => eprintln!("wasm rule {} (favored): {msg}", self.rule),
            false => eprintln!("wasm rule {}: {msg}", self.rule),
        }
    }
}

impl Propose for Asking {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let (site, reads) = site::find(tree, at, self.kind)?;
        let config = match tree.config.as_deref().map(serde_json::from_str).transpose() {
            Ok(config) => config,
            Err(e) => {
                self.warn(&e.to_string());
                return None;
            }
        };
        let input = match serde_json::to_vec(&Ask {
            rule: self.rule.into(),
            site,
            config,
        }) {
            Ok(input) => input,
            Err(e) => {
                self.warn(&e.to_string());
                return None;
            }
        };
        let output = match self.guest.call("propose", Some(&input), FUEL) {
            Ok(output) => output,
            Err(e) => {
                let msg = e.to_string();
                self.sink.record(&input, None, Some(&msg));
                self.warn(&msg);
                return None;
            }
        };
        let answer: Answer = match serde_json::from_slice(&output) {
            Ok(answer) => answer,
            Err(e) => {
                let msg = e.to_string();
                self.sink.record(&input, Some(&output), Some(&msg));
                self.warn(&msg);
                return None;
            }
        };
        match answer {
            Answer::Unread(u) => {
                let msg = Error::Unread(u.unread).to_string();
                self.sink.record(&input, Some(&output), Some(&msg));
                self.warn(&msg);
                None
            }
            Answer::Offers(None) => {
                self.sink.record(&input, Some(&output), None);
                if self.is_favored() {
                    self.warn("declined site");
                }
                None
            }
            Answer::Offers(Some(entries)) => {
                let made = site::drafts(entries, self.kernels, tree, &reads);
                let err = made.is_none().then_some("offer did not fit site");
                self.sink.record(&input, Some(&output), err);
                if let Some(msg) = err
                    && self.is_favored()
                {
                    self.warn(msg);
                }
                made
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
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<u64>,
}

impl Module {
    /// The module in `bytes`. Its rules are never freed.
    pub fn open(bytes: &[u8]) -> Result<Self, Error> {
        Self::open_with(bytes, Trace::default())
    }

    /// Opens the module in `bytes` with diagnostic `trace`.
    pub fn open_with(bytes: &[u8], trace: Trace) -> Result<Self, Error> {
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
        let sink = Arc::new(Sink::new(trace));
        let mut rules = Vec::with_capacity(manifest.rules.len());
        for r in manifest.rules {
            let rule = leak(r.name);
            let asking = Asking {
                guest: guest.clone(),
                rule,
                kind: r.site,
                kernels,
                sink: sink.clone(),
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
    fn build(
        &self,
        name: &'static str,
        config: &str,
        seqs: u64,
        context: Option<u64>,
    ) -> Result<Option<Vec<u8>>, Error> {
        if !self.graph {
            return Ok(None);
        }
        let config = serde_json::value::RawValue::from_string(config.into())?;
        let input = serde_json::to_vec(&Build {
            config: &config,
            seqs,
            context,
        })?;
        Ok(Some(self.guest.call(name, Some(&input), BUILD_FUEL)?))
    }

    /// The graph for `config` JSON, if the module builds one.
    pub fn graph(
        &self,
        config: &str,
        seqs: u64,
        context: Option<u64>,
    ) -> Result<Option<Graph>, Error> {
        let Some(out) = self.build("graph", config, seqs, context)? else {
            return Ok(None);
        };
        let text = std::str::from_utf8(&out).map_err(Error::wasm)?;
        Ok(Some(Graph::from_json(text).map_err(Error::wasm)?))
    }

    /// The tables that graph reads, by dotted path.
    pub fn tables(&self, config: &str) -> Result<Vec<(String, Vec<f32>)>, Error> {
        let Some(out) = self.build("tables", config, 1, None)? else {
            return Ok(Vec::new());
        };
        tessera_guest::decode(&out).ok_or_else(|| Error::wasm("tables: a short or bad buffer"))
    }
}

/// The rules `bytes` declares. They live until the process ends.
pub fn load(bytes: &[u8]) -> Result<Vec<&'static Rule>, Error> {
    Ok(Module::open(bytes)?.rules)
}
