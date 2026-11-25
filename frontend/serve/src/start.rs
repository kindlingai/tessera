//! Loads one model on a thread and serves HTTP.

use crate::Slot;
use crate::task::{self, Load};
use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use tessera_bundle::Overrides;
use tessera_engine::Recipe;
use tessera_engine::progress::{Open, Progress, Unit};
use tessera_engine::wasm::Module;
use tessera_tasks::api::Info;

/// The flags for `Options::flag` and `Options::switch`.
pub fn flags() -> String {
    let head = "[--host <addr>] [--port <n>] [--max-model-len <tokens>] [--slots <n>] \
[--max-batch-rows <n>] [--layers <n>] [--kv-dtype <f32|bf16|fp8>] [--kv-cache-tokens <tokens>] [--served-model-name <name>]";
    let tail = "[--max-connections <n>] [--reject-unknown-fields] [--require-favored] [--explain-lowering]";
    format!("{head} {} {tail}", task::usage())
}

pub struct Options {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    /// Tokens one answer holds, prompt included.
    pub max_model_len: usize,
    /// Answers in flight at once.
    pub slots: usize,
    /// Optional row cap per batched step.
    pub max_batch_rows: Option<usize>,
    /// Optional KV cache float form: f32, bf16 or fp8.
    pub kv_dtype: Option<String>,
    /// Optional shared paged KV row pool across all slots.
    pub kv_cache_tokens: Option<usize>,
    pub name: Option<String>,
    /// The `--task` value, one the registry lists.
    pub(crate) task: &'static str,
    /// Values of the flags task entries own. Switches hold "".
    pub(crate) given: BTreeMap<&'static str, String>,
    /// Unknown request fields refuse the request.
    pub strict: bool,
    /// Fail lowering unless the favored rule covers a step.
    pub require_favored: bool,
    /// Print top slow or unfused lowering sites on boot.
    pub explain_lowering: bool,
    pub max_connections: usize,
    /// A recipe module that may build the model's graph.
    pub module: Option<Arc<Module>>,
    /// Lowering choices. Empty pins take the model's own.
    pub lower: Option<Recipe>,
    /// Weight-name rules, ahead of `TESSERA_NAMES`.
    pub names: Option<PathBuf>,
    /// This rank and its peers. Empty serves alone.
    pub ranks: crate::ranks::Ranks,
    /// Config values the recipe sets, by file.
    pub config: Overrides,
    /// Where the load reports its steps.
    pub progress: Progress,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            host: "127.0.0.1".into(),
            port: 8080,
            max_model_len: 4096,
            slots: 4,
            max_batch_rows: None,
            kv_dtype: None,
            kv_cache_tokens: None,
            name: None,
            task: "generate",
            given: BTreeMap::new(),
            strict: false,
            require_favored: false,
            explain_lowering: false,
            max_connections: 64,
            module: None,
            lower: None,
            names: None,
            ranks: Default::default(),
            config: Overrides::default(),
            progress: Progress::default(),
        }
    }
}

impl Options {
    /// Takes a flag with no value. `false` if unknown.
    pub fn switch(&mut self, flag: &str) -> bool {
        match flag {
            "--reject-unknown-fields" => self.strict = true,
            "--require-favored" => self.require_favored = true,
            "--explain-lowering" => self.explain_lowering = true,
            _ => match task::owned(flag) {
                Some(owned) if owned.value.is_empty() => {
                    self.given.insert(owned.name, String::new());
                }
                _ => return false,
            },
        }
        true
    }

    /// Takes `flag value`. `None` if either is bad.
    pub fn flag(&mut self, flag: &str, value: String) -> Option<()> {
        match flag {
            "--host" => self.host = value,
            "--port" => self.port = value.parse().ok()?,
            "--max-model-len" => self.max_model_len = value.parse().ok()?,
            "--slots" => self.slots = value.parse().ok()?,
            "--max-batch-rows" => {
                let n: usize = value.parse().ok()?;
                if n == 0 {
                    return None;
                }
                self.max_batch_rows = Some(n);
            }
            "--kv-dtype" => {
                let form = tessera_engine::form::named(&value)?;
                if !(form.stores() && form.dtype().is_float()) {
                    return None;
                }
                self.kv_dtype = Some(form.name().into());
            }
            "--kv-cache-tokens" => {
                let n: usize = value.parse().ok()?;
                if n == 0 {
                    return None;
                }
                self.kv_cache_tokens = Some(n);
            }
            "--max-connections" => self.max_connections = value.parse().ok()?,
            // Keeps the first n layers, for quick kernel timing.
            "--layers" => {
                let n: u64 = value.parse().ok().filter(|n| *n > 0)?;
                self.config
                    .set("config.json", "num_hidden_layers", n.into());
            }
            "--served-model-name" => self.name = Some(value),
            "--task" => self.task = task::names().into_iter().find(|t| *t == value)?,
            _ => match task::owned(flag) {
                Some(owned) if !owned.value.is_empty() && (owned.check)(&value) => {
                    self.given.insert(owned.name, value);
                }
                Some(_) => return None,
                None => return self.ranks.flag(flag, &value),
            },
        }
        Some(())
    }
}

/// Loads the model on a thread. The slot fills after.
pub fn spawn(options: Options) -> Slot {
    let slot: Slot = Arc::new(OnceLock::new());
    let loading = slot.clone();
    std::thread::spawn(move || load(options, loading));
    slot
}

/// Loads the model, then answers requests until killed.
pub fn run(mut options: Options) -> Result<(), String> {
    let (host, port) = (options.host.clone(), options.port);
    let limits = crate::http::Limits {
        max_connections: options.max_connections,
        ..Default::default()
    };
    if options.ranks.follows() {
        load(options, Slot::default());
        return Ok(());
    }
    let open = Arc::new(Open::default());
    options.progress = options.progress.also(open.clone());
    let slot = spawn(options);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|e| e.to_string())?;
    eprintln!("listening on {host}:{port}");
    crate::http::run(listener, limits, move |request| {
        crate::route_open(&slot, &open, request)
    });
    Ok(())
}

/// Loads the model into `slot`. A failure ends the process.
fn load(mut options: Options, slot: Slot) {
    let mut settings = tessera_frontend_settings::read();
    settings.names = options.names.clone().or(settings.names);
    let info = Info {
        name: options
            .name
            .clone()
            .unwrap_or_else(|| options.model.display().to_string()),
        context: options.max_model_len,
    };
    let name = options.name.as_deref().unwrap_or("model");
    let step = (options.progress).step(format!("load {name}"), Unit::Items, None);
    options.progress = step.progress();
    let load = Load {
        options: &options,
        info,
        slot: &slot,
        step: Mutex::new(Some(step)),
    };
    let failed = task::pick(&options).and_then(|entry| (entry.open)(&load, settings));
    if let Err(e) = failed {
        eprintln!("tesserax: {e}");
        std::process::exit(1);
    }
}
