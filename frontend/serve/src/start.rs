//! Loads each model on a thread and serves HTTP.

use crate::task::{self, Load};
use crate::{Models, Slot};
use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tessera_bundle::Overrides;
use tessera_engine::hf::LAYERS;
use tessera_engine::progress::{Open, Progress, Unit};
use tessera_engine::turns::{Rules, Turns};
use tessera_engine::wasm::Module;
use tessera_engine::{Memory, Recipe};
use tessera_tasks::api::Info;

/// The flags for `Options::flag` and `Options::switch`.
pub fn flags() -> String {
    let head = "[--host <addr>] [--port <n>] [--port-file <path>] [--max-model-len <tokens>] [--slots <n>] \
[--max-batch-rows <n>] [--layers <n>] [--kv-dtype <f32|bf16|fp8>] [--dtype <spot>=<dtype>]... [--shape <kernel>:<knob>=<n>]... [--kv-cache-tokens <tokens>] [--kv-store <spec>]... [--kv-keep <system|sessions|all>] [--kv-system-after <n>] [--served-model-name <name>]";
    let tail = "[--max-connections <n>] [--api-key <key>] [--cors-origin <origin>] [--reject-unknown-fields] [--require-favored] [--explain-lowering] [--allow-failed-lowering] [--allow-weight-streaming] [--attic] \
[--priority <n>] [--swap-quantum <x>] [--also <model and its flags>]...";
    format!("{head} {} {tail}", task::usage())
}

pub struct Options {
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    /// Gets the bound port once the server listens.
    pub port_file: Option<PathBuf>,
    /// Tokens one answer holds, prompt included.
    pub max_model_len: usize,
    /// Answers in flight at once.
    pub slots: usize,
    /// Optional row cap per batched step.
    pub max_batch_rows: Option<usize>,
    /// Optional KV cache float form: f32, bf16 or fp8.
    pub kv_dtype: Option<String>,
    /// Dtype overrides by spot, after the recipe's.
    pub dtypes: Vec<tessera_engine::Dtype>,
    /// Kernel shape values, after the recipe's.
    pub shapes: tessera_engine::Knobs,
    /// Optional shared paged KV row pool across all slots.
    pub kv_cache_tokens: Option<usize>,
    /// Repeatable `--kv-store` tier specs, or `off`/`none`.
    pub kv_stores: Vec<String>,
    /// Default retention policy for tiers omitting `keep=`.
    pub kv_keep: Option<tessera_engine::kv::Keep>,
    /// Conversations before `keep=system` stores a prefix.
    pub kv_system_after: Option<u32>,
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
    /// Boot although lowering fell off the fast path.
    pub allow_failed_lowering: bool,
    /// Weights may page from the host or load by stage.
    pub allow_weight_streaming: bool,
    /// Borrow GB10 display carveout from `spark-attic` `atticd`.
    pub attic: bool,
    pub max_connections: usize,
    /// Optional API key required on non-health routes.
    pub api_key: Option<String>,
    /// Optional `Access-Control-Allow-Origin` response header value.
    pub cors_origin: Option<String>,
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
    /// A higher number steps first and takes memory first.
    pub priority: u32,
    /// Resident seconds a busy model keeps per load second.
    pub quantum: f64,
    /// More models to serve beside this one.
    pub also: Vec<Options>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            host: "127.0.0.1".into(),
            port: 8080,
            port_file: None,
            max_model_len: 4096,
            slots: 4,
            max_batch_rows: None,
            kv_dtype: None,
            dtypes: Vec::new(),
            shapes: Default::default(),
            kv_cache_tokens: None,
            kv_stores: Vec::new(),
            kv_keep: None,
            kv_system_after: None,
            name: None,
            task: "generate",
            given: BTreeMap::new(),
            strict: false,
            require_favored: false,
            explain_lowering: false,
            allow_failed_lowering: false,
            allow_weight_streaming: false,
            attic: false,
            max_connections: 64,
            api_key: None,
            cors_origin: None,
            module: None,
            lower: None,
            names: None,
            ranks: Default::default(),
            config: Overrides::default(),
            progress: Progress::default(),
            priority: 0,
            quantum: 4.0,
            also: Vec::new(),
        }
    }
}

impl Options {
    /// The recipe's kernel shape values, then the flags'.
    pub fn knobs(&self) -> tessera_engine::Knobs {
        let filed = self.lower.as_ref().map(|r| r.shape.clone());
        filed.unwrap_or_default().then(&self.shapes)
    }

    /// Takes a flag with no value. `false` if unknown.
    pub fn switch(&mut self, flag: &str) -> bool {
        match flag {
            "--reject-unknown-fields" => self.strict = true,
            "--require-favored" => self.require_favored = true,
            "--explain-lowering" => self.explain_lowering = true,
            "--allow-failed-lowering" => self.allow_failed_lowering = true,
            "--allow-weight-streaming" => self.allow_weight_streaming = true,
            "--attic" => self.attic = true,
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
            "--port-file" => self.port_file = Some(value.into()),
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
            "--shape" => self.shapes.parse(&value).ok()?,
            "--dtype" => self.dtypes.push(tessera_engine::Dtype::parse(&value).ok()?),
            "--kv-cache-tokens" => {
                let n: usize = value.parse().ok()?;
                if n == 0 {
                    return None;
                }
                self.kv_cache_tokens = Some(n);
            }
            "--kv-store" => {
                let mut trial = self.kv_stores.clone();
                trial.push(value.clone());
                tessera_engine::kv::KvConfig::parse_all(&trial, self.kv_keep).ok()?;
                self.kv_stores.push(value);
            }
            "--kv-keep" => {
                self.kv_keep = Some(tessera_engine::kv::Keep::parse(&value).ok()?);
            }
            "--kv-system-after" => {
                let n: u32 = value.parse().ok().filter(|&n| n > 0)?;
                self.kv_system_after = Some(n);
            }
            "--max-connections" => self.max_connections = value.parse().ok()?,
            "--api-key" => {
                if value.is_empty() {
                    return None;
                }
                self.api_key = Some(value);
            }
            "--cors-origin" => {
                if value.is_empty() || value.contains(['\r', '\n']) {
                    return None;
                }
                self.cors_origin = Some(value);
            }
            // Keeps the first n layers, for quick kernel timing.
            "--layers" => {
                let n: u64 = value.parse().ok().filter(|n| *n > 0)?;
                self.config.set("config.json", LAYERS.name, n.into());
            }
            "--served-model-name" => self.name = Some(value),
            "--priority" => self.priority = value.parse().ok()?,
            "--swap-quantum" => self.quantum = value.parse().ok().filter(|q| *q >= 0.0)?,
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

impl Options {
    /// The name requests give for this model.
    fn served(&self) -> String {
        let path = || self.model.display().to_string();
        self.name.clone().unwrap_or_else(path)
    }

    /// The table's rules, from these options and `TESSERA_MEMORY`.
    fn seating(&self) -> Rules {
        let memory = tessera_frontend_settings::read().memory;
        Rules {
            budget: match memory {
                Memory::Bytes(bytes) => Some(bytes),
                Memory::Device | Memory::Off => None,
            },
            learn: memory == Memory::Device,
            quantum: self.quantum,
        }
    }
}

/// Loads the model on a thread and waits for it.
pub fn open(options: Options) -> Result<Slot, String> {
    let (models, failed) = spawn_all(options);
    let slot = models.named[0].1.clone();
    loop {
        match failed.recv_timeout(Duration::from_millis(20)) {
            Ok(e) => return Err(e),
            Err(RecvTimeoutError::Timeout) if slot.get().is_none() => {}
            Err(_) => break,
        }
    }
    slot.get()
        .is_some()
        .then_some(slot)
        .ok_or("no model".into())
}

/// Loads each model on a thread and sends back failures.
fn spawn_all(mut options: Options) -> (Models, Receiver<String>) {
    let turns = Turns::new(options.seating());
    let also = std::mem::take(&mut options.also);
    let (failures, failed) = channel();
    let mut named = Vec::new();
    for options in std::iter::once(options).chain(also) {
        let slot = Slot::default();
        named.push((options.served(), slot.clone()));
        let (turns, failures) = (turns.clone(), failures.clone());
        std::thread::spawn(move || {
            if let Err(e) = load(options, slot, &turns) {
                let _ = failures.send(e);
            }
        });
    }
    (Models { named, turns }, failed)
}

/// Loads the models, then answers requests until killed.
pub fn run(mut options: Options) -> Result<(), String> {
    let (host, port) = (options.host.clone(), options.port);
    let port_file = options.port_file.take();
    let limits = crate::http::Limits {
        gate: Arc::new(crate::http::Capped::new(options.max_connections)),
        cors: options.cors_origin.clone(),
        ..Default::default()
    };
    let alone = options.ranks.alone() && options.also.iter().all(|o| o.ranks.alone());
    if !options.also.is_empty() && !alone {
        return Err("--also cannot combine with ranks".into());
    }
    if options.ranks.follows() {
        let turns = Turns::new(options.seating());
        return load(options, Slot::default(), &turns);
    }
    let open = Arc::new(Open::default());
    options.progress = options.progress.also(open.clone());
    for more in &mut options.also {
        more.progress = more.progress.also(open.clone());
    }
    let key = options.api_key.take();
    let (models, failed) = spawn_all(options);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|e| e.to_string())?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?;
    if let Some(path) = &port_file {
        write_port(path, bound.port()).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    eprintln!("listening on {bound}");
    let serving = std::thread::spawn(move || {
        crate::http::run(listener, limits, move |request| {
            if let Some(expected) = key.as_deref()
                && let Some(denied) = check_auth(expected, request)
            {
                return denied;
            }
            models.route(Some(&open), request)
        })
    });
    if let Ok(e) = failed.recv() {
        return Err(e);
    }
    serving
        .join()
        .map_err(|_| "the server thread panicked".into())
}

/// Writes `port` to `path` by rename, never in part.
fn write_port(path: &Path, port: u16) -> std::io::Result<()> {
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    std::fs::write(&part, format!("{port}\n"))?;
    std::fs::rename(&part, path)
}

/// Constant-time equality over two byte slices.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Checks `Authorization: Bearer` or `x-api-key` on non-health routes.
pub(crate) fn check_auth(
    expected: &str,
    request: &crate::http::Request,
) -> Option<crate::http::Response> {
    if request.method == "GET"
        && matches!(
            request.path.as_str(),
            "/health" | "/v1/health" | "/v1/health/ready" | "/ping"
        )
    {
        return None;
    }
    let bearer = request.header("authorization").and_then(|v| {
        v.strip_prefix("Bearer ")
            .or_else(|| v.strip_prefix("bearer "))
    });
    let provided = bearer.or_else(|| request.header("x-api-key"));
    if provided.is_some_and(|got| ct_eq(got.as_bytes(), expected.as_bytes())) {
        return None;
    }
    let body = if request.path == "/v1/messages" {
        serde_json::json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid api key"},
        })
    } else {
        serde_json::json!({
            "error": {
                "message": "invalid api key",
                "type": "AuthenticationError",
                "param": null,
                "code": 401,
            }
        })
    };
    Some(crate::http::Response::json(401, &body))
}

/// Loads the model into `slot`.
fn load(mut options: Options, slot: Slot, turns: &Arc<Turns>) -> Result<(), String> {
    let mut settings = tessera_frontend_settings::read();
    settings.names = options.names.clone().or(settings.names);
    settings.weight_streaming |= options.allow_weight_streaming;
    if options.attic && settings.attic.is_none() {
        settings.attic = Some(tessera_frontend_settings::attic_default());
    }
    let info = Info {
        name: options.served(),
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
        asks: settings.asks.clone(),
        turns,
    };
    task::pick(&options).and_then(|entry| (entry.open)(&load, settings))
}
