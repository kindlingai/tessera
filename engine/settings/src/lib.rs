//! Knobs the front end sets once and the engine reads.

mod backend;
#[cfg(test)]
mod lib_test;
pub mod metrics;
#[cfg(test)]
mod metrics_test;
pub mod progress;
#[cfg(test)]
mod progress_test;

pub use backend::{Backend, ByBackend};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Where plans run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// The host running kernel trees in lockstep.
    Host,
    /// The host running native kernels.
    Cpu,
    Gpu(Backend),
}

/// The most bytes one plan may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Memory {
    /// Most of what the device reports.
    #[default]
    Device,
    Bytes(u64),
    /// No check.
    Off,
}

/// How weight files are read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reads {
    /// Through the file cache.
    #[default]
    Cached,
    /// Past the file cache, where the file system can.
    Direct,
}

type WakeFn = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct AsksInner {
    count: AtomicUsize,
    on_empty: Mutex<Vec<WakeFn>>,
}

/// Requests that want a profile now. Clones share one count.
#[derive(Clone, Default)]
pub struct Asks(Arc<AsksInner>);

impl fmt::Debug for Asks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Asks")
            .field(&self.0.count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Asks {
    /// Counts one request until the result drops.
    pub fn hold(&self) -> Ask {
        self.0.count.fetch_add(1, Ordering::AcqRel);
        Ask(self.0.clone())
    }

    /// Whether any request wants a profile.
    pub fn wanted(&self) -> bool {
        self.0.count.load(Ordering::Acquire) > 0
    }

    /// Calls `wake` each time the last active `Ask` drops.
    pub fn on_empty(&self, wake: impl Fn() + Send + Sync + 'static) {
        self.0
            .on_empty
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Arc::new(wake));
    }
}

/// One request's hold on an `Asks` count.
pub struct Ask(Arc<AsksInner>);

impl fmt::Debug for Ask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Ask")
            .field(&self.0.count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Drop for Ask {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            let hooks = self
                .0
                .on_empty
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            for wake in hooks {
                wake();
            }
        }
    }
}

/// Display-carveout lender settings for `spark-attic`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attic {
    pub socket: PathBuf,
    pub key_file: PathBuf,
}

impl Attic {
    pub const DEFAULT_SOCKET: &str = "/run/spark-attic/attic.sock";
    pub const DEFAULT_KEY_FILE: &str = "/etc/spark-attic/key";
}

impl Default for Attic {
    fn default() -> Self {
        Self {
            socket: PathBuf::from(Self::DEFAULT_SOCKET),
            key_file: PathBuf::from(Self::DEFAULT_KEY_FILE),
        }
    }
}

/// Settings for the planner, the runners and the tools.
#[derive(Clone, Debug)]
pub struct Settings {
    pub backend: Pick,
    /// Lanes per SIMD group the host executor simulates.
    pub host_lanes: usize,
    /// Where calibrate keeps cost samples. `None` stores nothing.
    pub costs: Option<PathBuf>,
    /// Shared models and caches for stickler. `None` uses the checkout.
    pub check: Option<PathBuf>,
    /// A JSON file of weight-name rules, applied last.
    pub names: Option<PathBuf>,
    /// Where fetched recipes and their weights are kept.
    pub recipes: Option<PathBuf>,
    /// The GPU whose tile rounding the host imitates.
    pub emulate: Option<Backend>,
    /// RDMA port specs the loopback tests run over.
    pub verbs: Option<String>,
    /// The plan size the runners refuse past.
    pub memory: Memory,
    /// Weights may page from the host or load by stage.
    pub weight_streaming: bool,
    pub reads: Reads,
    /// The Hugging Face hub cache.
    pub hub: Option<PathBuf>,
    /// The directory for short-lived files.
    pub temp: Option<PathBuf>,
    /// The Hugging Face access token, for gated repositories.
    pub hf_token: Option<String>,
    /// Launches per profile report. `None` times nothing.
    pub profile: Option<u64>,
    /// Profiles only while requests ask, then reports.
    pub asks: Option<Asks>,
    /// Kernel files that replace planned kernels, named by entry.
    pub kernels: Option<PathBuf>,
    /// Directory for cached PTX and HSACO images.
    pub kernel_cache: Option<PathBuf>,
    /// Extra flags passed to NVRTC or hiprtc.
    pub nvrtc: Vec<String>,
    /// File path for wasm site and reply logs.
    pub dump_sites: Option<PathBuf>,
    /// GPU device ordinal. `None` picks device 0.
    pub device: Option<usize>,
    /// Display-carveout lender settings. `None` turns borrowing off.
    pub attic: Option<Attic>,
    /// Minimum batch rows for split-batch comm/compute overlap. `0` disables.
    pub split_rows: usize,
    /// Minimum batch rows for row ownership across ranks. `0` disables.
    pub own_rows: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: Pick::Host,
            host_lanes: 32,
            costs: None,
            check: None,
            names: None,
            recipes: None,
            emulate: None,
            verbs: None,
            memory: Memory::Device,
            weight_streaming: false,
            reads: Reads::Cached,
            hub: None,
            temp: None,
            hf_token: None,
            profile: None,
            asks: None,
            kernels: None,
            kernel_cache: None,
            nvrtc: Vec::new(),
            dump_sites: None,
            device: None,
            attic: None,
            split_rows: 4096,
            own_rows: 16,
        }
    }
}

impl Settings {
    /// The GPU plans run on, or `None` for the host.
    pub fn gpu(&self) -> Option<Backend> {
        match self.backend {
            Pick::Gpu(b) => Some(b),
            Pick::Host | Pick::Cpu => None,
        }
    }
}
