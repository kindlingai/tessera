//! Knobs the front end sets once and the engine reads.

use std::path::PathBuf;

/// A GPU that runs kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, derive_more::Display)]
#[serde(rename_all = "lowercase")]
#[display(rename_all = "lowercase")]
pub enum Backend {
    Metal,
    Cuda,
    Rocm,
}

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

/// Settings for the planner, the runners and the tools.
#[derive(Clone, Debug)]
pub struct Settings {
    pub backend: Pick,
    /// Lanes per SIMD group the host executor simulates.
    pub host_lanes: usize,
    /// Where calibrate keeps cost samples. `None` stores nothing.
    pub costs: Option<PathBuf>,
    /// Shared models and caches for tessera-check. `None` uses the checkout.
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
    /// The Hugging Face hub cache.
    pub hub: Option<PathBuf>,
    /// The Hugging Face access token, for gated repositories.
    pub hf_token: Option<String>,
    /// Launches per profile report. `None` times nothing.
    pub profile: Option<u64>,
    /// Kernel files that replace planned kernels, named by entry.
    pub kernels: Option<PathBuf>,
    /// Directory for cached PTX and HSACO images.
    pub kernel_cache: Option<PathBuf>,
    /// Extra flags passed to NVRTC or hiprtc.
    pub nvrtc: Vec<String>,
    /// File path for wasm site and reply logs.
    pub dump_sites: Option<PathBuf>,
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
            hub: None,
            hf_token: None,
            profile: None,
            kernels: None,
            kernel_cache: None,
            nvrtc: Vec::new(),
            dump_sites: None,
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
