//! Knobs the front end sets once and the engine reads.

use std::path::PathBuf;

/// A GPU that runs kernels. gpukit has its own copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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

/// How many bytes one plan may hold.
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
