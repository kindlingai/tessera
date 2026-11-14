//! Knobs the front end sets once and passes down.

use crate::backend::{self, Backend};
use std::path::PathBuf;

/// Where plans run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// This build's GPU if one is up, else the host.
    Auto,
    Host,
    Gpu(Backend),
}

/// Settings for the planner, the runners and the tools.
#[derive(Clone, Debug)]
pub struct Settings {
    pub backend: Pick,
    /// Lanes per SIMD group the host executor simulates.
    pub host_lanes: usize,
    /// Where calibrate keeps cost samples. `None` stores nothing.
    pub costs: Option<PathBuf>,
    /// The Qwen3 reference files the parity test reads.
    pub qwen3: Option<PathBuf>,
    /// The EmbeddingGemma 2 reference files the parity test reads.
    pub eg2: Option<PathBuf>,
    /// The spaCy pipeline and tokens the parity test reads.
    pub spacy: Option<PathBuf>,
    /// A JSON file of weight-name rules, applied last.
    pub names: Option<PathBuf>,
    /// The GPU whose tile rounding the host imitates.
    pub emulate: Option<Backend>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            backend: Pick::Auto,
            host_lanes: 32,
            costs: None,
            qwen3: None,
            eg2: None,
            spacy: None,
            names: None,
            emulate: None,
        }
    }
}

/// Whether `backend` can run kernels here.
fn ready(backend: Backend) -> bool {
    match backend {
        Backend::Metal => cfg!(target_os = "macos"),
        Backend::Cuda => crate::testing::cuda_ready(),
        Backend::Rocm => crate::testing::rocm_ready(),
    }
}

impl Settings {
    /// The GPU plans run on, or `None` for the host.
    pub fn gpu(&self) -> Option<Backend> {
        match self.backend {
            Pick::Host => None,
            Pick::Gpu(b) => Some(b),
            Pick::Auto => backend::available().filter(|b| ready(*b)),
        }
    }
}
