//! Runs lowered plans on the host or a GPU.

mod cpu;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod device;
mod held;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod kernels;
mod loopback;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod marks;
#[cfg(all(test, any(target_os = "macos", feature = "cuda", feature = "rocm")))]
mod marks_test;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod overrides;
#[cfg(all(test, any(target_os = "macos", feature = "cuda", feature = "rocm")))]
mod overrides_test;
mod profile;
mod rank;
mod run;
pub mod testing;

pub use held::Held;
pub use loopback::Loopback;
pub use profile::{Marks, Profile, Region, Spent};
pub use rank::Rank;
pub use run::{Base, Outcome, Reload, Run};

/// A failure while loading or running a plan.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Lower(#[from] sensai_lower::Error),
    #[error("gpu: {0}")]
    Gpu(#[from] sensai_gpukit::Error),
    #[error("memory: {0}")]
    Memory(#[from] sensai_mem::Error),
    #[error("comm: {0}")]
    Comm(#[from] sensai_engine_comm::Error),
    #[error("{0} is hand-written")]
    Handwritten(&'static str),
    #[error("{arg:?} as {param:?} on the host")]
    HostArg { arg: Arg, param: Param },
    #[error("jit: {0}")]
    Jit(&'static str),
    #[error("plans out of rank order")]
    RankOrder,
    #[error("ranks disagree on an output")]
    OutputsDisagree,
    #[error("rank {plan} of {plans} on rank {rank} of {ranks}")]
    RankMismatch {
        plan: u32,
        plans: u32,
        rank: u32,
        ranks: u32,
    },
    #[error("a sum of {0} floats does not fit the device")]
    SumTooLong(usize),
    #[error("a plan over several zones")]
    Zones,
    #[error("a fixed kernel on this runner")]
    FixedKernel,
    #[error("kernel file {0}: {1}")]
    KernelFile(std::path::PathBuf, std::io::Error),
    #[error("weight {weight} does not fit {form}")]
    WeightSize { weight: u32, form: &'static str },
    #[error("weight {0} has the wrong length")]
    WeightLen(u32),
    #[error("weight {weight} as {form} split on axis {axis}")]
    WeightSplit {
        weight: u32,
        form: &'static str,
        axis: u32,
    },
    #[error("weight {weight} as {form} cannot fill a {into} slot")]
    WeightInto {
        weight: u32,
        form: &'static str,
        into: &'static str,
    },
    #[error("weight {weight} as {form}: {why}")]
    WeightParts {
        weight: u32,
        form: &'static str,
        why: &'static str,
    },
    #[error(
        "the plan needs {planned} bytes, over the {budget} available. \
         TESSERA_MEMORY sets the budget in bytes, or `off`"
    )]
    OverBudget { planned: u64, budget: u64 },
    #[error("the input list does not match the plan")]
    InputCount,
    #[error("input {0} does not match the plan")]
    InputShape(usize),
}

use sensai_codegen::Precision;
use sensai_codegen::ir::Param;
use sensai_engine_comm::Pinning;
use sensai_engine_settings::Backend;
use sensai_engine_settings::Settings;
use sensai_lower::{Arg, Plan};
use std::sync::Arc;

/// This build's GPU, when it can run kernels here.
pub fn gpu() -> Option<Backend> {
    sensai_gpukit::backend::picked()
}

/// How `backend`'s tiles round.
pub fn precision(backend: Backend) -> Precision {
    match backend {
        Backend::Metal => Precision::METAL,
        Backend::Cuda => Precision::CUDA,
        Backend::Rocm => Precision::HIP,
    }
}

/// Pinned memory on the GPU `settings` picks, if any.
pub fn pinning(settings: &Settings) -> Option<Arc<dyn Pinning>> {
    match settings.gpu() {
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => {
            let ctx = sensai_gpukit::cuda::cached_context().ok()?;
            Some(Arc::new(device::Pin(ctx)))
        }
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => {
            let ctx = sensai_gpukit::rocm::cached_context().ok()?;
            Some(Arc::new(device::Pin(ctx)))
        }
        _ => None,
    }
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    match settings.gpu() {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(Box::new(
            device::Runner::<sensai_gpukit::metal::Context>::new(plan, settings)?,
        )),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(Box::new(
            device::Runner::<sensai_gpukit::cuda::Context>::new(plan, settings)?,
        )),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(Box::new(
            device::Runner::<sensai_gpukit::rocm::Context>::new(plan, settings)?,
        )),
        _ => Ok(Box::new(cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, precision),
            settings.memory,
        )?)),
    }
}
