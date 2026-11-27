//! Runs lowered plans on the host or a GPU.

mod cache;
pub mod check;
mod cpu;
mod device;
mod grid;
mod held;
mod kernels;
mod lanes;
#[cfg(test)]
mod lanes_test;
mod loopback;
pub mod marks;
#[cfg(test)]
mod marks_test;
mod overrides;
#[cfg(test)]
mod overrides_test;
mod paged;
#[cfg(test)]
mod paged_test;
mod profile;
#[cfg(test)]
mod profile_test;
mod promote;
#[cfg(test)]
mod promote_test;
mod rank;
mod run;
pub mod sizer;
pub mod testing;

pub use grid::{FormMeta, Grid, Item, ItemKind, KeptDesc, Layout};
pub use held::{Held, Holds};
pub use loopback::Loopback;
pub use overrides::{Flattener, set_flattener, text as kernel_text};
pub use profile::{Marks, Profile, Ranks, Region, Spent};
pub use rank::{Rank, Sum, sum_wire};
pub use run::{Base, Outcome, Reload, Run};
pub use tessera_gpukit::device::KernelAttrs;

/// A failure while loading or running a plan.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Lower(#[from] tessera_lower::Error),
    #[error("gpu: {0}")]
    Gpu(#[from] tessera_gpukit::Error),
    #[error("step {0} ({1}): {2}")]
    Launch(usize, String, Box<Error>),
    #[error("memory: {0}")]
    Memory(#[from] tessera_mem::Error),
    #[error("comm: {0}")]
    Comm(#[from] tessera_engine_comm::Error),
    #[error("{0} is hand-written")]
    Handwritten(&'static str),
    #[error("{arg:?} as {param:?} on the host")]
    HostArg { arg: Box<Arg>, param: Box<Param> },
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
    #[error("sums: {0}")]
    Sums(Box<dyn std::error::Error + Send + Sync>),
    #[error("grid: {0}")]
    Grid(Box<dyn std::error::Error + Send + Sync>),
    #[error("step {0} runs in a persistent grid, and none is given")]
    Gridless(usize),
    #[error("this runner has no persistent grids")]
    NoGrids,
    #[error("step {0} has no device address")]
    Unaddressed(usize),
    #[error("a sum of {0} floats does not fit the device")]
    SumTooLong(usize),
    #[error("the sum transport does not carry {0:?} parts")]
    Wire(tessera_mem::DType),
    #[error("a plan over several zones")]
    Zones,
    #[error("kernel file {0}: {1}")]
    KernelFile(std::path::PathBuf, std::io::Error),
    #[error("weight {weight} does not fit {form}")]
    WeightSize { weight: u32, form: &'static str },
    #[error("weight {0}: {1}")]
    WeightRead(u32, std::io::Error),
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
    #[error(
        "the plan needs {planned} bytes, over the {budget} available. \
         --allow-weight-streaming pages expert weights from host memory"
    )]
    Streaming { planned: u64, budget: u64 },
    #[error("a sizing runner runs nothing")]
    Sizing,
    #[error("runner state is inconsistent after an earlier failure")]
    Inconsistent,
    #[error("the input list does not match the plan")]
    InputCount,
    #[error("input {0} does not match the plan")]
    InputShape(usize),
}

use std::sync::Arc;
use tessera_codegen::Precision;
use tessera_codegen::ir::Param;
use tessera_engine_comm::Pinning;
use tessera_engine_settings::Backend;
use tessera_engine_settings::Settings;
use tessera_lower::{Arg, Plan};

/// This build's GPU, when it can run kernels here.
pub fn gpu() -> Option<Backend> {
    check::backend().or_else(tessera_gpukit::backend::picked)
}

/// Applies cache, device and compiler settings to the GPU runtime.
pub fn configure(settings: &Settings) {
    if let Some(ordinal) = settings.device {
        tessera_gpukit::device::select(ordinal);
    }
    tessera_gpukit::cache::configure(settings);
    tessera_gpukit::attic::configure(settings);
}

/// Picks the GPU ordinal opened on first context use.
pub fn select(ordinal: usize) {
    tessera_gpukit::device::select(ordinal);
}

/// Pinned memory on the GPU `settings` picks, if any.
pub fn pinning(settings: &Settings) -> Option<Arc<dyn Pinning>> {
    configure(settings);
    if let Ok(pins) = tessera_gpukit::backend::dispatch(settings.gpu().filter(|b| b.pins()), Pins) {
        return pins;
    }
    None
}

struct Pins;
impl tessera_gpukit::backend::OnGpu for Pins {
    type Out = Option<Arc<dyn Pinning>>;
    fn on<D>(self) -> Self::Out
    where
        D: tessera_gpukit::device::Device
            + tessera_gpukit::launch::Launch<Kernel = <D as tessera_gpukit::device::Device>::Kernel>,
    {
        Some(Arc::new(Pin(
            tessera_gpukit::device::cached_context::<D>().ok()?
        )))
    }
}

struct Asked;
impl tessera_gpukit::backend::OnGpu for Asked {
    type Out = Option<tessera_gpukit::Platform>;
    fn on<D>(self) -> Self::Out
    where
        D: tessera_gpukit::device::Device
            + tessera_gpukit::launch::Launch<Kernel = <D as tessera_gpukit::device::Device>::Kernel>,
    {
        Some(
            tessera_gpukit::device::cached_context::<D>()
                .ok()?
                .platform(),
        )
    }
}

struct Free;
impl tessera_gpukit::backend::OnGpu for Free {
    type Out = Option<u64>;
    fn on<D>(self) -> Self::Out
    where
        D: tessera_gpukit::device::Device
            + tessera_gpukit::launch::Launch<Kernel = <D as tessera_gpukit::device::Device>::Kernel>,
    {
        tessera_gpukit::device::cached_context::<D>()
            .ok()?
            .free_memory()
    }
}

/// The platform a runner for `settings` would report.
pub(crate) fn asked(settings: &Settings) -> Option<tessera_gpukit::Platform> {
    if let Some(platform) = check::platform() {
        return Some(platform);
    }
    configure(settings);
    tessera_gpukit::backend::dispatch(settings.gpu(), Asked).ok()?
}

/// Free bytes on the GPU `settings` picks, if known.
pub(crate) fn free(settings: &Settings) -> Option<u64> {
    configure(settings);
    tessera_gpukit::backend::dispatch(settings.gpu(), Free).ok()?
}

tessera_lower::rule::scatter! {
    #[scatter(tessera_lower::ASKERS)]
    static ASK: tessera_lower::Ask = asked;
}

struct Pin<D: tessera_gpukit::device::Device>(&'static D);
impl<D: tessera_gpukit::device::Device> Pinning for Pin<D> {
    fn pin(&self, len: usize) -> Result<tessera_engine_comm::Pinned, tessera_engine_comm::Error> {
        let (host, device) = self
            .0
            .pin(len)
            .map_err(|e| tessera_engine_comm::Error::Link(e.to_string()))?;
        Ok(tessera_engine_comm::Pinned {
            host: host as *mut u8,
            device,
            len,
        })
    }

    fn unpin(&self, pinned: tessera_engine_comm::Pinned) {
        self.0.unpin(pinned.host as u64);
    }
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    configure(settings);
    if let Some(checker) = check::open(&plan) {
        return Ok(checker);
    }
    if let Some(sizer) = sizer::Sizer::open(&plan, settings) {
        return Ok(sizer);
    }
    let open = tessera_gpukit::backend::dispatch(settings.gpu(), device::Open(plan, settings));
    let mut runner: Box<dyn Run> = match open {
        Ok(run) => run?,
        Err(device::Open(plan, _)) => Box::new(cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, Precision::of),
            settings.memory,
            true,
        )?),
    };
    runner.base_mut().reads = settings.reads;
    Ok(runner)
}
