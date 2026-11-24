//! Runs lowered plans on the host or a GPU.

mod cache;
mod cpu;
mod device;
mod grid;
mod held;
mod kernels;
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
mod rank;
mod run;
pub mod testing;

pub use grid::{FormMeta, Grid, Item, ItemKind, KeptDesc, Layout};
pub use held::{Held, Holds};
pub use loopback::Loopback;
pub use profile::{Marks, Profile, Ranks, Region, Spent};
pub use rank::{Rank, Sum};
pub use run::{Base, Outcome, Reload, Run};

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
    tessera_gpukit::backend::picked()
}

/// Applies cache, device and compiler settings to the GPU runtime.
pub fn configure(settings: &Settings) {
    if let Some(ordinal) = settings.device {
        tessera_gpukit::device::select(ordinal);
    }
    tessera_gpukit::cache::configure(settings);
}

/// Picks the GPU ordinal opened on first context use.
pub fn select(ordinal: usize) {
    tessera_gpukit::device::select(ordinal);
}

/// Pinned memory on the GPU `settings` picks, if any.
pub fn pinning(settings: &Settings) -> Option<Arc<dyn Pinning>> {
    configure(settings);
    if let Ok(pins) =
        tessera_gpukit::backend::dispatch(settings.gpu().filter(|b| b.pins()), device::Pins)
    {
        return pins;
    }
    None
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    configure(settings);
    let open = tessera_gpukit::backend::dispatch(settings.gpu(), device::Open(plan, settings));
    let mut runner: Box<dyn Run> = match open {
        Ok(run) => run?,
        Err(device::Open(plan, _)) => Box::new(cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, Precision::of),
            settings.memory,
        )?),
    };
    runner.base_mut().reads = settings.reads;
    Ok(runner)
}
