//! The settings tests run under: host, then a ready GPU.

use crate::{Error, Run, gpu};
use tessera_codegen::Precision;
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_gpukit::device::Device;
use tessera_gpukit::launch::Launch;
use tessera_lower::Plan;

/// A device runner on any `Device`, for tests.
pub fn device<D>(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    Ok(Box::new(crate::device::Runner::<D>::new(plan, settings)?))
}

/// A host runner that runs every tree in lockstep.
pub fn lockstep(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    let (lanes, memory) = (settings.host_lanes, settings.memory);
    let runner = crate::cpu::Runner::new(plan, lanes, Precision::EXACT, memory, false)?;
    Ok(Box::new(runner))
}

/// Settings for plans on the host.
pub fn host() -> Settings {
    Settings::default()
}

/// The host, then this build's GPU when one is ready.
pub fn runs() -> Vec<Settings> {
    tessera_gpukit::testing::require_gpu();
    let mut runs = vec![host()];
    if let Some(b) = gpu() {
        runs.push(Settings {
            backend: Pick::Gpu(b),
            ..host()
        });
    }
    runs
}

/// `runs`, then the host imitating each GPU's rounding.
pub fn emulations() -> Vec<Settings> {
    let imitated = Backend::ALL.iter().map(|b| Settings {
        emulate: Some(*b),
        ..host()
    });
    runs().into_iter().chain(imitated).collect()
}
