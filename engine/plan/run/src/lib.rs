//! Runs lowered plans on the host or a GPU.

mod cpu;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod device;
mod held;
#[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
mod kernels;
#[cfg(target_os = "macos")]
mod metal;
mod run;

pub use held::Held;
pub use run::{Base, Run};

use codegen::Precision;
use gpukit::Settings;
use gpukit::backend::Backend;
use lower::{Error, Plan};

/// How `backend`'s tiles round.
pub fn precision(backend: Backend) -> Precision {
    match backend {
        Backend::Metal => Precision::METAL,
        Backend::Cuda => Precision::CUDA,
        Backend::Rocm => Precision::HIP,
    }
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    match settings.gpu() {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(Box::new(metal::Runner::new(plan)?)),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(Box::new(device::cuda::Runner::new(plan)?)),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(Box::new(device::rocm::Runner::new(plan)?)),
        _ => Ok(Box::new(cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, precision),
        )?)),
    }
}
