//! Which GPU backend this build dispatches to.

use crate::device::Device;
use crate::launch::Launch;
pub use tessera_engine_settings::Backend;

/// Work generic over one backend's context type.
pub trait OnGpu {
    type Out;
    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out;
}

/// Runs `work` on the context type `gpu` names, if built.
pub fn dispatch<W: OnGpu>(gpu: Option<Backend>, work: W) -> Result<W::Out, W> {
    match gpu {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(work.on::<crate::metal::Context>()),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(work.on::<crate::cuda::Context>()),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(work.on::<crate::rocm::Context>()),
        _ => Err(work),
    }
}

/// The backend this build uses, if any.
pub fn available() -> Option<Backend> {
    if cfg!(target_os = "macos") {
        return Some(Backend::Metal);
    }
    // Runtimes load only at dispatch, unless both are built.
    match (cfg!(feature = "cuda"), cfg!(feature = "rocm")) {
        (true, true) if !cuda_ready() && rocm_ready() => Some(Backend::Rocm),
        (true, _) => Some(Backend::Cuda),
        (false, true) => Some(Backend::Rocm),
        (false, false) => None,
    }
}

/// Whether `backend` can run kernels here.
pub fn ready(backend: Backend) -> bool {
    match backend {
        // Hosted macOS runners lack a working Metal device.
        Backend::Metal => cfg!(target_os = "macos") && !ci(),
        Backend::Cuda => cuda_ready(),
        Backend::Rocm => rocm_ready(),
    }
}

/// This build's backend, when it can run kernels here.
pub fn picked() -> Option<Backend> {
    available().filter(|b| ready(*b))
}

/// Whether a CUDA driver loads on this host.
fn cuda_ready() -> bool {
    #[cfg(feature = "cuda")]
    return crate::cuda::driver::driver().is_ok();
    #[cfg(not(feature = "cuda"))]
    false
}

/// Whether a HIP runtime loads and sees a device.
fn rocm_ready() -> bool {
    #[cfg(feature = "rocm")]
    return crate::rocm::cached_context().is_ok();
    #[cfg(not(feature = "rocm"))]
    false
}

/// Parses a CUDA target into `(major, minor, suffix)`.
pub fn cuda_arch(target: &str) -> Result<(u32, u32, &'static str), crate::Error> {
    let raw = target.trim();
    if raw.is_empty() || !raw.is_ascii() {
        return Err(crate::Error::Cuda(format!(
            "invalid CUDA target {target:?}"
        )));
    }
    let stripped = raw
        .strip_prefix("compute_")
        .or_else(|| raw.strip_prefix("sm_"))
        .unwrap_or(raw);
    let (body, suffix) = match stripped.as_bytes().last().copied() {
        Some(b'a') => (&stripped[..stripped.len() - 1], "a"),
        Some(b'f') => (&stripped[..stripped.len() - 1], "f"),
        _ => (stripped, ""),
    };
    if !body.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return Err(crate::Error::Cuda(format!(
            "invalid CUDA target {target:?}"
        )));
    }
    let (major, minor) = match body.split_once('.') {
        Some((hi, lo)) => (hi.parse::<u32>(), lo.parse::<u32>()),
        None if body.len() >= 2 => {
            let (hi, lo) = body.split_at(body.len() - 1);
            (hi.parse::<u32>(), lo.parse::<u32>())
        }
        None => {
            return Err(crate::Error::Cuda(format!(
                "invalid CUDA target {target:?}"
            )));
        }
    };
    let (Ok(major), Ok(minor)) = (major, minor) else {
        return Err(crate::Error::Cuda(format!(
            "invalid CUDA target {target:?}"
        )));
    };
    if major == 0 || minor > 9 {
        return Err(crate::Error::Cuda(format!(
            "invalid CUDA target {target:?}"
        )));
    }
    // Blackwell targets require arch-specific PTX for MXFP8 MMA.
    let suffix = match suffix.is_empty() && major >= 12 {
        true => "a",
        false => suffix,
    };
    Ok((major, minor, suffix))
}

/// Whether a hosted CI runner is running this process.
#[allow(clippy::disallowed_methods, reason = "the one variable tests read")]
fn ci() -> bool {
    std::env::var_os("CI").is_some()
}
