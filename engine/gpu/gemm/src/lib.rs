//! Tiled f32 GEMM, `C = alpha * op(A) @ op(B) + beta * C`.

mod cpu;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod device;
#[cfg(feature = "cuda")]
pub use device::cuda;
#[cfg(feature = "rocm")]
pub use device::rocm;
#[cfg(test)]
mod fixtures;
#[cfg(target_os = "macos")]
mod metal;
pub mod problem;
#[cfg(test)]
mod testing;

pub use gpukit::Error;
pub use problem::{Call, Problem, out_len};

/// Block tile. Must match the #defines in gemm.metal / gemm.cu.
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;
pub const THREADS: u32 = 256;

pub const ENTRY: &str = "gemm_f32";
/// The same tiles, with B stored as bf16.
pub const ENTRY_BF16: &str = "gemm_bf16";

/// Kernel argument bytes, for callers that launch it themselves.
pub fn abi_params(p: &Problem) -> Vec<u8> {
    let abi = crate::problem::AbiParams::from(*p);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            std::ptr::from_ref(&abi).cast::<u8>(),
            size_of::<crate::problem::AbiParams>(),
        )
    };
    bytes.to_vec()
}
pub const METAL: &str = include_str!("gemm.metal");

/// CUDA C++ source. NVRTC or hiprtc compiles it on use.
pub const CUDA: &str = include_str!("gemm.cu");

/// The CPU reference that GPU results are checked against.
pub fn cpu(call: &Call) -> Vec<f32> {
    cpu::cpu(call)
}

/// Run on the backend `gpukit::backend` picks.
pub fn gpu(call: &Call) -> Result<Vec<f32>, Error> {
    let _ = call;
    match gpukit::backend::available() {
        #[cfg(target_os = "macos")]
        Some(gpukit::backend::Backend::Metal) => metal::gpu(call),
        #[cfg(feature = "cuda")]
        Some(gpukit::backend::Backend::Cuda) => cuda::gpu(call),
        #[cfg(feature = "rocm")]
        Some(gpukit::backend::Backend::Rocm) => rocm::gpu(call),
        _ => Err(Error::Gpu(
            "no GPU backend enabled (build with --features cuda or rocm)",
        )),
    }
}

#[cfg(test)]
mod gemm_test;
