//! Tiled f32 GEMM, `C = alpha * op(A) @ op(B) + beta * C`.

mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod fixtures;
pub mod format;
#[cfg(target_os = "macos")]
mod metal;
pub mod problem;
pub mod testing;

pub use gpukit::Error;
pub use problem::{Call, Problem, out_len};

/// Block tile. Must match the #defines in gemm.metal / gemm.cu.
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;
pub const THREADS: u32 = 256;

pub const ENTRY: &str = "gemm_f32";

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

/// CUDA C++ source, compiled by NVRTC on first dispatch.
pub const CUDA: &str = include_str!("gemm.cu");

/// The CPU reference that GPU results are checked against.
pub fn cpu(call: &Call) -> Vec<f32> {
    cpu::cpu(call)
}

/// Run on this build's GPU backend.
pub fn gpu(call: &Call) -> Result<Vec<f32>, Error> {
    #[cfg(target_os = "macos")]
    {
        metal::gpu(call)
    }
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    {
        cuda::gpu(call)
    }
    #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
    {
        let _ = call;
        Err(Error::Gpu(
            "no GPU backend enabled (build with --features cuda on a CUDA host)",
        ))
    }
}

#[cfg(test)]
mod gemm_test;
