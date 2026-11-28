//! Device CUDA rules for attention and calibration.

mod attend;
pub mod calib;
#[cfg(test)]
mod calib_test;
mod packed;

pub use packed::{packed_cuda_source, packed_rocm_source};

use tessera_device::{Backend, flatten_for};

fn flatten_cuda(name: &str, kernel: &str) -> &'static str {
    flatten_for(Backend::Cuda, name, kernel)
}
