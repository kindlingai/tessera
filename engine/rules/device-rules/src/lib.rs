//! Device CUDA rules for attention and calibration.

mod attend;
pub mod calib;
#[cfg(test)]
mod calib_test;

use tessera_device::{Backend, flatten_for};

fn flatten_cuda(name: &str, kernel: &str) -> &'static str {
    flatten_for(Backend::Cuda, name, kernel)
}
