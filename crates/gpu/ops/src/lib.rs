//! Core of the portable GPU ops: backend, registration, test helpers.

pub mod backend;
pub mod kernel;
pub mod testing;

pub use gpukit::Error;
