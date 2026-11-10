//! Reads weights from safetensors files.

mod file;

pub use file::{Error, File};

#[cfg(test)]
mod file_test;
