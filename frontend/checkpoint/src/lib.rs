//! Reads weights from safetensors files.

pub mod exlr8;
mod file;

pub use file::{Error, File};

#[cfg(test)]
mod file_test;
