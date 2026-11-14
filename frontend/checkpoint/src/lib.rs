//! Reads weights from safetensors files.

pub mod exlr8;
mod file;
mod gguf;
pub use gguf::{Gguf, GgufType, Meta};

pub use file::{Error, File};

#[cfg(test)]
mod file_test;

#[cfg(test)]
mod gguf_test;
