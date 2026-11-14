//! Reads weights from safetensors files.

pub mod exlr8;
mod file;
mod gguf;

pub use file::{Error, File};
pub use gguf::{Gguf, GgufType, Meta};

#[cfg(test)]
mod file_test;
#[cfg(test)]
mod gguf_test;
