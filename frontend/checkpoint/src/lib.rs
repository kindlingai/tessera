//! Reads weights from safetensors files.

pub mod exlr8;
mod file;
mod gguf;
mod offer;

pub use file::{Error, File};
pub use gguf::{Gguf, GgufType, Meta};
pub use offer::{Offer, Slice, Stored};

#[cfg(test)]
mod file_test;
#[cfg(test)]
mod gguf_test;
