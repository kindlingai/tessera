//! Reads weights from safetensors, GGUF and PyTorch files.

pub mod exlr8;
mod file;
mod gguf;
mod gguf_write;
mod offer;
mod pickle;
mod torch;
mod zip;

pub use file::{Error, File, Shards};
pub use gguf::{Gguf, GgufType, Meta};
pub use gguf_write::{GgufTensor, write_gguf};
pub use offer::{Offer, Slice, Stored};
pub use torch::Torch;

#[cfg(test)]
mod exlr8_test;
#[cfg(test)]
mod file_test;
#[cfg(test)]
mod gguf_test;
#[cfg(test)]
mod offer_test;
#[cfg(test)]
mod torch_test;
