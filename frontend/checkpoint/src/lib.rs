//! Reads weights from safetensors, GGUF and PyTorch files.

mod disk;
pub mod exlr8;
mod exlr8_experts;
#[cfg(test)]
mod exlr8_experts_test;
#[cfg(test)]
mod exlr8_test;
mod file;
#[cfg(test)]
mod file_test;
#[cfg(test)]
mod gemma4_test;
mod gguf;
#[cfg(test)]
mod gguf_test;
mod gguf_write;
mod modelopt;
#[cfg(test)]
mod modelopt_test;
mod offer;
#[cfg(test)]
mod offer_test;
mod pickle;
mod torch;
#[cfg(test)]
mod torch_test;
mod zip;

pub use file::{Error, File, Shards};
pub use gguf::{Gguf, GgufType, Meta};
pub use gguf_write::{GgufTensor, write_gguf};
pub use offer::{Offer, Slice, Stored};
pub use torch::Torch;

use std::path::Path;

/// Opens a checkpoint file or directory as an `Offer`.
pub fn open(path: &Path, manifest: Option<&str>) -> Result<Box<dyn Offer>, Error> {
    if path.is_file() {
        return match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => {
                let shards = Shards::from_files(vec![File::open(path)?]);
                if modelopt::has_quantized(&shards) {
                    Ok(Box::new(modelopt::ModelOpt::new(shards)))
                } else {
                    Ok(Box::new(shards))
                }
            }
            Some("pt" | "pth" | "bin") => Ok(Box::new(Torch::open(path)?)),
            _ => Ok(Box::new(Gguf::open(path)?)),
        };
    }
    if exlr8::is_release(path) {
        return Ok(Box::new(exlr8::open_manifest(path, manifest)?));
    }
    let shards = Shards::open(path)?;
    if modelopt::has_quantized(&shards) {
        return Ok(Box::new(modelopt::ModelOpt::new(shards)));
    }
    Ok(Box::new(shards))
}
