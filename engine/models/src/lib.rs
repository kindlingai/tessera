//! Model descriptions, traced into graphs.

pub mod clip;
pub mod decoder;
pub mod dflash;
mod dflash_shard;
pub mod eg2;
pub mod flux2;
pub mod gemma4;
pub mod hy4;
pub mod inkling2;
pub mod keys;
pub mod mla;
pub mod mtp;
pub mod nanogpt;
pub mod preset;
pub mod qwen35;
pub mod qwen3_vl;
pub mod qwen4_exp;
mod shared;
pub mod sprite;
pub mod tagger;
pub mod vae;

pub use shared::{Rotated, Stack, Steps};
pub use tessera_trace::{FEW, hf};
