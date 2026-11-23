//! Model descriptions, traced into graphs.

pub mod clip;
pub mod decoder;
pub mod dflash;
mod dflash_shard;
pub mod eg2;
pub mod flux2;
pub mod gemma4;
pub mod keys;
pub mod mla;
pub mod nanogpt;
pub mod qwen35;
mod shared;
pub mod small;
pub mod sprite;
pub mod tagger;
pub mod vae;

pub use shared::{Attend, Dense, FEW, Rotated, Side, Stack, Steps};
