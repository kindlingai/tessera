//! Quantized weight formats: byte layouts, decoders and CPU oracles.

pub mod bf16;
pub mod block;
pub mod fp4;
#[cfg(test)]
mod fp4_test;
pub mod layout;
pub mod nvfp4;

/// A weight storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Raw,
    Q4,
    Q6,
    Q8,
    Nvfp4,
}
