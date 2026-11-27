//! Buffers in zones, typed views onto them, and storage forms.

#![allow(
    clippy::len_without_is_empty,
    reason = "the workspace dead code check removes unused is_empty"
)]

pub mod block;
mod buffer;
pub mod form;
mod layout;
pub mod ops;
mod tensor;
pub mod trellis;
mod view;

pub use buffer::{Backing, Buffer, Host};
pub use layout::Layout;
pub use tensor::Tensor;
pub use tessera_graph::DType;
pub use view::{View, ViewMut, copy};

use std::fmt;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A view and its use disagree on type or size.
    #[error("mismatch: {0}")]
    Mismatch(String),
    /// The backend refused a copy.
    #[error("device: {0}")]
    Device(String),
}

pub(crate) fn mismatch<T>(what: impl fmt::Display) -> Result<T, Error> {
    Err(Error::Mismatch(what.to_string()))
}

#[cfg(test)]
mod block_test;
#[cfg(test)]
mod mem_test;
#[cfg(test)]
mod trellis_test;
