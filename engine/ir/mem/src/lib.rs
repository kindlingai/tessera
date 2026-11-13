//! Buffers in zones, and typed views onto them.

mod buffer;
mod layout;
mod tensor;
mod view;

pub use buffer::{Backing, Buffer, Host};
pub use graph::DType;
pub use layout::Layout;
pub use tensor::Tensor;
pub use view::{View, ViewMut, copy};

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A view and its use disagree on type or size.
    Mismatch(String),
    /// The backend refused a copy.
    Device(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Mismatch(m) => write!(f, "mismatch: {m}"),
            Error::Device(m) => write!(f, "device: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn mismatch<T>(what: impl fmt::Display) -> Result<T, Error> {
    Err(Error::Mismatch(what.to_string()))
}

#[cfg(test)]
mod mem_test;
