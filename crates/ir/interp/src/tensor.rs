//! The interpreter's errors. Its tensors come from `mem`.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The graph needs something this interpreter lacks.
    Unsupported(String),
    /// Values disagree with the graph's types.
    Mismatch(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Mismatch(m) => write!(f, "mismatch: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// A view that did not fit is a mismatch.
impl From<mem::Error> for Error {
    fn from(e: mem::Error) -> Self {
        Error::Mismatch(e.to_string())
    }
}
