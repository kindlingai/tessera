//! GPU compute mechanism; callers supply all kernel and cache policy.

mod error;
pub use error::Error;

pub mod includes;

mod kernel;

pub mod testing;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
