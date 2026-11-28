//! GPU compute mechanism. Callers supply all kernel and cache policy.

mod error;
pub use ::tessera_platform::{Limits, Platform, platform};
pub use error::Error;

pub mod attic;
#[cfg(test)]
mod attic_test;
pub mod backend;
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub mod cache;
#[cfg(test)]
mod cache_test;
pub mod testing;

#[cfg(target_os = "macos")]
pub mod metal;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod pending;
#[cfg(test)]
mod pending_test;

#[cfg(any(feature = "cuda", feature = "rocm"))]
mod api;
pub mod device;
mod dl;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod gate;
pub mod launch;
#[cfg_attr(
    not(all(target_os = "linux", any(feature = "cuda", feature = "rocm"))),
    allow(dead_code)
)]
mod numa;
#[cfg(test)]
mod numa_test;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod rtc;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;
