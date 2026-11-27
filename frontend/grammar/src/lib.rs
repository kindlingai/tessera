//! Grammars that limit which tokens a decode step may pick.

#[cfg(feature = "llguidance")]
mod guidance;
#[cfg(all(test, feature = "llguidance"))]
mod guidance_test;

#[cfg(feature = "llguidance")]
pub use guidance::Vocab;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
