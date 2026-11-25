//! Layer-streaming BF16 Hessian calibration generator.

use tessera_rules as _;

mod accum;
pub mod args;
pub mod driver;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod parity;
mod state;
