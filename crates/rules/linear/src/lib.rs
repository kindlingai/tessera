//! Generated matmul kernels, registered as lowering rules.

mod lanes;
mod tiled;

pub use lanes::{After, Lanes};
pub use tiled::{Schedule, Tiled};
