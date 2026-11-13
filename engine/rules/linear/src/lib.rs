//! Generated matmul kernels, registered as lowering rules.

mod lanes;
mod tiled;

pub use lanes::Lanes;
pub use tiled::{SCHEDULES, Schedule, Tiled};
