//! Generated matmul kernels, registered as lowering rules.

mod lanes;
mod product;
mod thin;
mod tiled;

pub use lanes::Lanes;
pub use product::Product;
pub use thin::{THIN_ROWS, Thin};
pub use tiled::{SCHEDULES, Schedule, Tiled};
