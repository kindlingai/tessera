//! Generated matmul kernels, registered as lowering rules.

mod lanes;
mod picked;
mod product;
mod stacked;
mod thin;
mod tiled;
mod turned;

pub use product::Product;
pub use tiled::{SCHEDULES, Schedule, Tiled};
pub use turned::Turned;
