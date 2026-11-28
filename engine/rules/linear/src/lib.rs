//! Generated matmul kernels, registered as lowering rules.

mod dense;
mod lanes;
mod picked;
mod product;
mod stacked;
mod thin;
mod tiled;
mod turned;

pub use dense::{Dense, launch_args};
pub use product::{Product, Variant, place};
pub use stacked::blocks;
pub use tiled::{BF16, PARTS, SCHEDULES, Schedule, Tiled};
pub use turned::Turned;
