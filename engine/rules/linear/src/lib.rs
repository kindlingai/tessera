//! Generated matmul kernels, registered as lowering rules.

mod dense;
mod lanes;
mod mm_metal;
mod lanes_metal;
mod picked;
mod product;
mod stacked;
mod thin;
mod tiled;
mod tiled_metal;
mod trellis_metal;
mod turned;

pub use dense::{Dense, launch_args};
pub use product::Product;
pub use tiled::{BF16, PARTS, SCHEDULES, Schedule, Tiled};
pub use turned::Turned;
