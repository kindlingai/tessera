//! Generated matmul kernels, registered as lowering rules.

mod dense;
mod lanes;
mod picked;
mod product;
mod stacked;
mod thin;
mod tiled;
mod trellis_metal;
mod turned;

pub use dense::{Dense, launch_args};
pub use product::Product;
pub use tessera_mem::trellis::parse_form;
pub use tiled::{BF16, PARTS, SCHEDULES, Schedule, Tiled};
pub use turned::Turned;
