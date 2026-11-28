//! Generated matmul kernels, registered as lowering rules.

mod block_metal;
mod dense;
mod lanes;
mod lanes_metal;
mod mm_metal;
mod moe_metal;
mod picked;
mod product;
mod stacked;
mod stacked_fused;
mod thin;
mod tiled;
mod tiled_metal;
mod trellis_metal;
mod turned;

pub use dense::{Dense, launch_args};
pub use product::Product;
pub use tiled::{BF16, PARTS, SCHEDULES, Schedule, Tiled};
pub use turned::Turned;

/// Emits only f32 values.
pub(crate) fn f32_only<K>(_: &K, form: &dyn tessera_lower::form::Load) -> bool {
    form.name() == "f32"
}
