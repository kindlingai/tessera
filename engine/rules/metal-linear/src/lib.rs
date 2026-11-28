//! Metal matmul kernels. Linking this crate registers them.

mod block_metal;
mod lanes_metal;
mod mm_metal;
mod moe_metal;
mod stacked_metal;
mod tiled_metal;
mod trellis_metal;

/// Emits only f32 values.
pub(crate) fn f32_only<K>(_: &K, form: &dyn tessera_lower::form::Load) -> bool {
    form.name() == "f32"
}
