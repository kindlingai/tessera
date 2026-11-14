//! One-shot dispatch helpers: encode, dispatch, and wait on one kernel.

use crate::Error;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLSize,
};

/// Bind `value`'s bytes as an argument. `T` must be `#[repr(C)]`.
pub fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::from_ref(value).cast(),
            std::mem::size_of_val(value),
            index,
        );
    }
}

pub fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// One threadgroup of `tg_width` threads per (x, y).
pub fn dispatch_grid(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
    height: usize,
    tg_width: usize,
    encode: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> Result<(), Error> {
    let cmd = queue.commandBuffer().ok_or(Error::Gpu("command buffer"))?;
    let enc = cmd.computeCommandEncoder().ok_or(Error::Gpu("encoder"))?;
    enc.setComputePipelineState(pipeline);
    encode(&enc);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width,
            height,
            depth: 1,
        },
        MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}
