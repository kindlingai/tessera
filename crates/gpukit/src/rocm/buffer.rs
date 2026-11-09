//! Device memory and a size-bucketed pool.

use super::context::Context;
use super::hip::{HipDeviceptr, MEMCPY_DEVICE_TO_HOST, MEMCPY_HOST_TO_DEVICE};
use crate::Error;

/// A device allocation, freed on drop.
pub struct DeviceBuffer {
    ctx: Context,
    ptr: HipDeviceptr,
    size: usize,
}

impl DeviceBuffer {
    /// Allocate one device buffer outside any pool.
    pub fn alloc(ctx: &Context, size: usize) -> Result<Self, Error> {
        ctx.set_current()?;
        let mut ptr: HipDeviceptr = 0;
        ctx.runtime().check(
            unsafe { (ctx.runtime().hip_malloc)(&mut ptr, size) },
            "hipMalloc",
        )?;
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
            size,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Raw device address, for binding as a kernel argument.
    pub fn device_ptr(&self) -> HipDeviceptr {
        self.ptr
    }

    fn assert_fits(&self, len: usize) {
        assert!(
            len <= self.size,
            "device buffer copy of {len} bytes exceeds allocation of {}",
            self.size
        );
    }

    /// Host to device. Synchronous.
    pub fn write_bytes(&self, bytes: &[u8]) -> Result<(), Error> {
        self.assert_fits(bytes.len());
        self.ctx.set_current()?;
        let rt = self.ctx.runtime();
        rt.check(
            unsafe {
                (rt.hip_memcpy)(
                    self.ptr as *mut _,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy to device",
        )
    }

    /// Device to host. Synchronous.
    pub fn read_bytes(&self, bytes: &mut [u8]) -> Result<(), Error> {
        self.assert_fits(bytes.len());
        self.ctx.set_current()?;
        let rt = self.ctx.runtime();
        rt.check(
            unsafe {
                (rt.hip_memcpy)(
                    bytes.as_mut_ptr().cast(),
                    self.ptr as *const _,
                    bytes.len(),
                    MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy to host",
        )
    }

    pub fn write_f32(&self, data: &[f32]) -> Result<(), Error> {
        // The byte view is exact: f32 is plain data.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.write_bytes(bytes)
    }

    pub fn read_f32(&self, out: &mut [f32]) -> Result<(), Error> {
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4) };
        self.read_bytes(bytes)
    }

    /// Zero the whole allocation.
    pub fn zero(&self) -> Result<(), Error> {
        self.ctx.set_current()?;
        self.ctx.runtime().check(
            unsafe { (self.ctx.runtime().hip_memset)(self.ptr, 0, self.size) },
            "hipMemset",
        )
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // At process exit the runtime may already be gone.
        if self.ctx.set_current().is_ok() {
            unsafe {
                (self.ctx.runtime().hip_free)(self.ptr);
            }
        }
    }
}

/// Reusable device buffers. `allocate` zeroes each one it returns.
#[derive(Default)]
pub struct BufferPool {
    free: Vec<DeviceBuffer>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate(&mut self, ctx: &Context, size: usize) -> Result<DeviceBuffer, Error> {
        let buf = match self.free.iter().position(|b| b.size >= size) {
            Some(idx) => self.free.swap_remove(idx),
            None => DeviceBuffer::alloc(ctx, size)?,
        };
        buf.zero()?;
        Ok(buf)
    }

    pub fn release(&mut self, buffer: DeviceBuffer) {
        self.free.push(buffer);
    }

    pub fn clear(&mut self) {
        self.free.clear();
    }

    pub fn trim(&mut self, max_buffers: usize) {
        self.free.truncate(max_buffers);
    }
}
