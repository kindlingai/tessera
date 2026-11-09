//! One-shot launch helpers and the kernel-argument builder.

use super::context::{Context, Kernel};
use super::driver::CUdeviceptr;
use crate::Error;

pub fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// Threads per block for the 1D helpers, as on Metal.
pub const THREADS_PER_BLOCK: u32 = 256;

/// Typed kernel arguments in aligned slots.
#[derive(Default)]
pub struct KernelArgs {
    args: Vec<Arg>,
}

enum Arg {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    /// Byte blob in 8-byte-aligned storage (for small POD structs).
    Aligned(Box<[u64]>),
}

impl Arg {
    fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            Arg::U32(v) => std::ptr::from_mut(v).cast(),
            Arg::I32(v) => std::ptr::from_mut(v).cast(),
            Arg::F32(v) => std::ptr::from_mut(v).cast(),
            Arg::U64(v) => std::ptr::from_mut(v).cast(),
            Arg::Aligned(v) => v.as_mut_ptr().cast(),
        }
    }
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.args.push(Arg::U32(v));
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.args.push(Arg::I32(v));
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.args.push(Arg::F32(v));
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.args.push(Arg::U64(v));
        self
    }

    /// Bind a device pointer (a kernel T* argument).
    pub fn device_ptr(&mut self, p: CUdeviceptr) -> &mut Self {
        self.u64(p)
    }

    /// Two consecutive u32s (a uint2 kernel parameter).
    pub fn u32x2(&mut self, a: u32, b: u32) -> &mut Self {
        let packed = [(a as u64) | ((b as u64) << 32)];
        self.args
            .push(Arg::Aligned(packed.to_vec().into_boxed_slice()));
        self
    }

    /// Bind a small value needing at most 8-byte alignment.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        let mut words = vec![0u64; div_up(bytes.len(), 8)].into_boxed_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                words.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
        }
        self.args.push(Arg::Aligned(words));
        self
    }

    // Call only on a complete list, so slots stay put.
    pub(crate) fn pointers(&mut self) -> Vec<*mut std::ffi::c_void> {
        self.args.iter_mut().map(Arg::as_mut_ptr).collect()
    }
}

/// 1D grid, one thread per element (256-thread blocks).
pub fn launch_1d(
    ctx: &Context,
    kernel: &Kernel,
    count: usize,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    let block = THREADS_PER_BLOCK.min(count.max(1) as u32);
    let grid = div_up(count, block as usize) as u32;
    ctx.launch(kernel, (grid, 1, 1), (block, 1, 1), 0, args)
}

/// 1D launches over `count`, split at the grid-x limit.
pub fn launch_1d_ranged(
    ctx: &Context,
    kernel: &Kernel,
    count: usize,
    mut bind: impl FnMut(&mut KernelArgs, u32, u32),
) -> Result<(), Error> {
    const MAX_GRID_X: usize = i32::MAX as usize;
    let chunk = MAX_GRID_X * THREADS_PER_BLOCK as usize;
    let mut base = 0usize;
    while base < count {
        let len = (count - base).min(chunk);
        let mut args = KernelArgs::new();
        bind(&mut args, base as u32, len as u32);
        launch_1d(ctx, kernel, len, &mut args)?;
        base += len;
    }
    Ok(())
}

/// One block per row; y would cap rows at 65535.
pub fn launch_rows(
    ctx: &Context,
    kernel: &Kernel,
    rows: usize,
    block_width: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    ctx.launch(kernel, (rows as u32, 1, 1), (block_width, 1, 1), 0, args)
}

const MAX_GRID_Y: usize = 65535;

/// 2D grid of blocks; put the large axis on x.
pub fn launch_grid(
    ctx: &Context,
    kernel: &Kernel,
    width: usize,
    height: usize,
    block_width: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    assert!(
        height <= MAX_GRID_Y,
        "launch_grid height {height} exceeds the CUDA gridDim.y limit of {MAX_GRID_Y}"
    );
    ctx.launch(
        kernel,
        (width as u32, height as u32, 1),
        (block_width, 1, 1),
        0,
        args,
    )
}
