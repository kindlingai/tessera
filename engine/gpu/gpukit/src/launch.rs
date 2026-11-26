//! Kernel arguments and launch helpers shared by CUDA and ROCm.

use crate::Error;
use tessera_platform::limits::GRID_DOWN;

fn div_up(value: usize, group: usize) -> usize {
    value.div_ceil(group)
}

/// Threads per block for the 1D helpers, as on Metal.
pub const THREADS_PER_BLOCK: u32 = 256;

/// A device context that can enqueue a kernel.
pub trait Launch {
    type Kernel;

    /// Enqueue a launch without waiting for it to finish.
    fn launch(
        &self,
        kernel: &Self::Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error>;

    /// Enqueues a launch on a stream of its own.
    fn launch_apart(
        &self,
        _kernel: &Self::Kernel,
        _grid: (u32, u32, u32),
        _block: (u32, u32, u32),
        _args: &mut KernelArgs,
    ) -> Result<Box<dyn Apart>, Error> {
        Err(Error::Gpu("this device has no separate streams"))
    }
}

/// A launch on its own stream.
pub trait Apart: Send {
    /// Waits for the launch to end.
    fn wait(&self) -> Result<(), Error>;
}

/// Typed kernel arguments in aligned slots.
#[derive(Default)]
pub struct KernelArgs {
    args: Vec<Arg>,
}

enum Arg {
    /// A device address.
    Pointer(u64),
    /// `len` bytes of a value, in 8-byte-aligned storage.
    Value { words: Box<[u64]>, len: usize },
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl Arg {
    fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            Arg::Pointer(v) => std::ptr::from_mut(v).cast(),
            Arg::Value { words, .. } => words.as_mut_ptr().cast(),
        }
    }
}

/// One argument, as a backend binds it.
pub enum Bound<'a> {
    Pointer(u64),
    Bytes(&'a [u8]),
}

impl KernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    fn word(&mut self, word: u64, len: usize) -> &mut Self {
        let words = vec![word].into_boxed_slice();
        self.args.push(Arg::Value { words, len });
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.word(v as u64, 4)
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.word(v as u32 as u64, 4)
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.word(v.to_bits() as u64, 4)
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.word(v, 8)
    }

    /// Bind a device pointer (a kernel T* argument).
    pub fn device_ptr(&mut self, p: u64) -> &mut Self {
        self.args.push(Arg::Pointer(p));
        self
    }

    /// Two consecutive u32s (a uint2 kernel parameter).
    pub fn u32x2(&mut self, a: u32, b: u32) -> &mut Self {
        self.word((a as u64) | ((b as u64) << 32), 8)
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
        let len = bytes.len();
        self.args.push(Arg::Value { words, len });
        self
    }

    // Call only on a complete list, so slots stay put.
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    pub(crate) fn pointers(&mut self) -> Vec<*mut std::ffi::c_void> {
        self.args.iter_mut().map(Arg::as_mut_ptr).collect()
    }

    /// Each argument in order, for binding by index.
    pub fn bound(&self) -> impl Iterator<Item = Bound<'_>> {
        self.args.iter().map(|a| match a {
            Arg::Pointer(p) => Bound::Pointer(*p),
            Arg::Value { words, len } => {
                // The words are plain data, read as bytes in place.
                let at = words.as_ptr().cast::<u8>();
                Bound::Bytes(unsafe { std::slice::from_raw_parts(at, *len) })
            }
        })
    }
}

/// 1D grid, one thread per element (256-thread blocks).
pub fn launch_1d<C: Launch>(
    ctx: &C,
    kernel: &C::Kernel,
    count: usize,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    let block = THREADS_PER_BLOCK.min(count.max(1) as u32);
    let grid = div_up(count, block as usize) as u32;
    ctx.launch(kernel, (grid, 1, 1), (block, 1, 1), 0, args)
}

/// 2D grid of blocks. Put the large axis on x.
pub fn launch_grid<C: Launch>(
    ctx: &C,
    kernel: &C::Kernel,
    width: usize,
    height: usize,
    block_width: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    assert!(
        height <= GRID_DOWN,
        "launch_grid height {height} exceeds the gridDim.y limit of {GRID_DOWN}"
    );
    ctx.launch(
        kernel,
        (width as u32, height as u32, 1),
        (block_width, 1, 1),
        0,
        args,
    )
}
