//! Device buffers and caches for the CUDA and ROCm backends.

use crate::Error;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A backend context: memory, kernels and process-wide caches.
pub trait Device: Clone + Send + Sync + 'static {
    type Kernel: Clone + Send + 'static;

    /// Opens the default device.
    fn open() -> Result<Self, Error>;
    /// Wraps a message as this backend's error.
    fn error(msg: String) -> Error;
    fn caches() -> &'static Caches<Self>;
    fn alloc(&self, size: usize) -> Result<u64, Error>;
    /// Skipped when the context is already gone.
    fn free(&self, ptr: u64);
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error>;
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error>;
    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error>;
    fn load_image(&self, image: &[u8], entry: &str) -> Result<Self::Kernel, Error>;
    fn compile(&self, source: &str, entry: &str) -> Result<Self::Kernel, Error>;
}

type KernelMap<K> = OnceLock<Mutex<HashMap<(usize, &'static str), K>>>;

/// One backend's context and kernels, made on first use.
pub struct Caches<D: Device> {
    ctx: OnceLock<Result<D, String>>,
    images: KernelMap<D::Kernel>,
    sources: KernelMap<D::Kernel>,
}

impl<D: Device> Caches<D> {
    pub const fn new() -> Self {
        Self {
            ctx: OnceLock::new(),
            images: OnceLock::new(),
            sources: OnceLock::new(),
        }
    }
}

impl<D: Device> Default for Caches<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide context, opened on first use.
pub fn cached_context<D: Device>() -> Result<&'static D, Error> {
    let ctx = D::caches()
        .ctx
        .get_or_init(|| D::open().map_err(|e| e.to_string()));
    ctx.as_ref().map_err(|msg| D::error(msg.clone()))
}

/// Keyed by address. Each embedded image builds once.
fn lookup<D: Device>(
    map: &KernelMap<D::Kernel>,
    key: (usize, &'static str),
    build: impl FnOnce(&D) -> Result<D::Kernel, Error>,
) -> Result<D::Kernel, Error> {
    let map = map.get_or_init(Default::default);
    if let Some(existing) = map.lock().unwrap().get(&key) {
        return Ok(existing.clone());
    }
    let kernel = build(cached_context::<D>()?)?;
    map.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

/// Resolve `entry` from a compiled image, once per process.
pub fn cached_kernel<D: Device>(
    image: &'static [u8],
    entry: &'static str,
) -> Result<D::Kernel, Error> {
    let key = (image.as_ptr() as usize, entry);
    lookup(&D::caches().images, key, |c: &D| c.load_image(image, entry))
}

/// Compile `source` and resolve `entry`, once per process.
pub fn cached_source<D: Device>(
    source: &'static str,
    entry: &'static str,
) -> Result<D::Kernel, Error> {
    let key = (source.as_ptr() as usize, entry);
    lookup(&D::caches().sources, key, |c: &D| c.compile(source, entry))
}

/// A device allocation, freed on drop.
pub struct DeviceBuffer<D: Device> {
    ctx: D,
    ptr: u64,
    size: usize,
}

impl<D: Device> DeviceBuffer<D> {
    /// Allocate one device buffer outside any pool.
    pub fn alloc(ctx: &D, size: usize) -> Result<Self, Error> {
        Ok(Self {
            ptr: ctx.alloc(size)?,
            ctx: ctx.clone(),
            size,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Raw device address, for binding as a kernel argument.
    pub fn device_ptr(&self) -> u64 {
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
        self.ctx.upload(self.ptr, bytes)
    }

    /// Device to host. Synchronous.
    pub fn read_bytes(&self, bytes: &mut [u8]) -> Result<(), Error> {
        self.assert_fits(bytes.len());
        self.ctx.download(self.ptr, bytes)
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
        self.ctx.zero(self.ptr, self.size)
    }
}

impl<D: Device> Drop for DeviceBuffer<D> {
    fn drop(&mut self) {
        self.ctx.free(self.ptr);
    }
}

/// Reusable device buffers. `allocate` zeroes each one it returns.
pub struct BufferPool<D: Device> {
    free: Vec<DeviceBuffer<D>>,
}

impl<D: Device> Default for BufferPool<D> {
    fn default() -> Self {
        Self { free: Vec::new() }
    }
}

impl<D: Device> BufferPool<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate(&mut self, ctx: &D, size: usize) -> Result<DeviceBuffer<D>, Error> {
        let buf = match self.free.iter().position(|b| b.size >= size) {
            Some(idx) => self.free.swap_remove(idx),
            None => DeviceBuffer::alloc(ctx, size)?,
        };
        buf.zero()?;
        Ok(buf)
    }

    pub fn release(&mut self, buffer: DeviceBuffer<D>) {
        self.free.push(buffer);
    }

    pub fn clear(&mut self) {
        self.free.clear();
    }

    pub fn trim(&mut self, max_buffers: usize) {
        self.free.truncate(max_buffers);
    }
}

/// A backend module's buffer types and cached entry points.
macro_rules! backend_api {
    ($ctx:ty, $kernel:ty) => {
        pub type DeviceBuffer = $crate::device::DeviceBuffer<$ctx>;
        pub type BufferPool = $crate::device::BufferPool<$ctx>;

        /// The process-wide context, opened on first use.
        pub fn cached_context() -> Result<&'static $ctx, $crate::Error> {
            $crate::device::cached_context()
        }

        /// Resolve `entry` from a compiled image, once per process.
        pub fn cached_kernel(
            image: &'static [u8],
            entry: &'static str,
        ) -> Result<$kernel, $crate::Error> {
            $crate::device::cached_kernel::<$ctx>(image, entry)
        }

        /// Compile `source` and resolve `entry`, once per process.
        pub fn cached_source_kernel(
            source: &'static str,
            entry: &'static str,
        ) -> Result<$kernel, $crate::Error> {
            $crate::device::cached_source::<$ctx>(source, entry)
        }
    };
}
pub(crate) use backend_api;
