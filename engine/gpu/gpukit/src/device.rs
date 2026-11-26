//! Device buffers and caches, shared by every GPU backend.

use crate::backend::Backend;
use crate::{Error, Platform};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Static register and memory counts for one compiled kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelStats {
    pub regs: u32,
    pub shared_bytes: u32,
    pub local_bytes: u32,
    pub max_threads: u32,
}

/// Opaque 64-byte handle for GPU IPC memory sharing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IpcHandle(pub [u8; 64]);

static ORDINAL: AtomicUsize = AtomicUsize::new(0);

/// Sets the GPU ordinal opened on first context use.
pub fn select(ordinal: usize) {
    ORDINAL.store(ordinal, Ordering::Relaxed);
}

/// The GPU ordinal selected for the process-wide context.
pub fn ordinal() -> usize {
    ORDINAL.load(Ordering::Relaxed)
}

/// A backend context: memory, kernels and process-wide caches.
pub trait Device: Clone + Send + Sync + 'static {
    type Kernel: Clone + Send + 'static;
    /// A whole plan's launches, recorded once.
    type Graph: Send + 'static;

    const BACKEND: Backend;

    /// Opens the default device.
    fn open() -> Result<Self, Error>;
    /// Wraps a message as this backend's error.
    fn error(msg: String) -> Error;
    fn caches() -> &'static Caches<Self>;
    fn name(&self) -> String;
    /// The device as asked, with its table row.
    fn platform(&self) -> Platform;
    fn alloc(&self, size: usize) -> Result<u64, Error>;
    /// Allocates device memory for kept KV rows.
    fn alloc_kept(&self, size: usize) -> Result<u64, Error> {
        self.alloc(size)
    }
    /// Skipped when the context is already gone.
    fn free(&self, ptr: u64);
    /// Device bytes free now, where the backend can tell.
    fn free_memory(&self) -> Option<u64> {
        None
    }
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error>;
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error>;
    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error>;
    /// Copies `size` bytes from `src` to `dst` on device.
    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        if size == 0 || dst == src {
            return Ok(());
        }
        let mut tmp = vec![0u8; size];
        self.download(src, &mut tmp)?;
        self.upload(dst, &tmp)
    }
    fn compile(&self, source: &str, entry: &str) -> Result<Self::Kernel, Error>;
    /// Waits for every launch. Returns the device's own seconds.
    fn finish(&self) -> Result<Option<f64>, Error>;

    /// Lets this thread's launches overlap the one before.
    fn programmatic(&self, _on: bool) {}

    /// Records a timing stamp. False where the device keeps none.
    fn stamp(&self) -> Result<bool, Error> {
        Ok(false)
    }

    /// Seconds between each pair of stamps. Waits, then frees all.
    fn stamps(&self) -> Result<Vec<f64>, Error> {
        Ok(Vec::new())
    }

    /// Holds queued work until `release`. False where it cannot.
    fn gate(&self) -> Result<bool, Error> {
        Ok(false)
    }

    /// Ends a `gate`.
    fn release(&self) {}

    /// The device address of `kernel`'s global `name`, where kept.
    fn global(&self, _kernel: &Self::Kernel, _name: &str) -> Result<Option<u64>, Error> {
        Ok(None)
    }

    /// Ticks per second of the clock kernels read.
    fn ticks(&self) -> f64 {
        1e9
    }

    /// The bytes at `ptr`, when the host addresses them.
    fn mapped(&self, _ptr: u64, _size: usize) -> Option<NonNull<u8>> {
        None
    }

    /// Pinned host memory: its host and device addresses.
    fn pin(&self, _size: usize) -> Result<(u64, u64), Error> {
        Err(Self::error("this device cannot pin host memory".into()))
    }

    fn unpin(&self, _host: u64) {}

    /// Whether pinned host memory serves kernels as device memory does.
    fn unified(&self) -> bool {
        false
    }

    /// The device's NUMA node on multi-node hosts, if known.
    fn numa_node(&self) -> Option<u32> {
        None
    }

    /// Narrows the calling thread to the device's NUMA CPUs.
    fn bind_numa(&self) {}

    /// Maps existing host memory for the device. Returns its address.
    fn register(&self, _host: *mut u8, _size: usize) -> Result<u64, Error> {
        Err(Self::error("this device cannot map host memory".into()))
    }

    fn unregister(&self, _host: *mut u8) {}

    /// Exports a device allocation for peer GPU processes.
    fn ipc_export(&self, _ptr: u64) -> Result<IpcHandle, Error> {
        Err(Self::error("this device has no GPU IPC".into()))
    }

    /// Maps a peer GPU allocation into this device's address space.
    fn ipc_open(&self, _handle: IpcHandle) -> Result<u64, Error> {
        Err(Self::error("this device has no GPU IPC".into()))
    }

    /// Unmaps a peer GPU allocation opened with `ipc_open`.
    fn ipc_close(&self, _ptr: u64) {}

    /// Blocks of `kernel` one unit holds at once.
    fn resident(&self, _kernel: &Self::Kernel, _threads: u32, _shared: u32) -> Result<u32, Error> {
        Err(Self::error("this device has no persistent grids".into()))
    }

    /// Static register and memory counts for `kernel`.
    fn stats(&self, _kernel: &Self::Kernel) -> Option<KernelStats> {
        None
    }

    /// Whether `capture` records launches here.
    fn graphs(&self) -> bool {
        false
    }

    /// Whether this device supports concurrent execution lanes.
    fn has_lanes(&self) -> bool {
        false
    }

    /// Selects the active execution lane on this thread.
    fn set_lane(&self, _lane: u8) {}

    /// Active execution lane on this thread (`0` = default).
    fn active_lane(&self) -> u8 {
        0
    }

    /// Records sync event `slot` on `lane`.
    fn record_lane(&self, _lane: u8, _slot: usize) -> Result<(), Error> {
        Ok(())
    }

    /// Makes `lane` wait on sync event `slot`.
    fn wait_lane(&self, _lane: u8, _slot: usize) -> Result<(), Error> {
        Ok(())
    }

    /// Whether this thread is recording a device graph now.
    fn capturing(&self) -> bool {
        false
    }

    /// Runs `record` and keeps its launches as one graph.
    fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Self::Graph, E> {
        let _ = record;
        Err(Self::error("this device cannot record graphs".into()).into())
    }

    /// Runs a captured graph, after copies.
    fn replay(&self, _: &Self::Graph) -> Result<(), Error> {
        Err(Self::error("this device cannot record graphs".into()))
    }
}

type KernelMap<K> = OnceLock<Mutex<HashMap<(usize, &'static str), K>>>;

/// One backend's context and kernels, made on first use.
pub struct Caches<D: Device> {
    ctx: OnceLock<Result<D, String>>,
    sources: KernelMap<D::Kernel>,
}

impl<D: Device> Caches<D> {
    pub const fn new() -> Self {
        Self {
            ctx: OnceLock::new(),
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
    /// The host address, when the memory is pinned.
    host: Option<NonNull<u8>>,
}

// The host address is plain memory the buffer owns.
unsafe impl<D: Device> Send for DeviceBuffer<D> {}
unsafe impl<D: Device> Sync for DeviceBuffer<D> {}

impl<D: Device> DeviceBuffer<D> {
    /// Allocate one device buffer outside any pool.
    pub fn alloc(ctx: &D, size: usize) -> Result<Self, Error> {
        Ok(Self {
            ptr: ctx.alloc(size)?,
            ctx: ctx.clone(),
            size,
            host: None,
        })
    }

    /// Allocates one device buffer for kept KV rows.
    pub fn kept(ctx: &D, size: usize) -> Result<Self, Error> {
        Ok(Self {
            ptr: ctx.alloc_kept(size)?,
            ctx: ctx.clone(),
            size,
            host: None,
        })
    }

    /// Pinned host memory the device maps.
    pub fn pinned(ctx: &D, size: usize) -> Result<Self, Error> {
        let (host, ptr) = ctx.pin(size)?;
        Ok(Self {
            ptr,
            ctx: ctx.clone(),
            size,
            host: NonNull::new(host as *mut u8),
        })
    }

    fn host(&self) -> Option<NonNull<u8>> {
        self.host.or_else(|| self.ctx.mapped(self.ptr, self.size))
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

/// Device memory for a zone, mapped when shared.
impl<D: Device> tessera_mem::Backing for DeviceBuffer<D> {
    fn bytes(&self) -> usize {
        self.size
    }

    fn mapped(&self) -> Option<&[u8]> {
        let at = self.host()?;
        // The mapping lives as long as the allocation.
        Some(unsafe { std::slice::from_raw_parts(at.as_ptr(), self.size) })
    }

    fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let at = self.host()?;
        Some(unsafe { std::slice::from_raw_parts_mut(at.as_ptr(), self.size) })
    }

    fn upload(&mut self, at: usize, bytes: &[u8]) -> Result<(), tessera_mem::Error> {
        self.assert_fits(at + bytes.len());
        if let Some(all) = self.host.and(self.mapped_mut()) {
            all[at..at + bytes.len()].copy_from_slice(bytes);
            return Ok(());
        }
        let ptr = self.ptr + at as u64;
        self.ctx
            .upload(ptr, bytes)
            .map_err(|e| tessera_mem::Error::Device(e.to_string()))
    }

    fn download(&self, at: usize, out: &mut [u8]) -> Result<(), tessera_mem::Error> {
        self.assert_fits(at + out.len());
        if let Some(all) = self.host.and(self.mapped()) {
            out.copy_from_slice(&all[at..at + out.len()]);
            return Ok(());
        }
        let ptr = self.ptr + at as u64;
        self.ctx
            .download(ptr, out)
            .map_err(|e| tessera_mem::Error::Device(e.to_string()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<D: Device> Drop for DeviceBuffer<D> {
    fn drop(&mut self) {
        match self.host {
            Some(host) => self.ctx.unpin(host.as_ptr() as u64),
            None => self.ctx.free(self.ptr),
        }
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
#[cfg(any(feature = "cuda", feature = "rocm"))]
macro_rules! backend_api {
    ($ctx:ty, $kernel:ty) => {
        pub type DeviceBuffer = $crate::device::DeviceBuffer<$ctx>;
        pub type BufferPool = $crate::device::BufferPool<$ctx>;

        /// The process-wide context, opened on first use.
        pub fn cached_context() -> Result<&'static $ctx, $crate::Error> {
            $crate::device::cached_context()
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
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) use backend_api;
