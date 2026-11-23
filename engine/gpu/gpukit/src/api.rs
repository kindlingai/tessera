//! A CUDA or HIP context: modules, kernels, graphs and memory.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::LocalKey;

use crate::backend::Backend;
use crate::device::{Caches, Device, IpcHandle, KernelStats};
use crate::dl::image_hash;
use crate::gate::Gate;
use crate::launch::{self, KernelArgs, Launch};
use crate::{Error, Platform};

/// A driver handle: a module, function, stream or graph.
pub type Handle = *mut c_void;

/// The launch entry point CUDA and HIP share.
pub type LaunchFn = unsafe extern "C" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut Handle,
    *mut Handle,
) -> c_int;

/// Other entry point types CUDA and HIP share.
pub(crate) mod entry {
    use super::Handle;
    use std::ffi::{c_char, c_int, c_void};

    pub(crate) type Init = unsafe extern "C" fn(u32) -> c_int;
    /// Writes one int, such as a count or version.
    pub(crate) type Int = unsafe extern "C" fn(*mut c_int) -> c_int;
    pub(crate) type Name = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    pub(crate) type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    /// Writes a device's total memory in bytes.
    pub(crate) type Total = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
    pub(crate) type Load = unsafe extern "C" fn(*mut Handle, *const c_void) -> c_int;
    pub(crate) type Function = unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> c_int;
    /// Unloads, destroys or synchronizes one handle.
    pub(crate) type OnHandle = unsafe extern "C" fn(Handle) -> c_int;
    pub(crate) type Alloc = unsafe extern "C" fn(*mut u64, usize) -> c_int;
    pub(crate) type Free = unsafe extern "C" fn(u64) -> c_int;
    pub(crate) type HostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> c_int;
    pub(crate) type HostMap = unsafe extern "C" fn(*mut u64, *mut c_void, u32) -> c_int;
    pub(crate) type HostFree = unsafe extern "C" fn(*mut c_void) -> c_int;
    pub(crate) type Register = unsafe extern "C" fn(*mut c_void, usize, u32) -> c_int;
    /// Blocks of a function one unit holds: threads, shared bytes.
    pub(crate) type Occupancy = unsafe extern "C" fn(*mut c_int, Handle, c_int, usize) -> c_int;
    pub(crate) type Stream = unsafe extern "C" fn(*mut Handle, u32) -> c_int;
    pub(crate) type Capture = unsafe extern "C" fn(Handle, c_int) -> c_int;
    pub(crate) type EndCapture = unsafe extern "C" fn(Handle, *mut Handle) -> c_int;
    pub(crate) type Replay = unsafe extern "C" fn(Handle, Handle) -> c_int;
    /// Milliseconds between two events.
    pub(crate) type Elapsed = unsafe extern "C" fn(*mut f32, Handle, Handle) -> c_int;
    /// A module global's address and size, by name.
    pub(crate) type Global =
        unsafe extern "C" fn(*mut u64, *mut usize, Handle, *const c_char) -> c_int;
    pub(crate) type IpcExport = unsafe extern "C" fn(*mut crate::device::IpcHandle, u64) -> c_int;
    pub(crate) type IpcOpen =
        unsafe extern "C" fn(*mut u64, crate::device::IpcHandle, u32) -> c_int;
    pub(crate) type IpcClose = unsafe extern "C" fn(u64) -> c_int;
}

/// The kind of handle to free.
pub enum Kind {
    Module,
    Exec,
    Graph,
    Stream,
    Event,
}

/// One kernel launch.
pub struct Call {
    pub kernel: Handle,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub stream: Handle,
    pub params: *mut Handle,
}

/// # Safety
/// `call` holds live handles and parameters.
pub unsafe fn plain(f: LaunchFn, call: &Call) -> c_int {
    let ([gx, gy, gz], [bx, by, bz]) = (call.grid, call.block);
    let null = std::ptr::null_mut();
    unsafe {
        f(
            call.kernel,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            call.shared_bytes,
            call.stream,
            call.params,
            null,
        )
    }
}

/// Which device to open.
#[derive(Debug, Clone, Default)]
pub struct ContextConfig {
    /// Zero-based. The backend's visible-devices variable still applies.
    pub device_ordinal: usize,
}

/// One backend's driver calls.
pub trait Api: Sync + Sized + 'static {
    /// Device state beyond the shared parts.
    type Own;

    const BACKEND: Backend;

    fn error(msg: String) -> Error;
    /// The device as asked, with its table row.
    fn describe(ctx: &Context<Self>) -> Platform;
    fn open() -> Result<Context<Self>, Error>;
    fn caches() -> &'static Caches<Context<Self>>;
    /// This thread's capture stream, or null.
    fn capturing() -> &'static LocalKey<Cell<Handle>>;
    /// Lets this thread's launches overlap the one before.
    fn programmatic(_on: bool) {}
    /// Runs after the modules unload.
    fn close(&self, _own: &Self::Own) {}
    /// Turns a failed return code into an error naming `what`.
    fn check(&self, code: c_int, what: &str) -> Result<(), Error>;
    fn set_current(&self, own: &Self::Own, device: c_int) -> Result<(), Error>;
    /// Waits for the default stream, where all plain work runs.
    fn synchronize(&self) -> Result<(), Error> {
        self.wait(std::ptr::null_mut())
    }
    fn wait(&self, stream: Handle) -> Result<(), Error>;
    fn stream(&self) -> Result<Handle, Error>;
    fn begin_capture(&self, stream: Handle) -> Result<(), Error>;
    fn end_capture(&self, stream: Handle, graph: &mut Handle) -> Result<(), Error>;
    fn instantiate(&self, graph: Handle) -> Result<Handle, Error>;
    /// Runs a graph on the default stream.
    fn replay(&self, exec: Handle) -> Result<(), Error>;
    fn destroy(&self, kind: Kind, handle: Handle);
    fn load(&self, image: &[u8]) -> Result<Handle, Error>;
    fn function(&self, module: Handle, name: &CStr) -> Result<Handle, Error>;
    fn launch(&self, call: &Call) -> Result<(), Error>;
    fn alloc(&self, size: usize) -> Result<u64, Error>;
    fn free(&self, ptr: u64);
    /// Pinned host memory the device maps: (host, device).
    fn host_alloc(&self, size: usize) -> Result<(u64, u64), Error>;
    fn host_free(&self, host: u64);
    /// Maps existing host memory for the device. Returns its address.
    fn register(&self, host: *mut c_void, size: usize) -> Result<u64, Error>;
    fn unregister(&self, host: *mut c_void);
    fn ipc_export(&self, ptr: u64) -> Result<IpcHandle, Error>;
    fn ipc_open(&self, handle: IpcHandle) -> Result<u64, Error>;
    fn ipc_close(&self, ptr: u64);
    fn occupancy(&self, kernel: Handle, threads: u32, shared: u32) -> Result<u32, Error>;
    /// Raises the dynamic shared memory limit on `kernel`.
    fn allow_shared(&self, _kernel: Handle, _shared: u32) -> Result<(), Error> {
        Ok(())
    }
    /// Static register and memory counts for `kernel`.
    fn stats(&self, _kernel: Handle) -> Option<KernelStats> {
        None
    }
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error>;
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error>;
    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error>;
    /// A timing event, recorded on the default stream.
    fn event(&self) -> Result<Handle, Error>;
    /// Milliseconds from `from` to `to`, once both are done.
    fn elapsed(&self, from: Handle, to: Handle) -> Result<f32, Error>;
    /// The address of `module`'s global `name`, if it has one.
    fn global(&self, module: Handle, name: &CStr) -> Option<u64>;
    /// Ticks per second of the clock kernels read.
    fn ticks(_ctx: &Context<Self>) -> f64 {
        1e9
    }
    /// A loadable image of `source` for this device.
    fn compile(ctx: &Context<Self>, source: &str) -> Result<Vec<u8>, Error>;

    /// Fills a handle through `f`, then checks its code.
    fn out(&self, what: &str, f: impl FnOnce(&mut Handle) -> c_int) -> Result<Handle, Error> {
        let mut handle = std::ptr::null_mut();
        let code = f(&mut handle);
        self.check(code, what).map(|_| handle)
    }

    /// Fills an int through `f`, then checks its code.
    fn int(&self, what: &str, f: impl FnOnce(&mut c_int) -> c_int) -> Result<c_int, Error> {
        let mut value = 0;
        let code = f(&mut value);
        self.check(code, what).map(|_| value)
    }

    /// Fails unless `ordinal` is below the count `f` reports.
    fn visible(&self, f: entry::Int, what: &str, ordinal: usize, kind: &str) -> Result<(), Error> {
        let count = self.int(what, |c| unsafe { f(c) })?;
        if ordinal < count as usize {
            return Ok(());
        }
        Err(Self::error(format!(
            "device ordinal {ordinal} out of range ({count} visible {kind} devices)"
        )))
    }

    /// The name `f` reports for `device`.
    fn device_name(&self, f: entry::Name, what: &str, device: c_int) -> Result<String, Error> {
        let mut buf = [0 as c_char; 256];
        self.check(unsafe { f(buf.as_mut_ptr(), 256, device) }, what)?;
        let name = unsafe { CStr::from_ptr(buf.as_ptr()) };
        Ok(name.to_string_lossy().into_owned())
    }
}

/// A device plus a cache of loaded modules.
pub struct Context<A: Api>(Arc<Inner<A>>);

impl<A: Api> Clone for Context<A> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

struct Inner<A: Api> {
    api: &'static A,
    own: A::Own,
    device: c_int,
    name: String,
    modules: Mutex<HashMap<u64, Arc<Module<A>>>>,
    /// Held through a capture, and by every other device call.
    exclusive: Mutex<()>,
    /// Events recorded since the last read, as addresses.
    stamps: Mutex<Vec<usize>>,
    /// Made on first use.
    gate: Mutex<Option<Gate<A>>>,
}

// The driver is thread-safe. Calls set the device first.
unsafe impl<A: Api> Send for Inner<A> {}
unsafe impl<A: Api> Sync for Inner<A> {}

impl<A: Api> Drop for Inner<A> {
    fn drop(&mut self) {
        // Modules must unload before their context goes.
        if let Ok(mut gate) = self.gate.lock()
            && self.api.set_current(&self.own, self.device).is_ok()
        {
            gate.take();
        }
        if let Ok(mut modules) = self.modules.lock() {
            modules.clear();
        }
        self.api.close(&self.own);
    }
}

impl<A: Api> Context<A> {
    pub(crate) fn made(api: &'static A, own: A::Own, device: c_int, name: String) -> Self {
        Self(Arc::new(Inner {
            api,
            own,
            device,
            name,
            modules: Mutex::new(HashMap::new()),
            exclusive: Mutex::new(()),
            stamps: Mutex::new(Vec::new()),
            gate: Mutex::new(None),
        }))
    }

    pub(crate) fn own(&self) -> &A::Own {
        &self.0.own
    }

    pub fn api(&self) -> &'static A {
        self.0.api
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn device(&self) -> c_int {
        self.0.device
    }

    /// Make this device current on the calling thread.
    pub fn set_current(&self) -> Result<(), Error> {
        self.0.api.set_current(&self.0.own, self.0.device)
    }

    /// The lock, unless this thread is the one capturing.
    fn hold(&self) -> Option<MutexGuard<'_, ()>> {
        match A::capturing().with(Cell::get).is_null() {
            true => Some(self.0.exclusive.lock().unwrap_or_else(|e| e.into_inner())),
            false => None,
        }
    }

    /// Runs `f` under the lock with this device current.
    fn locked<T>(&self, f: impl FnOnce(&'static A) -> Result<T, Error>) -> Result<T, Error> {
        let _hold = self.hold();
        self.set_current()?;
        f(self.0.api)
    }

    /// Waits for the default stream, where all plain work runs.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.locked(|a| a.synchronize())
    }

    /// Records a timing event after the work queued so far.
    pub fn stamp(&self) -> Result<(), Error> {
        let event = self.locked(|a| a.event())?;
        let mut stamps = self.0.stamps.lock().unwrap_or_else(|e| e.into_inner());
        stamps.push(event as usize);
        Ok(())
    }

    /// Seconds between each pair of stamps. Waits, then frees all.
    pub fn stamps(&self) -> Result<Vec<f64>, Error> {
        self.synchronize()?;
        let taken = std::mem::take(&mut *self.0.stamps.lock().unwrap_or_else(|e| e.into_inner()));
        let events: Vec<Handle> = taken.into_iter().map(|e| e as Handle).collect();
        let gaps = self.locked(|a| {
            let pairs = events.windows(2);
            let ms: Result<Vec<f32>, Error> = pairs.map(|w| a.elapsed(w[0], w[1])).collect();
            for e in &events {
                a.destroy(Kind::Event, *e);
            }
            ms
        })?;
        Ok(gaps.into_iter().map(|ms| ms as f64 / 1e3).collect())
    }

    /// The device address of `kernel`'s module global `name`.
    pub fn global(&self, kernel: &Kernel<A>, name: &str) -> Result<Option<u64>, Error> {
        let name = CString::new(name).map_err(|e| A::error(format!("global {name:?}: {e}")))?;
        let module = kernel._module.raw;
        self.locked(|a| Ok(a.global(module, &name)))
    }

    /// Holds the default stream until `release`.
    pub fn gate(&self) -> Result<(), Error> {
        let mut gate = self.0.gate.lock().unwrap_or_else(|e| e.into_inner());
        let _hold = self.hold();
        self.set_current()?;
        if gate.is_none() {
            *gate = Some(Gate::new(self)?);
        }
        gate.as_ref().map_or(Ok(()), Gate::close)
    }

    /// Lets the default stream run past the gate.
    pub fn release(&self) {
        let gate = self.0.gate.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(gate) = gate.as_ref() {
            gate.open();
        }
    }

    /// Runs `record` and keeps its launches as one graph.
    pub fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Graph<A>, E> {
        let _hold = self.hold();
        self.set_current()?;
        let api = self.0.api;
        let mut graph = Graph {
            api,
            stream: api.stream()?,
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
        };
        api.begin_capture(graph.stream)?;
        A::capturing().with(|c| c.set(graph.stream));
        let recorded = record();
        A::capturing().with(|c| c.set(std::ptr::null_mut()));
        // The capture ends either way, or the stream stays stuck.
        let ended = api.end_capture(graph.stream, &mut graph.graph);
        recorded.and(ended.map_err(E::from))?;
        graph.exec = api.instantiate(graph.graph)?;
        Ok(graph)
    }

    /// Runs a captured graph on the default stream, after copies.
    pub fn replay(&self, graph: &Graph<A>) -> Result<(), Error> {
        self.set_current()?;
        self.0.api.replay(graph.exec)
    }

    /// Load a code image, reusing identical images.
    pub fn load_module(&self, image: &[u8]) -> Result<Arc<Module<A>>, Error> {
        let key = image_hash(image);
        if let Some(existing) = self.0.modules.lock().unwrap().get(&key) {
            return Ok(Arc::clone(existing));
        }
        self.set_current()?;
        let module = Arc::new(Module {
            api: self.0.api,
            raw: self.0.api.load(image)?,
        });
        self.0
            .modules
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Pinned host memory the device maps: (host, device).
    pub fn pin(&self, size: usize) -> Result<(u64, u64), Error> {
        self.locked(|a| a.host_alloc(size))
    }

    pub fn unpin(&self, host: u64) {
        let _hold = self.hold();
        if self.set_current().is_ok() {
            self.0.api.host_free(host);
        }
    }

    /// Maps `size` bytes at `host` for the device.
    pub fn register(&self, host: *mut u8, size: usize) -> Result<u64, Error> {
        self.locked(|a| a.register(host.cast(), size))
    }

    pub fn unregister(&self, host: *mut u8) {
        let _hold = self.hold();
        if self.set_current().is_ok() {
            self.0.api.unregister(host.cast());
        }
    }

    pub fn ipc_export(&self, ptr: u64) -> Result<IpcHandle, Error> {
        self.locked(|a| a.ipc_export(ptr))
    }

    pub fn ipc_open(&self, handle: IpcHandle) -> Result<u64, Error> {
        self.locked(|a| a.ipc_open(handle))
    }

    pub fn ipc_close(&self, ptr: u64) {
        let _hold = self.hold();
        if self.set_current().is_ok() {
            self.0.api.ipc_close(ptr);
        }
    }

    /// Blocks of `kernel` one unit holds at once.
    pub fn resident(&self, kernel: &Kernel<A>, threads: u32, shared: u32) -> Result<u32, Error> {
        self.locked(|a| {
            Self::opt_in(a, kernel, shared)?;
            a.occupancy(kernel.raw, threads, shared)
        })
    }

    fn opt_in(api: &A, kernel: &Kernel<A>, shared: u32) -> Result<(), Error> {
        if shared <= kernel.shared.load(Ordering::Relaxed) {
            return Ok(());
        }
        api.allow_shared(kernel.raw, shared)?;
        kernel.shared.store(shared, Ordering::Relaxed);
        Ok(())
    }

    /// Static register and memory counts for `kernel`.
    pub fn stats(&self, kernel: &Kernel<A>) -> Option<KernelStats> {
        self.locked(|a| Ok(a.stats(kernel.raw))).ok().flatten()
    }

    /// Compile for this device and resolve `entry`.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<Kernel<A>, Error> {
        let image = A::compile(self, source)?;
        self.load_module(&image)?.function(entry)
    }
}

impl<A: Api> Launch for Context<A> {
    type Kernel = Kernel<A>;

    fn launch(
        &self,
        kernel: &Kernel<A>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        self.locked(|api| {
            Self::opt_in(api, kernel, shared_bytes)?;
            let mut params = args.pointers();
            api.launch(&Call {
                kernel: kernel.raw,
                grid: [grid.0, grid.1, grid.2],
                block: [block.0, block.1, block.2],
                shared_bytes,
                stream: A::capturing().with(Cell::get),
                params: params.as_mut_ptr(),
            })
        })
    }

    fn launch_apart(
        &self,
        kernel: &Kernel<A>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut KernelArgs,
    ) -> Result<Box<dyn launch::Apart>, Error> {
        self.locked(|api| {
            let stream = Stream {
                api,
                raw: api.stream()?,
            };
            let mut params = args.pointers();
            api.launch(&Call {
                kernel: kernel.raw,
                grid: [grid.0, grid.1, grid.2],
                block: [block.0, block.1, block.2],
                shared_bytes: 0,
                stream: stream.raw,
                params: params.as_mut_ptr(),
            })?;
            Ok(Box::new(Apart {
                ctx: self.clone(),
                stream,
            }) as Box<dyn launch::Apart>)
        })
    }
}

/// A stream of its own, destroyed on drop.
struct Stream<A: Api> {
    api: &'static A,
    raw: Handle,
}

unsafe impl<A: Api> Send for Stream<A> {}

impl<A: Api> Drop for Stream<A> {
    fn drop(&mut self) {
        self.api.destroy(Kind::Stream, self.raw);
    }
}

/// A launch on its own stream, with its context.
struct Apart<A: Api> {
    ctx: Context<A>,
    stream: Stream<A>,
}

impl<A: Api> launch::Apart for Apart<A> {
    fn wait(&self) -> Result<(), Error> {
        self.ctx.locked(|a| a.wait(self.stream.raw))
    }
}

/// A loaded module, unloaded when its last kernel drops.
pub struct Module<A: Api> {
    api: &'static A,
    raw: Handle,
}

impl<A: Api> Module<A> {
    pub fn function(self: &Arc<Self>, name: &str) -> Result<Kernel<A>, Error> {
        let c_name = CString::new(name)
            .map_err(|e| A::error(format!("kernel name {name:?} has a NUL: {e}")))?;
        let raw = self.api.function(self.raw, &c_name).map_err(|e| match e {
            // The bare driver error does not name the entry.
            Error::Cuda(msg) | Error::Rocm(msg) => A::error(format!("{msg} for entry {name:?}")),
            other => other,
        })?;
        Ok(Kernel {
            _module: Arc::clone(self),
            raw,
            name: name.to_string(),
            shared: Arc::new(AtomicU32::new(DEFAULT_SHARED)),
        })
    }
}

/// Dynamic shared memory available before opt-in.
const DEFAULT_SHARED: u32 = 48 * 1024;

// Thread-safe for the same reason as `Inner`.
unsafe impl<A: Api> Send for Module<A> {}
unsafe impl<A: Api> Sync for Module<A> {}

impl<A: Api> Drop for Module<A> {
    fn drop(&mut self) {
        self.api.destroy(Kind::Module, self.raw);
    }
}

/// A resolved kernel entry point that keeps its module alive.
pub struct Kernel<A: Api> {
    /// Holding it keeps the module loaded.
    _module: Arc<Module<A>>,
    pub(crate) raw: Handle,
    name: String,
    /// Highest dynamic shared size already opted into.
    shared: Arc<AtomicU32>,
}

impl<A: Api> Clone for Kernel<A> {
    fn clone(&self) -> Self {
        Self {
            _module: Arc::clone(&self._module),
            raw: self.raw,
            name: self.name.clone(),
            shared: Arc::clone(&self.shared),
        }
    }
}

// Thread-safe like `Inner`, and the Arc keeps the module alive.
unsafe impl<A: Api> Send for Kernel<A> {}
unsafe impl<A: Api> Sync for Kernel<A> {}

impl<A: Api> Kernel<A> {
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Launches recorded once, replayed on their own stream.
pub struct Graph<A: Api> {
    api: &'static A,
    stream: Handle,
    graph: Handle,
    exec: Handle,
}

unsafe impl<A: Api> Send for Graph<A> {}
unsafe impl<A: Api> Sync for Graph<A> {}

impl<A: Api> Drop for Graph<A> {
    fn drop(&mut self) {
        let made = [
            (Kind::Exec, self.exec),
            (Kind::Graph, self.graph),
            (Kind::Stream, self.stream),
        ];
        for (kind, handle) in made {
            if !handle.is_null() {
                self.api.destroy(kind, handle);
            }
        }
    }
}

impl<A: Api> Device for Context<A> {
    type Kernel = Kernel<A>;
    type Graph = Graph<A>;

    const BACKEND: Backend = A::BACKEND;

    fn open() -> Result<Self, Error> {
        A::open()
    }

    fn error(msg: String) -> Error {
        A::error(msg)
    }

    fn caches() -> &'static Caches<Self> {
        A::caches()
    }

    fn name(&self) -> String {
        self.0.name.clone()
    }

    fn platform(&self) -> Platform {
        A::describe(self)
    }

    fn finish(&self) -> Result<Option<f64>, Error> {
        self.synchronize()?;
        Ok(None)
    }

    fn stamp(&self) -> Result<bool, Error> {
        Context::stamp(self).map(|_| true)
    }

    fn stamps(&self) -> Result<Vec<f64>, Error> {
        Context::stamps(self)
    }

    fn gate(&self) -> Result<bool, Error> {
        Context::gate(self).map(|_| true)
    }

    fn global(&self, kernel: &Kernel<A>, name: &str) -> Result<Option<u64>, Error> {
        Context::global(self, kernel, name)
    }

    fn ticks(&self) -> f64 {
        A::ticks(self)
    }

    fn release(&self) {
        Context::release(self);
    }

    fn programmatic(&self, on: bool) {
        A::programmatic(on);
    }

    fn pin(&self, size: usize) -> Result<(u64, u64), Error> {
        Context::pin(self, size)
    }

    fn unpin(&self, host: u64) {
        Context::unpin(self, host);
    }

    fn unified(&self) -> bool {
        let platform = A::describe(self);
        platform.integrated && platform.row.is_some_and(|r| r.pinned)
    }

    fn register(&self, host: *mut u8, size: usize) -> Result<u64, Error> {
        Context::register(self, host, size)
    }

    fn unregister(&self, host: *mut u8) {
        Context::unregister(self, host);
    }

    fn ipc_export(&self, ptr: u64) -> Result<IpcHandle, Error> {
        Context::ipc_export(self, ptr)
    }

    fn ipc_open(&self, handle: IpcHandle) -> Result<u64, Error> {
        Context::ipc_open(self, handle)
    }

    fn ipc_close(&self, ptr: u64) {
        Context::ipc_close(self, ptr);
    }

    fn resident(&self, kernel: &Kernel<A>, threads: u32, shared: u32) -> Result<u32, Error> {
        Context::resident(self, kernel, threads, shared)
    }

    fn stats(&self, kernel: &Kernel<A>) -> Option<KernelStats> {
        Context::stats(self, kernel)
    }

    fn graphs(&self) -> bool {
        true
    }

    fn capturing(&self) -> bool {
        !A::capturing().with(Cell::get).is_null()
    }

    fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Graph<A>, E> {
        Context::capture(self, record)
    }

    fn replay(&self, graph: &Graph<A>) -> Result<(), Error> {
        Context::replay(self, graph)
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        self.locked(|a| a.alloc(size))
    }

    fn free(&self, ptr: u64) {
        let _hold = self.hold();
        // At process exit the context may already be gone.
        if self.set_current().is_ok() {
            self.0.api.free(ptr);
        }
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        self.locked(|a| a.upload(ptr, bytes))
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.locked(|a| a.download(ptr, bytes))
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.locked(|a| a.zero(ptr, size))
    }

    fn compile(&self, source: &str, entry: &str) -> Result<Kernel<A>, Error> {
        self.compile_kernel(source, entry)
    }
}

mod shared;
pub(crate) use shared::shared_api;
