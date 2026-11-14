//! A CUDA or HIP context: modules, kernels, graphs and memory.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_int, c_void};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::LocalKey;

use crate::Error;
use crate::device::{Caches, Device};
use crate::dl::image_hash;
use crate::launch::{KernelArgs, Launch};

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

/// What a handle to free is.
pub enum Kind {
    Module,
    Exec,
    Graph,
    Stream,
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

    fn error(msg: String) -> Error;
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
    fn synchronize(&self) -> Result<(), Error>;
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
    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error>;
    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error>;
    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error>;
    /// A loadable image of `source` for this device.
    fn compile(ctx: &Context<Self>, source: &str) -> Result<Vec<u8>, Error>;

    /// Calls `f` with a handle to fill, then checks it.
    fn out(&self, what: &str, f: impl FnOnce(&mut Handle) -> c_int) -> Result<Handle, Error> {
        let mut handle = std::ptr::null_mut();
        let code = f(&mut handle);
        self.check(code, what).map(|_| handle)
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
}

// The driver is thread-safe. Calls set the device first.
unsafe impl<A: Api> Send for Inner<A> {}
unsafe impl<A: Api> Sync for Inner<A> {}

impl<A: Api> Drop for Inner<A> {
    fn drop(&mut self) {
        // Modules must unload before their context goes.
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

    /// Lets this thread's launches overlap the one before.
    pub fn programmatic(&self, on: bool) {
        A::programmatic(on);
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
        })
    }
}

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
    raw: Handle,
    name: String,
}

impl<A: Api> Clone for Kernel<A> {
    fn clone(&self) -> Self {
        Self {
            _module: Arc::clone(&self._module),
            raw: self.raw,
            name: self.name.clone(),
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

    fn open() -> Result<Self, Error> {
        A::open()
    }

    fn error(msg: String) -> Error {
        A::error(msg)
    }

    fn caches() -> &'static Caches<Self> {
        A::caches()
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
