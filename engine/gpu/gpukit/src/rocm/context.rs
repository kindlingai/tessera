//! HIP device context, module cache, and kernel handles.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::KernelArgs;
use super::hip::{self, HipFunction, HipGraph, HipGraphExec, HipModule, HipStream, Runtime};
use crate::device::{Caches, Device};
use crate::dl::image_hash;
use crate::launch::Launch;
use crate::{Error, Limits};

/// Which HIP device to open.
#[derive(Debug, Clone, Default)]
pub struct ContextConfig {
    /// Zero-based device ordinal (HIP_VISIBLE_DEVICES still applies).
    pub device_ordinal: usize,
}

/// A HIP device plus a cache of loaded modules.
#[derive(Clone)]
pub struct Context(Arc<Inner>);

struct Inner {
    runtime: &'static Runtime,
    device: c_int,
    name: String,
    modules: Mutex<HashMap<u64, Arc<Module>>>,
    /// `warpSize`, asked once.
    lanes: OnceLock<usize>,
    /// Held through a capture, and by every other device call.
    exclusive: Mutex<()>,
}

impl Context {
    /// Initialize the runtime and select one device.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let runtime = hip::runtime()?;
        runtime.check(unsafe { (runtime.hip_init)(0) }, "hipInit")?;

        let mut count: c_int = 0;
        runtime.check(
            unsafe { (runtime.hip_get_device_count)(&mut count) },
            "hipGetDeviceCount",
        )?;
        if config.device_ordinal >= count as usize {
            return Err(Error::Rocm(format!(
                "device ordinal {} out of range ({count} visible HIP devices)",
                config.device_ordinal
            )));
        }
        let device = config.device_ordinal as c_int;

        let mut name_buf = [0 as c_char; 256];
        runtime.check(
            unsafe { (runtime.hip_device_get_name)(name_buf.as_mut_ptr(), 256, device) },
            "hipDeviceGetName",
        )?;
        let mut name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // APU GPU agents often report an empty marketing name.
        if name.is_empty() {
            name = format!("HIP device {device}");
        }

        let ctx = Self(Arc::new(Inner {
            runtime,
            device,
            name,
            modules: Mutex::new(HashMap::new()),
            lanes: OnceLock::new(),
            exclusive: Mutex::new(()),
        }));
        ctx.set_current()?;
        Ok(ctx)
    }

    /// Lanes come from the device. The rest are AMD constants.
    pub fn limits(&self) -> Limits {
        let lanes = *self
            .0
            .lanes
            .get_or_init(|| self.probe_lanes().unwrap_or(64));
        Limits {
            lanes,
            group_bytes: 64 * 1024,
            group_threads: 1024,
            compiler: true,
        }
    }

    /// A kernel's `warpSize`: 32 on RDNA, 64 on CDNA.
    fn probe_lanes(&self) -> Result<usize, Error> {
        let source = "extern \"C\" __global__ void tessera_lanes(unsigned *out) {
    if (threadIdx.x == 0) {
        out[0] = warpSize;
    }
}
";
        let kernel = self.compile(source, "tessera_lanes")?;
        let out = self.alloc(4)?;
        let mut args = crate::launch::KernelArgs::new();
        args.device_ptr(out);
        let launched = crate::launch::launch_1d(self, &kernel, 64, &mut args).and_then(|_| {
            self.synchronize()?;
            let mut bytes = [0u8; 4];
            self.download(out, &mut bytes)?;
            Ok(u32::from_ne_bytes(bytes) as usize)
        });
        self.free(out);
        launched
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn device(&self) -> c_int {
        self.0.device
    }

    pub fn runtime(&self) -> &'static Runtime {
        self.0.runtime
    }

    /// Make this device current on the calling thread.
    pub fn set_current(&self) -> Result<(), Error> {
        self.0.runtime.check(
            unsafe { (self.0.runtime.hip_set_device)(self.0.device) },
            "hipSetDevice",
        )
    }

    /// The lock, unless this thread is the one capturing.
    fn hold(&self) -> Option<MutexGuard<'_, ()>> {
        match CAPTURING.with(Cell::get).is_null() {
            true => Some(self.0.exclusive.lock().unwrap_or_else(|e| e.into_inner())),
            false => None,
        }
    }

    /// Waits for the null stream, where all plain work runs.
    pub fn synchronize(&self) -> Result<(), Error> {
        let _hold = self.hold();
        self.set_current()?;
        self.0.runtime.check(
            unsafe { (self.0.runtime.hip_stream_synchronize)(std::ptr::null_mut()) },
            "hipStreamSynchronize",
        )
    }

    /// Runs `record` and keeps its launches as one graph.
    pub fn capture<E: From<Error>>(
        &self,
        record: impl FnOnce() -> Result<(), E>,
    ) -> Result<Graph, E> {
        let _hold = self.hold();
        self.set_current()?;
        let r = self.0.runtime;
        let mut stream: HipStream = std::ptr::null_mut();
        r.check(
            unsafe { (r.hip_stream_create)(&mut stream, hip::HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        let mut graph = Graph {
            runtime: r,
            stream,
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
        };
        let mode = hip::HIP_STREAM_CAPTURE_MODE_RELAXED;
        r.check(
            unsafe { (r.hip_stream_begin_capture)(stream, mode) },
            "hipStreamBeginCapture",
        )?;
        CAPTURING.with(|c| c.set(stream));
        let recorded = record();
        CAPTURING.with(|c| c.set(std::ptr::null_mut()));
        // The capture ends either way, or the stream stays stuck.
        let ended = r.check(
            unsafe { (r.hip_stream_end_capture)(stream, &mut graph.graph) },
            "hipStreamEndCapture",
        );
        recorded.and(ended.map_err(E::from))?;
        let (none, nothing) = (std::ptr::null_mut(), std::ptr::null_mut());
        r.check(
            unsafe { (r.hip_graph_instantiate)(&mut graph.exec, graph.graph, none, nothing, 0) },
            "hipGraphInstantiate",
        )?;
        Ok(graph)
    }

    /// Runs a captured graph on the null stream, after copies.
    pub fn replay(&self, graph: &Graph) -> Result<(), Error> {
        self.set_current()?;
        let r = self.0.runtime;
        r.check(
            unsafe { (r.hip_graph_launch)(graph.exec, std::ptr::null_mut()) },
            "hipGraphLaunch",
        )
    }

    /// Load a code object, reusing identical images.
    pub fn load_module(&self, image: &[u8]) -> Result<Arc<Module>, Error> {
        let key = image_hash(image);
        if let Some(existing) = self.0.modules.lock().unwrap().get(&key) {
            return Ok(Arc::clone(existing));
        }
        self.set_current()?;
        let mut handle: HipModule = std::ptr::null_mut();
        self.0.runtime.check(
            unsafe { (self.0.runtime.hip_module_load_data)(&mut handle, image.as_ptr().cast()) },
            "hipModuleLoadData",
        )?;
        let module = Arc::new(Module {
            runtime: self.0.runtime,
            raw: handle,
        });
        self.0
            .modules
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Compile with hiprtc for this device and resolve `entry`.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<Kernel, Error> {
        self.set_current()?;
        let image = super::hiprtc::compile_to_code(source)?;
        self.load_module(&image)?.function(entry)
    }
}

impl Launch for Context {
    type Kernel = Kernel;

    fn launch(
        &self,
        kernel: &Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        let _hold = self.hold();
        self.set_current()?;
        let mut params = args.pointers();
        self.0.runtime.check(
            unsafe {
                (self.0.runtime.hip_module_launch_kernel)(
                    kernel.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    CAPTURING.with(Cell::get),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "hipModuleLaunchKernel",
        )
    }
}

/// A loaded HIP module, unloaded when its last kernel drops.
pub struct Module {
    runtime: &'static Runtime,
    raw: HipModule,
}

impl Module {
    pub fn function(self: &Arc<Self>, name: &str) -> Result<Kernel, Error> {
        let c_name = CString::new(name)
            .map_err(|e| Error::Rocm(format!("kernel name {name:?} has a NUL: {e}")))?;
        let mut raw: HipFunction = std::ptr::null_mut();
        self.runtime
            .check(
                unsafe {
                    (self.runtime.hip_module_get_function)(&mut raw, self.raw, c_name.as_ptr())
                },
                "hipModuleGetFunction",
            )
            .map_err(|e| match e {
                // The bare runtime error does not name the entry.
                Error::Rocm(msg) => Error::Rocm(format!("{msg} for entry {name:?}")),
                other => other,
            })?;
        Ok(Kernel {
            _module: Arc::clone(self),
            raw,
            name: name.to_string(),
        })
    }
}

// The HIP runtime is thread-safe. Calls select the device first.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            (self.runtime.hip_module_unload)(self.raw);
        }
    }
}

/// A resolved kernel entry point that keeps its module alive.
#[derive(Clone)]
pub struct Kernel {
    _module: Arc<Module>,
    raw: HipFunction,
    name: String,
}

thread_local! {
    /// This thread's capture stream, or null outside a capture.
    static CAPTURING: Cell<*mut c_void> = const { Cell::new(std::ptr::null_mut()) };
}

/// Launches recorded once, replayed on their own stream.
pub struct Graph {
    runtime: &'static Runtime,
    stream: HipStream,
    graph: HipGraph,
    exec: HipGraphExec,
}

unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                (self.runtime.hip_graph_exec_destroy)(self.exec);
            }
            if !self.graph.is_null() {
                (self.runtime.hip_graph_destroy)(self.graph);
            }
            (self.runtime.hip_stream_destroy)(self.stream);
        }
    }
}

// Thread-safe like `Module`, and the Arc keeps the module alive.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Device for Context {
    type Kernel = Kernel;

    fn open() -> Result<Self, Error> {
        Context::new(ContextConfig::default())
    }

    fn error(msg: String) -> Error {
        Error::Rocm(msg)
    }

    fn caches() -> &'static Caches<Self> {
        static CACHES: Caches<Context> = Caches::new();
        &CACHES
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        let _hold = self.hold();
        self.set_current()?;
        let mut ptr = 0;
        let rt = self.runtime();
        rt.check(unsafe { (rt.hip_malloc)(&mut ptr, size) }, "hipMalloc")?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        let _hold = self.hold();
        // At process exit the runtime may already be gone.
        if self.set_current().is_ok() {
            unsafe { (self.runtime().hip_free)(ptr) };
        }
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let _hold = self.hold();
        self.set_current()?;
        let rt = self.runtime();
        let (dst, src) = (ptr as *mut _, bytes.as_ptr().cast());
        let code = unsafe { (rt.hip_memcpy)(dst, src, bytes.len(), hip::MEMCPY_HOST_TO_DEVICE) };
        rt.check(code, "hipMemcpy to device")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let _hold = self.hold();
        self.set_current()?;
        let rt = self.runtime();
        let (dst, src) = (bytes.as_mut_ptr().cast(), ptr as *const _);
        let code = unsafe { (rt.hip_memcpy)(dst, src, bytes.len(), hip::MEMCPY_DEVICE_TO_HOST) };
        rt.check(code, "hipMemcpy to host")
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        let _hold = self.hold();
        self.set_current()?;
        let rt = self.runtime();
        rt.check(unsafe { (rt.hip_memset)(ptr, 0, size) }, "hipMemset")
    }

    fn load_image(&self, image: &[u8], entry: &str) -> Result<Kernel, Error> {
        self.load_module(image)?.function(entry)
    }

    fn compile(&self, source: &str, entry: &str) -> Result<Kernel, Error> {
        self.compile_kernel(source, entry)
    }
}
