//! The HIP runtime behind the shared context.

use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int};
use std::sync::OnceLock;
use std::thread::LocalKey;

use super::hip::{self, Runtime};
use crate::api::{self, Api, Call, Handle, Kind};
use crate::device::{Caches, Device};
use crate::{Error, Limits};

pub use crate::api::ContextConfig;

/// A HIP device plus a cache of loaded modules.
pub type Context = api::Context<Runtime>;
pub type Module = api::Module<Runtime>;
pub type Kernel = api::Kernel<Runtime>;
pub type Graph = api::Graph<Runtime>;

/// `warpSize`, asked once.
pub struct Own {
    lanes: OnceLock<usize>,
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

        let own = Own {
            lanes: OnceLock::new(),
        };
        let ctx = Self::made(runtime, own, device, name);
        ctx.set_current()?;
        Ok(ctx)
    }

    /// Lanes come from the device. The rest are AMD constants.
    pub fn limits(&self) -> Limits {
        let lanes = *self
            .own()
            .lanes
            .get_or_init(|| self.probe_lanes().unwrap_or(64));
        Limits {
            lanes,
            group_bytes: 64 * 1024,
            group_threads: 1024,
            compiler: true,
            pdl: false,
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

    pub fn runtime(&self) -> &'static Runtime {
        self.api()
    }
}

thread_local! {
    /// This thread's capture stream, or null outside a capture.
    static CAPTURING: Cell<Handle> = const { Cell::new(std::ptr::null_mut()) };
}

impl Api for Runtime {
    type Own = Own;

    fn error(msg: String) -> Error {
        Error::Rocm(msg)
    }

    fn open() -> Result<Context, Error> {
        Context::new(ContextConfig::default())
    }

    fn caches() -> &'static Caches<Context> {
        static CACHES: Caches<Context> = Caches::new();
        &CACHES
    }

    fn capturing() -> &'static LocalKey<Cell<Handle>> {
        &CAPTURING
    }

    fn check(&self, code: c_int, what: &str) -> Result<(), Error> {
        Runtime::check(self, code, what)
    }

    fn set_current(&self, _: &Own, device: c_int) -> Result<(), Error> {
        self.check(unsafe { (self.hip_set_device)(device) }, "hipSetDevice")
    }

    fn synchronize(&self) -> Result<(), Error> {
        let code = unsafe { (self.hip_stream_synchronize)(std::ptr::null_mut()) };
        self.check(code, "hipStreamSynchronize")
    }

    fn stream(&self) -> Result<Handle, Error> {
        let flags = hip::HIP_STREAM_NON_BLOCKING;
        self.out("hipStreamCreateWithFlags", |s| unsafe {
            (self.hip_stream_create)(s, flags)
        })
    }

    fn begin_capture(&self, stream: Handle) -> Result<(), Error> {
        let mode = hip::HIP_STREAM_CAPTURE_MODE_RELAXED;
        let code = unsafe { (self.hip_stream_begin_capture)(stream, mode) };
        self.check(code, "hipStreamBeginCapture")
    }

    fn end_capture(&self, stream: Handle, graph: &mut Handle) -> Result<(), Error> {
        let code = unsafe { (self.hip_stream_end_capture)(stream, graph) };
        self.check(code, "hipStreamEndCapture")
    }

    fn instantiate(&self, graph: Handle) -> Result<Handle, Error> {
        let (none, nothing) = (std::ptr::null_mut(), std::ptr::null_mut());
        self.out("hipGraphInstantiate", |e| unsafe {
            (self.hip_graph_instantiate)(e, graph, none, nothing, 0)
        })
    }

    fn replay(&self, exec: Handle) -> Result<(), Error> {
        let code = unsafe { (self.hip_graph_launch)(exec, std::ptr::null_mut()) };
        self.check(code, "hipGraphLaunch")
    }

    fn destroy(&self, kind: Kind, handle: Handle) {
        let free = match kind {
            Kind::Module => self.hip_module_unload,
            Kind::Exec => self.hip_graph_exec_destroy,
            Kind::Graph => self.hip_graph_destroy,
            Kind::Stream => self.hip_stream_destroy,
        };
        unsafe { free(handle) };
    }

    fn load(&self, image: &[u8]) -> Result<Handle, Error> {
        self.out("hipModuleLoadData", |m| unsafe {
            (self.hip_module_load_data)(m, image.as_ptr().cast())
        })
    }

    fn function(&self, module: Handle, name: &CStr) -> Result<Handle, Error> {
        self.out("hipModuleGetFunction", |f| unsafe {
            (self.hip_module_get_function)(f, module, name.as_ptr())
        })
    }

    fn launch(&self, call: &Call) -> Result<(), Error> {
        let code = unsafe { api::plain(self.hip_module_launch_kernel, call) };
        self.check(code, "hipModuleLaunchKernel")
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        let mut ptr = 0;
        self.check(unsafe { (self.hip_malloc)(&mut ptr, size) }, "hipMalloc")?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        unsafe { (self.hip_free)(ptr) };
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let (dst, src) = (ptr as *mut _, bytes.as_ptr().cast());
        let to = hip::MEMCPY_HOST_TO_DEVICE;
        let code = unsafe { (self.hip_memcpy)(dst, src, bytes.len(), to) };
        self.check(code, "hipMemcpy to device")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let (dst, src) = (bytes.as_mut_ptr().cast(), ptr as *const _);
        let to = hip::MEMCPY_DEVICE_TO_HOST;
        let code = unsafe { (self.hip_memcpy)(dst, src, bytes.len(), to) };
        self.check(code, "hipMemcpy to host")
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.check(unsafe { (self.hip_memset)(ptr, 0, size) }, "hipMemset")
    }

    fn compile(ctx: &Context, source: &str) -> Result<Vec<u8>, Error> {
        ctx.set_current()?;
        super::hiprtc::compile_to_code(source)
    }
}
