//! The CUDA driver behind the shared context.

use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int};
use std::thread::LocalKey;

use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, CU_STREAM_CAPTURE_MODE_RELAXED,
    CU_STREAM_NON_BLOCKING, CUcontext, CUdevice, Driver, LaunchAttribute, LaunchConfig,
};
use crate::api::{self, Api, Call, Handle, Kind};
use crate::device::Caches;
use crate::{Error, Limits};

pub use crate::api::ContextConfig;

/// A CUDA context plus a cache of loaded modules.
pub type Context = api::Context<Driver>;
pub type Module = api::Module<Driver>;
pub type Kernel = api::Kernel<Driver>;
pub type Graph = api::Graph<Driver>;

/// The driver context and what the device reports.
pub struct Own {
    raw: CUcontext,
    compute_capability: (i32, i32),
    limits: Limits,
}

fn device_attr(driver: &Driver, device: CUdevice, attr: c_int) -> Result<i32, Error> {
    let mut value: c_int = 0;
    driver.check(
        unsafe { (driver.cu_device_get_attribute)(&mut value, attr, device) },
        "cuDeviceGetAttribute",
    )?;
    Ok(value)
}

impl Context {
    /// Initialize the driver and open a context on one device.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let driver = driver::driver()?;
        driver.check(unsafe { (driver.cu_init)(0) }, "cuInit")?;

        let mut count: c_int = 0;
        driver.check(
            unsafe { (driver.cu_device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if config.device_ordinal >= count as usize {
            return Err(Error::Cuda(format!(
                "device ordinal {} out of range ({count} visible CUDA devices)",
                config.device_ordinal
            )));
        }

        let mut device: CUdevice = 0;
        driver.check(
            unsafe { (driver.cu_device_get)(&mut device, config.device_ordinal as c_int) },
            "cuDeviceGet",
        )?;

        let mut raw: CUcontext = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.cu_ctx_create)(&mut raw, 0, device) },
            "cuCtxCreate",
        )?;

        let mut name_buf = [0 as c_char; 256];
        driver.check(
            unsafe { (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, device) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let major = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let limits = Limits {
            lanes: device_attr(driver, device, CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as usize,
            group_bytes: device_attr(
                driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )? as usize,
            group_threads: device_attr(driver, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as usize,
            compiler: true,
            // Hopper on: griddepcontrol and the launch attribute.
            pdl: major >= 9,
        };
        let own = Own {
            raw,
            compute_capability: (major, minor),
            limits,
        };
        Ok(Self::made(driver, own, device, name))
    }

    /// Lanes, group memory and group size, from the device.
    pub fn limits(&self) -> Limits {
        self.own().limits
    }

    /// (major, minor), e.g. (12, 1) for sm_121.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.own().compute_capability
    }

    pub fn driver(&self) -> &'static Driver {
        self.api()
    }
}

thread_local! {
    /// This thread's capture stream, or null outside a capture.
    static CAPTURING: Cell<Handle> = const { Cell::new(std::ptr::null_mut()) };
    /// Whether this thread's next launches may overlap the last.
    static PROGRAMMATIC: Cell<bool> = const { Cell::new(false) };
}

impl Driver {
    /// A launch that may overlap the kernel before it.
    fn overlap(&self, call: &Call) -> Result<(), Error> {
        let mut attr = LaunchAttribute {
            id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: 0,
            value: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        let config = LaunchConfig {
            grid: call.grid,
            block: call.block,
            shared_bytes: call.shared_bytes,
            pad: 0,
            stream: call.stream,
            attrs: &mut attr,
            attr_count: 1,
            pad2: 0,
        };
        let null = std::ptr::null_mut();
        let code = unsafe { (self.cu_launch_kernel_ex)(&config, call.kernel, call.params, null) };
        self.check(code, "cuLaunchKernelEx")
    }
}

impl Api for Driver {
    type Own = Own;

    fn error(msg: String) -> Error {
        Error::Cuda(msg)
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

    fn programmatic(on: bool) {
        PROGRAMMATIC.with(|p| p.set(on));
    }

    fn close(&self, own: &Own) {
        unsafe { (self.cu_ctx_destroy)(own.raw) };
    }

    fn check(&self, code: c_int, what: &str) -> Result<(), Error> {
        Driver::check(self, code, what)
    }

    fn set_current(&self, own: &Own, _: c_int) -> Result<(), Error> {
        let code = unsafe { (self.cu_ctx_set_current)(own.raw) };
        self.check(code, "cuCtxSetCurrent")
    }

    fn synchronize(&self) -> Result<(), Error> {
        let code = unsafe { (self.cu_stream_synchronize)(std::ptr::null_mut()) };
        self.check(code, "cuStreamSynchronize")
    }

    fn stream(&self) -> Result<Handle, Error> {
        self.out("cuStreamCreate", |s| unsafe {
            (self.cu_stream_create)(s, CU_STREAM_NON_BLOCKING)
        })
    }

    fn begin_capture(&self, stream: Handle) -> Result<(), Error> {
        let mode = CU_STREAM_CAPTURE_MODE_RELAXED;
        let code = unsafe { (self.cu_stream_begin_capture)(stream, mode) };
        self.check(code, "cuStreamBeginCapture")
    }

    fn end_capture(&self, stream: Handle, graph: &mut Handle) -> Result<(), Error> {
        let code = unsafe { (self.cu_stream_end_capture)(stream, graph) };
        self.check(code, "cuStreamEndCapture")
    }

    fn instantiate(&self, graph: Handle) -> Result<Handle, Error> {
        self.out("cuGraphInstantiate", |e| unsafe {
            (self.cu_graph_instantiate)(e, graph, 0)
        })
    }

    fn replay(&self, exec: Handle) -> Result<(), Error> {
        let code = unsafe { (self.cu_graph_launch)(exec, std::ptr::null_mut()) };
        self.check(code, "cuGraphLaunch")
    }

    fn destroy(&self, kind: Kind, handle: Handle) {
        let free = match kind {
            Kind::Module => self.cu_module_unload,
            Kind::Exec => self.cu_graph_exec_destroy,
            Kind::Graph => self.cu_graph_destroy,
            Kind::Stream => self.cu_stream_destroy,
        };
        unsafe { free(handle) };
    }

    fn load(&self, image: &[u8]) -> Result<Handle, Error> {
        self.out("cuModuleLoadData", |m| unsafe {
            (self.cu_module_load_data)(m, image.as_ptr().cast())
        })
    }

    fn function(&self, module: Handle, name: &CStr) -> Result<Handle, Error> {
        self.out("cuModuleGetFunction", |f| unsafe {
            (self.cu_module_get_function)(f, module, name.as_ptr())
        })
    }

    fn launch(&self, call: &Call) -> Result<(), Error> {
        if PROGRAMMATIC.with(Cell::get) {
            return self.overlap(call);
        }
        let code = unsafe { api::plain(self.cu_launch_kernel, call) };
        self.check(code, "cuLaunchKernel")
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        let mut ptr = 0;
        self.check(unsafe { (self.cu_mem_alloc)(&mut ptr, size) }, "cuMemAlloc")?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        unsafe { (self.cu_mem_free)(ptr) };
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_htod)(ptr, bytes.as_ptr().cast(), bytes.len()) };
        self.check(code, "cuMemcpyHtoD")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), ptr, bytes.len()) };
        self.check(code, "cuMemcpyDtoH")
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.check(unsafe { (self.cu_memset_d8)(ptr, 0, size) }, "cuMemsetD8")
    }

    fn compile(ctx: &Context, source: &str) -> Result<Vec<u8>, Error> {
        let (major, minor) = ctx.compute_capability();
        super::nvrtc::compile_to_ptx(source, major, minor)
    }
}
