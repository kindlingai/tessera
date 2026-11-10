//! CUDA context, module cache, and kernel handles.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::{Arc, Mutex};

use super::KernelArgs;
use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUcontext, CUdevice, CUfunction, CUmodule,
    Driver,
};
use crate::Error;
use crate::device::{Caches, Device};
use crate::dl::image_hash;
use crate::launch::Launch;

/// How to open the CUDA context.
#[derive(Debug, Clone, Default)]
pub struct ContextConfig {
    /// Zero-based device ordinal (CUDA_VISIBLE_DEVICES still applies).
    pub device_ordinal: usize,
}

/// A CUDA context plus a cache of loaded modules.
#[derive(Clone)]
pub struct Context(Arc<Inner>);

struct Inner {
    driver: &'static Driver,
    raw: CUcontext,
    device: CUdevice,
    name: String,
    compute_capability: (i32, i32),
    modules: Mutex<HashMap<u64, Arc<Module>>>,
}

// The driver API is thread-safe. Calls set the context first.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // Modules must unload before their context goes.
        if let Ok(mut modules) = self.modules.lock() {
            modules.clear();
        }
        unsafe {
            (self.driver.cu_ctx_destroy)(self.raw);
        }
    }
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

        Ok(Self(Arc::new(Inner {
            driver,
            raw,
            device,
            name,
            compute_capability: (major, minor),
            modules: Mutex::new(HashMap::new()),
        })))
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// (major, minor), e.g. (12, 1) for sm_121.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.0.compute_capability
    }

    pub fn device(&self) -> CUdevice {
        self.0.device
    }

    pub fn driver(&self) -> &'static Driver {
        self.0.driver
    }

    /// Make this context current on the calling thread.
    pub fn set_current(&self) -> Result<(), Error> {
        self.0.driver.check(
            unsafe { (self.0.driver.cu_ctx_set_current)(self.0.raw) },
            "cuCtxSetCurrent",
        )
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        self.set_current()?;
        self.0.driver.check(
            unsafe { (self.0.driver.cu_ctx_synchronize)() },
            "cuCtxSynchronize",
        )
    }

    /// Load an ELF cubin or NUL-terminated PTX, reusing identical images.
    pub fn load_module(&self, image: &[u8]) -> Result<Arc<Module>, Error> {
        let key = image_hash(image);
        if let Some(existing) = self.0.modules.lock().unwrap().get(&key) {
            return Ok(Arc::clone(existing));
        }
        self.set_current()?;
        let mut handle: CUmodule = std::ptr::null_mut();
        self.0.driver.check(
            unsafe { (self.0.driver.cu_module_load_data)(&mut handle, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let module = Arc::new(Module {
            driver: self.0.driver,
            raw: handle,
        });
        self.0
            .modules
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Compile with NVRTC for this device and resolve `entry`.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<Kernel, Error> {
        let (major, minor) = self.compute_capability();
        let ptx = super::nvrtc::compile_to_ptx(source, major, minor)?;
        self.load_module(&ptx)?.function(entry)
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
        self.set_current()?;
        let mut params = args.pointers();
        self.0.driver.check(
            unsafe {
                (self.0.driver.cu_launch_kernel)(
                    kernel.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

/// A loaded CUDA module, unloaded when its last kernel drops.
pub struct Module {
    driver: &'static Driver,
    raw: CUmodule,
}

impl Module {
    pub fn function(self: &Arc<Self>, name: &str) -> Result<Kernel, Error> {
        let c_name = CString::new(name)
            .map_err(|e| Error::Cuda(format!("kernel name {name:?} has a NUL: {e}")))?;
        let mut raw: CUfunction = std::ptr::null_mut();
        self.driver
            .check(
                unsafe {
                    (self.driver.cu_module_get_function)(&mut raw, self.raw, c_name.as_ptr())
                },
                "cuModuleGetFunction",
            )
            .map_err(|e| match e {
                // The bare driver error does not name the entry.
                Error::Cuda(msg) => Error::Cuda(format!("{msg} for entry {name:?}")),
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
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            (self.driver.cu_module_unload)(self.raw);
        }
    }
}

/// A resolved kernel entry point that keeps its module alive.
#[derive(Clone)]
pub struct Kernel {
    /// Holding it keeps the module loaded.
    _module: Arc<Module>,
    raw: CUfunction,
    name: String,
}

// Thread-safe like `Inner`, and the Arc keeps the module alive.
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
        Error::Cuda(msg)
    }

    fn caches() -> &'static Caches<Self> {
        static CACHES: Caches<Context> = Caches::new();
        &CACHES
    }

    fn alloc(&self, size: usize) -> Result<u64, Error> {
        self.set_current()?;
        let mut ptr = 0;
        let d = self.driver();
        d.check(unsafe { (d.cu_mem_alloc)(&mut ptr, size) }, "cuMemAlloc")?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) {
        // At process exit the context may already be gone.
        if self.set_current().is_ok() {
            unsafe { (self.driver().cu_mem_free)(ptr) };
        }
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        self.set_current()?;
        let d = self.driver();
        let code = unsafe { (d.cu_memcpy_htod)(ptr, bytes.as_ptr().cast(), bytes.len()) };
        d.check(code, "cuMemcpyHtoD")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.set_current()?;
        let d = self.driver();
        let code = unsafe { (d.cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), ptr, bytes.len()) };
        d.check(code, "cuMemcpyDtoH")
    }

    fn zero(&self, ptr: u64, size: usize) -> Result<(), Error> {
        self.set_current()?;
        let d = self.driver();
        d.check(unsafe { (d.cu_memset_d8)(ptr, 0, size) }, "cuMemsetD8")
    }

    fn load_image(&self, image: &[u8], entry: &str) -> Result<Kernel, Error> {
        self.load_module(image)?.function(entry)
    }

    fn compile(&self, source: &str, entry: &str) -> Result<Kernel, Error> {
        self.compile_kernel(source, entry)
    }
}
