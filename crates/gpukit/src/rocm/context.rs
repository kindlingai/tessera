//! HIP device context, module cache, and kernel handles.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::{Arc, Mutex};

use super::KernelArgs;
use super::hip::{self, HipFunction, HipModule, Runtime};
use crate::Error;
use crate::dl::image_hash;
use crate::launch::Launch;

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
        }));
        ctx.set_current()?;
        Ok(ctx)
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

    pub fn synchronize(&self) -> Result<(), Error> {
        self.set_current()?;
        self.0.runtime.check(
            unsafe { (self.0.runtime.hip_device_synchronize)() },
            "hipDeviceSynchronize",
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
                    std::ptr::null_mut(),
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

// Thread-safe like `Module`, and the Arc keeps the module alive.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }
}
