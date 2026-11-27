//! The CUDA driver behind the shared context.

use std::cell::Cell;
use std::ffi::c_int;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, CU_STREAM_CAPTURE_MODE_RELAXED,
    CU_STREAM_NON_BLOCKING, CUcontext, Driver, LaunchAttribute, LaunchConfig,
};
use crate::api::{self, Api, Call, Handle};
use crate::backend::Backend;
use crate::cache;
use crate::platform::{self, Asked, Platform};
use crate::{Error, Limits};

pub use crate::api::ContextConfig;

/// A CUDA context plus a cache of loaded modules.
pub type Context = api::Context<Driver>;
pub type Module = api::Module<Driver>;
pub type Kernel = api::Kernel<Driver>;
pub type Graph = api::Graph<Driver>;

/// Driver context, capabilities and carveout state.
pub struct Own {
    raw: CUcontext,
    device: c_int,
    compute_capability: (i32, i32),
    limits: Limits,
    platform: Platform,
    attic: Mutex<crate::attic::Session>,
    attic_live: AtomicUsize,
}

impl Context {
    /// Initialize the driver and open a context on one device.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let driver = driver::driver()?;
        driver.check(unsafe { (driver.cu_init)(0) }, "cuInit")?;
        let (count, ordinal) = (driver.cu_device_get_count, config.device_ordinal);
        driver.visible(count, "cuDeviceGetCount", ordinal, "CUDA")?;
        let device = driver.int("cuDeviceGet", |d| unsafe {
            (driver.cu_device_get)(d, ordinal as c_int)
        })?;

        let mut raw: CUcontext = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.cu_ctx_create)(&mut raw, 0, device) },
            "cuCtxCreate",
        )?;
        let get_name = driver.cu_device_get_name;
        let name = driver.device_name(get_name, "cuDeviceGetName", device)?;

        let attr = |a| {
            driver.int("cuDeviceGetAttribute", |v| unsafe {
                (driver.cu_device_get_attribute)(v, a, device)
            })
        };
        let major = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let group_bytes = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as usize;
        let dynamic_group_bytes = attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map(|v| v as usize)
            .unwrap_or(group_bytes);
        let limits = Limits {
            lanes: attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as usize,
            group_bytes,
            dynamic_group_bytes,
            group_threads: attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)? as usize,
            compiler: true,
            pdl: false,
        };
        let version = driver.int("cuDriverGetVersion", |v| unsafe {
            (driver.cu_driver_get_version)(v)
        });
        let mut total: usize = 0;
        let asked = unsafe { (driver.cu_device_total_mem)(&mut total, device) };
        driver.check(asked, "cuDeviceTotalMem")?;
        let integrated = attr(CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0;
        let platform = Platform::new(Asked {
            backend: crate::backend::Backend::Cuda.name(),
            arch: format!("sm_{major}{minor}"),
            name: name.clone(),
            // 13000 is `13.0`.
            driver: version.map(|v| format!("{}.{}", v / 1000, (v % 1000) / 10))?,
            limits,
            units: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize,
            bandwidth: platform::bandwidth(
                attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
                attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
            ),
            integrated,
            memory: platform::device_memory(total as u64, integrated),
        });
        let own = Own {
            raw,
            device,
            compute_capability: (major, minor),
            // The table says whether launches may overlap.
            limits: platform.limits,
            platform,
            attic: Mutex::new(crate::attic::Session::default()),
            attic_live: AtomicUsize::new(0),
        };
        let pci = driver
            .device_name(
                driver.cu_device_get_pci_bus_id,
                "cuDeviceGetPCIBusId",
                device,
            )
            .ok();
        let numa = pci.as_deref().and_then(crate::numa::device_node);
        Ok(Self::made(driver, own, device, name, numa))
    }

    /// Lanes, group memory and group size, from the device.
    pub fn limits(&self) -> Limits {
        self.own().limits
    }

    /// The device as asked, with its table row.
    pub fn platform(&self) -> Platform {
        self.own().platform.clone()
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

    const BACKEND: Backend = Backend::Cuda;

    api::shared_api! {
        context: Context,
        error: Error::Cuda,
        check: Driver::check,
        synchronize: (cu_stream_synchronize, "cuStreamSynchronize"),
        begin_capture: (
            cu_stream_begin_capture, "cuStreamBeginCapture", CU_STREAM_CAPTURE_MODE_RELAXED
        ),
        end_capture: (cu_stream_end_capture, "cuStreamEndCapture"),
        replay: (cu_graph_launch, "cuGraphLaunch"),
        unload: (cu_module_unload, cu_graph_exec_destroy, cu_graph_destroy, cu_stream_destroy),
        events: (
            cu_event_create,
            cu_event_record,
            cu_stream_wait_event,
            cu_event_elapsed_time,
            cu_event_destroy
        ),
        global: cu_module_get_global,
        load: (cu_module_load_data, "cuModuleLoadData"),
        function: (cu_module_get_function, "cuModuleGetFunction"),
        alloc: (cu_mem_alloc, "cuMemAlloc"),
        free: cu_mem_free,
        free_memory: (cu_mem_get_info, "cuMemGetInfo"),
        host_alloc: (
            cu_mem_host_alloc, "cuMemHostAlloc", driver::CU_MEMHOSTALLOC_DEVICEMAP,
            cu_mem_host_get_device_pointer, "cuMemHostGetDevicePointer"
        ),
        host_free: cu_mem_free_host,
        register: (
            cu_mem_host_register, "cuMemHostRegister", driver::CU_MEMHOSTREGISTER_DEVICEMAP,
            cu_mem_host_unregister
        ),
        ipc: (
            cu_ipc_get_mem_handle, "cuIpcGetMemHandle",
            cu_ipc_open_mem_handle, "cuIpcOpenMemHandle",
            driver::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            cu_ipc_close_mem_handle
        ),
        occupancy: (cu_occupancy, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
        func_attr: (cu_func_get_attribute, "cuFuncGetAttribute"),
        zero: (cu_memset_d8, "cuMemsetD8"),
    }

    fn programmatic(on: bool) {
        PROGRAMMATIC.with(|p| p.set(on));
    }

    fn alloc_kept(&self, own: &Own, device: c_int, size: usize) -> Result<u64, Error> {
        let cfg = crate::attic::current();
        if cfg.is_some() {
            let vmm = driver::CudaVmm {
                driver: self,
                table: driver::vmm_table(),
                device,
            };
            let mut session = own.attic.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(va) = session.alloc(cfg.as_ref(), &vmm, size) {
                own.attic_live.fetch_add(1, Ordering::Relaxed);
                return Ok(va);
            }
        }
        self.alloc(size)
    }

    fn free_ptr(&self, own: &Own, device: c_int, ptr: u64) {
        if own.attic_live.load(Ordering::Relaxed) > 0 {
            let vmm = driver::CudaVmm {
                driver: self,
                table: driver::vmm_table(),
                device,
            };
            let mut session = own.attic.lock().unwrap_or_else(|e| e.into_inner());
            if session.free(&vmm, ptr) {
                own.attic_live.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
        self.free(ptr);
    }

    fn close(&self, own: &Own) {
        if own.attic_live.load(Ordering::Relaxed) > 0 && self.set_current(own, own.device).is_ok() {
            let vmm = driver::CudaVmm {
                driver: self,
                table: driver::vmm_table(),
                device: own.device,
            };
            let mut session = own.attic.lock().unwrap_or_else(|e| e.into_inner());
            session.release_all(&vmm);
            own.attic_live.store(0, Ordering::Relaxed);
        }
        unsafe { (self.cu_ctx_destroy)(own.raw) };
    }

    fn set_current(&self, own: &Own, _: c_int) -> Result<(), Error> {
        let code = unsafe { (self.cu_ctx_set_current)(own.raw) };
        self.check(code, "cuCtxSetCurrent")
    }

    fn stream(&self) -> Result<Handle, Error> {
        self.out("cuStreamCreate", |s| unsafe {
            (self.cu_stream_create)(s, CU_STREAM_NON_BLOCKING)
        })
    }

    fn stream_high(&self) -> Result<Handle, Error> {
        let (mut least, mut greatest) = (0, 0);
        let rc = unsafe { (self.cu_ctx_stream_priority)(&mut least, &mut greatest) };
        if rc != 0 {
            return self.stream();
        }
        self.out("cuStreamCreateWithPriority", |s| unsafe {
            (self.cu_stream_create_priority)(s, CU_STREAM_NON_BLOCKING, greatest)
        })
    }

    fn instantiate(&self, graph: Handle) -> Result<Handle, Error> {
        self.out("cuGraphInstantiate", |e| unsafe {
            (self.cu_graph_instantiate)(e, graph, 0)
        })
    }

    fn launch(&self, call: &Call) -> Result<(), Error> {
        if PROGRAMMATIC.with(Cell::get) {
            return self.overlap(call);
        }
        let code = unsafe { api::plain(self.cu_launch_kernel, call) };
        self.check(code, "cuLaunchKernel")
    }

    fn allow_shared(&self, kernel: Handle, shared: u32) -> Result<(), Error> {
        let bytes = c_int::try_from(shared)
            .map_err(|_| Error::Cuda(format!("shared bytes {shared} out of range")))?;
        let code = unsafe {
            (self.cu_func_set_attribute)(
                kernel,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        };
        self.check(code, "cuFuncSetAttribute")
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_htod)(ptr, bytes.as_ptr().cast(), bytes.len()) };
        self.check(code, "cuMemcpyHtoD")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), ptr, bytes.len()) };
        self.check(code, "cuMemcpyDtoH")
    }

    fn upload_async(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let stream = std::ptr::null_mut();
        let code =
            unsafe { (self.cu_memcpy_htod_async)(ptr, bytes.as_ptr().cast(), bytes.len(), stream) };
        self.check(code, "cuMemcpyHtoDAsync")
    }

    fn download_async(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let stream = std::ptr::null_mut();
        let code = unsafe {
            (self.cu_memcpy_dtoh_async)(bytes.as_mut_ptr().cast(), ptr, bytes.len(), stream)
        };
        self.check(code, "cuMemcpyDtoHAsync")
    }

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        let stream = std::ptr::null_mut();
        let code = unsafe { (self.cu_memcpy_dtod_async)(dst, src, size, stream) };
        self.check(code, "cuMemcpyDtoDAsync")
    }

    fn compile(ctx: &Context, name: &str, source: &str) -> Result<Vec<u8>, Error> {
        let (major, minor) = ctx.compute_capability();
        compile_ptx(&format!("sm_{major}{minor}"), name, source)
    }
}

/// Normalizes a target string to `compute_XY[a|f]` for NVRTC.
pub(super) fn compute_arch(target: &str) -> Result<String, Error> {
    let (major, minor, suffix) = crate::backend::cuda_arch(target)?;
    Ok(format!("compute_{major}{minor}{suffix}"))
}

/// Reports the loaded NVRTC `(major, minor)` version, if present.
pub fn nvrtc_version() -> Option<(i32, i32)> {
    driver::NVRTC.version()
}

/// Compiles CUDA `source` to PTX without a GPU context.
pub fn compile_ptx(target: &str, name: &str, source: &str) -> Result<Vec<u8>, Error> {
    let arch = compute_arch(target)?;
    let gpu_arch = format!("--gpu-architecture={arch}");
    let base = [gpu_arch.as_str(), "--std=c++17", "--diag-suppress=177"];
    let compiler = driver::NVRTC.version_tag();
    let unit = match name.is_empty() {
        true => "kernel.cu".into(),
        false if name.contains('.') => name.to_string(),
        false => format!("{name}.cu"),
    };
    let key = cache::CacheKey {
        backend: "cuda",
        compiler: &compiler,
        arch: &arch,
        unit: &unit,
        base: &base,
        source,
    };
    cache::current().load_or(key, "ptx", |opts| {
        driver::NVRTC.compile(&unit, source, opts)
    })
}
