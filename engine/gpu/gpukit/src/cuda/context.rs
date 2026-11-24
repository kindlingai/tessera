//! The CUDA driver behind the shared context.

use std::cell::Cell;
use std::ffi::c_int;

use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_FUNC_ATTRIBUTE_NUM_REGS,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
    CU_STREAM_CAPTURE_MODE_RELAXED, CU_STREAM_NON_BLOCKING, CUcontext, Driver, LaunchAttribute,
    LaunchConfig,
};
use crate::api::{self, Api, Call, Handle};
use crate::backend::Backend;
use crate::cache;
use crate::device::KernelStats;
use crate::platform::{self, Asked, Platform};
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
    platform: Platform,
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
        let limits = Limits {
            lanes: attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as usize,
            group_bytes: attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as usize,
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
            compute_capability: (major, minor),
            // The table says whether launches may overlap.
            limits: platform.limits,
            platform,
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
        events: (cu_event_create, cu_event_record, cu_event_elapsed_time, cu_event_destroy),
        global: cu_module_get_global,
        load: (cu_module_load_data, "cuModuleLoadData"),
        function: (cu_module_get_function, "cuModuleGetFunction"),
        alloc: (cu_mem_alloc, "cuMemAlloc"),
        free: cu_mem_free,
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
        zero: (cu_memset_d8, "cuMemsetD8"),
    }

    fn programmatic(on: bool) {
        PROGRAMMATIC.with(|p| p.set(on));
    }

    fn close(&self, own: &Own) {
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

    fn copy_d2d(&self, dst: u64, src: u64, size: usize) -> Result<(), Error> {
        let stream = std::ptr::null_mut();
        let code = unsafe { (self.cu_memcpy_dtod_async)(dst, src, size, stream) };
        self.check(code, "cuMemcpyDtoDAsync")
    }

    fn stats(&self, kernel: Handle) -> Option<KernelStats> {
        let attr = |which| {
            self.int("cuFuncGetAttribute", |v| unsafe {
                (self.cu_func_get_attribute)(v, which, kernel)
            })
            .ok()
            .map(|v| v.max(0) as u32)
        };
        Some(KernelStats {
            regs: attr(CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            shared_bytes: attr(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
            local_bytes: attr(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            max_threads: attr(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
        })
    }

    fn compile(ctx: &Context, source: &str) -> Result<Vec<u8>, Error> {
        let (major, minor) = ctx.compute_capability();
        let arch = format!("compute_{major}{minor}");
        let gpu_arch = format!("--gpu-architecture={arch}");
        let base = [gpu_arch.as_str(), "--std=c++17", "--diag-suppress=177"];
        cache::current().load_or("cuda", &arch, "ptx", &base, source, |opts| {
            driver::NVRTC.compile(source, opts)
        })
    }
}
