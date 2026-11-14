//! CUDA driver and NVRTC bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::dl::{message, once};
use crate::rtc::Rtc;
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const CU_STREAM_NON_BLOCKING: u32 = 1;
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 1;

/// `CUlaunchAttribute`: an id, then a 64-byte value.
#[repr(C)]
pub struct LaunchAttribute {
    pub id: u32,
    pub pad: u32,
    pub value: [u64; 8],
}

/// `CUlaunchConfig`, as the driver lays it out.
#[repr(C)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_bytes: u32,
    pub pad: u32,
    pub stream: CUstream,
    pub attrs: *mut LaunchAttribute,
    pub attr_count: u32,
    pub pad2: u32,
}
/// Other threads may still synchronize while a capture is open.
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

type CuInit = unsafe extern "C" fn(u32) -> CUresult;
type CuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type CuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type CuLaunchKernelEx = unsafe extern "C" fn(
    *const LaunchConfig,
    CUfunction,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type CuStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type CuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type CuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type CuStreamBeginCapture = unsafe extern "C" fn(CUstream, c_int) -> CUresult;
type CuStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type CuGraphInstantiate = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type CuGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type CuGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
type CuGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
type CuGetErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

crate::dl::library! {
    /// Resolved driver entry points. The dlopen handle stays open.
    pub struct Driver as "CUDA driver" in ["libcuda.so.1"] {
        pub cu_init: CuInit = "cuInit",
        pub cu_driver_get_version: CuDriverGetVersion = "cuDriverGetVersion",
        pub cu_device_get_count: CuDeviceGetCount = "cuDeviceGetCount",
        pub cu_device_get: CuDeviceGet = "cuDeviceGet",
        pub cu_device_get_name: CuDeviceGetName = "cuDeviceGetName",
        pub cu_device_get_attribute: CuDeviceGetAttribute = "cuDeviceGetAttribute",
        pub cu_ctx_create: CuCtxCreate = "cuCtxCreate_v2",
        pub cu_ctx_destroy: CuCtxDestroy = "cuCtxDestroy_v2",
        pub cu_ctx_set_current: CuCtxSetCurrent = "cuCtxSetCurrent",
        pub cu_module_load_data: CuModuleLoadData = "cuModuleLoadData",
        pub cu_module_unload: CuModuleUnload = "cuModuleUnload",
        pub cu_module_get_function: CuModuleGetFunction = "cuModuleGetFunction",
        pub cu_mem_alloc: CuMemAlloc = "cuMemAlloc_v2",
        pub cu_mem_free: CuMemFree = "cuMemFree_v2",
        pub cu_memcpy_htod: CuMemcpyHtoD = "cuMemcpyHtoD_v2",
        pub cu_memcpy_dtoh: CuMemcpyDtoH = "cuMemcpyDtoH_v2",
        pub cu_memset_d8: CuMemsetD8 = "cuMemsetD8_v2",
        pub cu_launch_kernel: CuLaunchKernel = "cuLaunchKernel",
        pub cu_launch_kernel_ex: CuLaunchKernelEx = "cuLaunchKernelEx",
        pub cu_stream_create: CuStreamCreate = "cuStreamCreate",
        pub cu_stream_destroy: CuStreamDestroy = "cuStreamDestroy_v2",
        pub cu_stream_synchronize: CuStreamSynchronize = "cuStreamSynchronize",
        pub cu_stream_begin_capture: CuStreamBeginCapture = "cuStreamBeginCapture_v2",
        pub cu_stream_end_capture: CuStreamEndCapture = "cuStreamEndCapture",
        pub cu_graph_instantiate: CuGraphInstantiate = "cuGraphInstantiateWithFlags",
        pub cu_graph_launch: CuGraphLaunch = "cuGraphLaunch",
        pub cu_graph_exec_destroy: CuGraphExecDestroy = "cuGraphExecDestroy",
        pub cu_graph_destroy: CuGraphDestroy = "cuGraphDestroy",
        cu_get_error_name: CuGetErrorName = "cuGetErrorName",
        cu_get_error_string: CuGetErrorString = "cuGetErrorString",
    }
}

impl Driver {
    /// The name of a driver return code.
    pub fn error_name(&self, code: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_name)(code, &mut p) };
        if p.is_null() {
            return format!("CUDA error {code}");
        }
        message(p)
    }

    /// Turn a failed return code into an error naming `what`.
    pub fn check(&self, code: CUresult, what: &str) -> Result<(), Error> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_string)(code, &mut p) };
        let msg = message(p);
        Err(Error::Cuda(format!(
            "{what} failed: {} ({code}) {msg}",
            self.error_name(code)
        )))
    }
}

/// The process-wide driver, opened on first use.
pub fn driver() -> Result<&'static Driver, Error> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    once(&DRIVER, Driver::open, |msg| {
        Error::Cuda(format!("CUDA driver unavailable: {msg}"))
    })
}

/// NVRTC ships with the CUDA toolkit.
pub(crate) static NVRTC: Rtc = Rtc {
    name: "NVRTC",
    prefix: "nvrtc",
    code: "PTX",
    libraries: &[
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "/usr/local/cuda/lib64/libnvrtc.so",
    ],
    install: "install the CUDA toolkit",
    error: Error::Cuda,
    lib: OnceLock::new(),
};
