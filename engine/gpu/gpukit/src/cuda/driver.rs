//! CUDA driver and NVRTC bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::api::{LaunchFn, entry};
use crate::dl::once;
use crate::rtc::Rtc;
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;
use tessera_dl::message;

pub type CUresult = c_int;
pub(crate) type CUdevice = c_int;
pub(crate) type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub(crate) type CUfunction = *mut c_void;
pub(crate) type CUstream = *mut c_void;
pub(crate) type CUgraph = *mut c_void;
pub(crate) type CUgraphExec = *mut c_void;

pub(crate) const CUDA_SUCCESS: CUresult = 0;
pub(crate) const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
pub(crate) const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub(crate) const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub(crate) const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub(crate) const CU_DEVICE_ATTRIBUTE_INTEGRATED: c_int = 18;
pub(crate) const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: c_int = 36;
pub(crate) const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub(crate) const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 0;
pub(crate) const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
pub(crate) const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub(crate) const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub(crate) const CU_STREAM_NON_BLOCKING: u32 = 1;
pub(crate) const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
pub(crate) const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 2;
pub(crate) const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 1;
pub(crate) const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 1;
pub(crate) const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

/// `CUlaunchAttribute`: an id, then a 64-byte value.
#[repr(C)]
pub(crate) struct LaunchAttribute {
    pub id: u32,
    pub pad: u32,
    pub value: [u64; 8],
}

/// `CUlaunchConfig`, in the driver's layout.
#[repr(C)]
pub(crate) struct LaunchConfig {
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
pub(crate) const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuMemcpyDtoDAsync =
    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
type CuMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type CuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
type CuLaunchKernelEx = unsafe extern "C" fn(
    *const LaunchConfig,
    CUfunction,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type CuGraphInstantiate = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type CuFuncGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUfunction) -> CUresult;
type CuGetErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

tessera_dl::library! {
    /// Resolved driver entry points. The dlopen handle stays open.
    pub struct Driver as "CUDA driver" in ["libcuda.so.1"] {
        pub(crate) cu_init: entry::Init = "cuInit",
        pub(crate) cu_driver_get_version: entry::Int = "cuDriverGetVersion",
        pub(crate) cu_device_get_count: entry::Int = "cuDeviceGetCount",
        pub(crate) cu_device_get: CuDeviceGet = "cuDeviceGet",
        pub(crate) cu_device_get_name: entry::Name = "cuDeviceGetName",
        pub(crate) cu_device_get_pci_bus_id: entry::Name = "cuDeviceGetPCIBusId",
        pub(crate) cu_device_get_attribute: entry::Attribute = "cuDeviceGetAttribute",
        pub(crate) cu_device_total_mem: entry::Total = "cuDeviceTotalMem_v2",
        pub(crate) cu_ctx_create: CuCtxCreate = "cuCtxCreate_v2",
        pub(crate) cu_ctx_destroy: CuCtxDestroy = "cuCtxDestroy_v2",
        pub(crate) cu_ctx_set_current: CuCtxSetCurrent = "cuCtxSetCurrent",
        pub(crate) cu_module_load_data: entry::Load = "cuModuleLoadData",
        pub(crate) cu_module_unload: entry::OnHandle = "cuModuleUnload",
        pub(crate) cu_module_get_function: entry::Function = "cuModuleGetFunction",
        pub(crate) cu_mem_alloc: entry::Alloc = "cuMemAlloc_v2",
        pub(crate) cu_mem_free: entry::Free = "cuMemFree_v2",
        pub(crate) cu_mem_host_alloc: entry::HostAlloc = "cuMemHostAlloc",
        pub(crate) cu_mem_host_get_device_pointer: entry::HostMap =
            "cuMemHostGetDevicePointer_v2",
        pub(crate) cu_mem_free_host: entry::HostFree = "cuMemFreeHost",
        pub(crate) cu_mem_host_register: entry::Register = "cuMemHostRegister_v2",
        pub(crate) cu_mem_host_unregister: entry::HostFree = "cuMemHostUnregister",
        pub(crate) cu_ipc_get_mem_handle: entry::IpcExport = "cuIpcGetMemHandle",
        pub(crate) cu_ipc_open_mem_handle: entry::IpcOpen = "cuIpcOpenMemHandle_v2",
        pub(crate) cu_ipc_close_mem_handle: entry::IpcClose = "cuIpcCloseMemHandle",
        pub(crate) cu_occupancy: entry::Occupancy =
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        pub(crate) cu_func_set_attribute: CuFuncSetAttribute = "cuFuncSetAttribute",
        pub(crate) cu_func_get_attribute: CuFuncGetAttribute = "cuFuncGetAttribute",
        pub(crate) cu_memcpy_htod: CuMemcpyHtoD = "cuMemcpyHtoD_v2",
        pub(crate) cu_memcpy_dtoh: CuMemcpyDtoH = "cuMemcpyDtoH_v2",
        pub(crate) cu_memcpy_dtod_async: CuMemcpyDtoDAsync = "cuMemcpyDtoDAsync_v2",
        pub(crate) cu_memset_d8: CuMemsetD8 = "cuMemsetD8_v2",
        pub(crate) cu_launch_kernel: LaunchFn = "cuLaunchKernel",
        pub(crate) cu_launch_kernel_ex: CuLaunchKernelEx = "cuLaunchKernelEx",
        pub(crate) cu_stream_create: entry::Stream = "cuStreamCreate",
        pub(crate) cu_stream_destroy: entry::OnHandle = "cuStreamDestroy_v2",
        pub(crate) cu_stream_synchronize: entry::OnHandle = "cuStreamSynchronize",
        pub(crate) cu_stream_begin_capture: entry::Capture = "cuStreamBeginCapture_v2",
        pub(crate) cu_stream_end_capture: entry::EndCapture = "cuStreamEndCapture",
        pub(crate) cu_graph_instantiate: CuGraphInstantiate = "cuGraphInstantiateWithFlags",
        pub(crate) cu_graph_launch: entry::Replay = "cuGraphLaunch",
        pub(crate) cu_graph_exec_destroy: entry::OnHandle = "cuGraphExecDestroy",
        pub(crate) cu_graph_destroy: entry::OnHandle = "cuGraphDestroy",
        pub(crate) cu_event_create: entry::Stream = "cuEventCreate",
        pub(crate) cu_event_record: entry::Replay = "cuEventRecord",
        pub(crate) cu_event_elapsed_time: entry::Elapsed = "cuEventElapsedTime",
        pub(crate) cu_event_destroy: entry::OnHandle = "cuEventDestroy_v2",
        pub(crate) cu_module_get_global: entry::Global = "cuModuleGetGlobal_v2",
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
        unsafe { message(p) }
    }

    /// Turn a failed return code into an error naming `what`.
    pub fn check(&self, code: CUresult, what: &str) -> Result<(), Error> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_string)(code, &mut p) };
        let msg = unsafe { message(p) };
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
