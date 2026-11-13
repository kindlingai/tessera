//! CUDA driver bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::dl::Library;
use std::ffi::{CStr, c_char, c_int, c_void};
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
pub const CU_STREAM_NON_BLOCKING: u32 = 1;
/// Other threads may still synchronize while a capture is open.
pub const CU_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

const DRIVER_LIB: &str = "libcuda.so.1";

type CuInit = unsafe extern "C" fn(u32) -> CUresult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
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

/// Resolved driver entry points. The dlopen handle stays open.
pub struct Driver {
    pub cu_init: CuInit,
    pub cu_device_get_count: CuDeviceGetCount,
    pub cu_device_get: CuDeviceGet,
    pub cu_device_get_name: CuDeviceGetName,
    pub cu_device_get_attribute: CuDeviceGetAttribute,
    pub cu_ctx_create: CuCtxCreate,
    pub cu_ctx_destroy: CuCtxDestroy,
    pub cu_ctx_set_current: CuCtxSetCurrent,
    pub cu_ctx_synchronize: CuCtxSynchronize,
    pub cu_module_load_data: CuModuleLoadData,
    pub cu_module_unload: CuModuleUnload,
    pub cu_module_get_function: CuModuleGetFunction,
    pub cu_mem_alloc: CuMemAlloc,
    pub cu_mem_free: CuMemFree,
    pub cu_memcpy_htod: CuMemcpyHtoD,
    pub cu_memcpy_dtoh: CuMemcpyDtoH,
    pub cu_memcpy_dtod: CuMemcpyDtoD,
    pub cu_memset_d8: CuMemsetD8,
    pub cu_launch_kernel: CuLaunchKernel,
    pub cu_stream_create: CuStreamCreate,
    pub cu_stream_destroy: CuStreamDestroy,
    pub cu_stream_synchronize: CuStreamSynchronize,
    pub cu_stream_begin_capture: CuStreamBeginCapture,
    pub cu_stream_end_capture: CuStreamEndCapture,
    pub cu_graph_instantiate: CuGraphInstantiate,
    pub cu_graph_launch: CuGraphLaunch,
    pub cu_graph_exec_destroy: CuGraphExecDestroy,
    pub cu_graph_destroy: CuGraphDestroy,
    cu_get_error_name: CuGetErrorName,
    cu_get_error_string: CuGetErrorString,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        unsafe { $lib.function::<$ty>($name) }.map_err(|e| format!("CUDA driver: {e}"))?
    };
}

impl Driver {
    fn open() -> Result<Self, String> {
        let lib =
            Library::open_first(&[DRIVER_LIB.into()]).map_err(|e| format!("cannot load {e}"))?;
        Ok(Driver {
            cu_init: sym!(lib, "cuInit", CuInit),
            cu_device_get_count: sym!(lib, "cuDeviceGetCount", CuDeviceGetCount),
            cu_device_get: sym!(lib, "cuDeviceGet", CuDeviceGet),
            cu_device_get_name: sym!(lib, "cuDeviceGetName", CuDeviceGetName),
            cu_device_get_attribute: sym!(lib, "cuDeviceGetAttribute", CuDeviceGetAttribute),
            cu_ctx_create: sym!(lib, "cuCtxCreate_v2", CuCtxCreate),
            cu_ctx_destroy: sym!(lib, "cuCtxDestroy_v2", CuCtxDestroy),
            cu_ctx_set_current: sym!(lib, "cuCtxSetCurrent", CuCtxSetCurrent),
            cu_ctx_synchronize: sym!(lib, "cuCtxSynchronize", CuCtxSynchronize),
            cu_module_load_data: sym!(lib, "cuModuleLoadData", CuModuleLoadData),
            cu_module_unload: sym!(lib, "cuModuleUnload", CuModuleUnload),
            cu_module_get_function: sym!(lib, "cuModuleGetFunction", CuModuleGetFunction),
            cu_mem_alloc: sym!(lib, "cuMemAlloc_v2", CuMemAlloc),
            cu_mem_free: sym!(lib, "cuMemFree_v2", CuMemFree),
            cu_memcpy_htod: sym!(lib, "cuMemcpyHtoD_v2", CuMemcpyHtoD),
            cu_memcpy_dtoh: sym!(lib, "cuMemcpyDtoH_v2", CuMemcpyDtoH),
            cu_memcpy_dtod: sym!(lib, "cuMemcpyDtoD_v2", CuMemcpyDtoD),
            cu_memset_d8: sym!(lib, "cuMemsetD8_v2", CuMemsetD8),
            cu_launch_kernel: sym!(lib, "cuLaunchKernel", CuLaunchKernel),
            cu_stream_create: sym!(lib, "cuStreamCreate", CuStreamCreate),
            cu_stream_destroy: sym!(lib, "cuStreamDestroy_v2", CuStreamDestroy),
            cu_stream_synchronize: sym!(lib, "cuStreamSynchronize", CuStreamSynchronize),
            cu_stream_begin_capture: sym!(lib, "cuStreamBeginCapture_v2", CuStreamBeginCapture),
            cu_stream_end_capture: sym!(lib, "cuStreamEndCapture", CuStreamEndCapture),
            cu_graph_instantiate: sym!(lib, "cuGraphInstantiateWithFlags", CuGraphInstantiate),
            cu_graph_launch: sym!(lib, "cuGraphLaunch", CuGraphLaunch),
            cu_graph_exec_destroy: sym!(lib, "cuGraphExecDestroy", CuGraphExecDestroy),
            cu_graph_destroy: sym!(lib, "cuGraphDestroy", CuGraphDestroy),
            cu_get_error_name: sym!(lib, "cuGetErrorName", CuGetErrorName),
            cu_get_error_string: sym!(lib, "cuGetErrorString", CuGetErrorString),
        })
    }

    /// The name of a driver return code.
    pub fn error_name(&self, code: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_name)(code, &mut p) };
        if p.is_null() {
            return format!("CUDA error {code}");
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Turn a failed return code into an error naming `what`.
    pub fn check(&self, code: CUresult, what: &str) -> Result<(), Error> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_string)(code, &mut p) };
        let msg = if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(Error::Cuda(format!(
            "{what} failed: {} ({code}) {msg}",
            self.error_name(code)
        )))
    }
}

/// The process-wide driver, opened on first use.
pub fn driver() -> Result<&'static Driver, Error> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    match DRIVER.get_or_init(Driver::open) {
        Ok(d) => Ok(d),
        Err(msg) => Err(Error::Cuda(format!("CUDA driver unavailable: {msg}"))),
    }
}
