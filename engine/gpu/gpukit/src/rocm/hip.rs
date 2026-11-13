//! HIP runtime bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::dl::Library;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

pub type HipError = c_int;
pub type HipDeviceptr = u64;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;
pub type HipGraphNode = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_STREAM_NON_BLOCKING: u32 = 1;
/// Other threads may still synchronize while a capture is open.
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

type HipInit = unsafe extern "C" fn(u32) -> HipError;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipError;
type HipSetDevice = unsafe extern "C" fn(c_int) -> HipError;
type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipError;
type HipDeviceSynchronize = unsafe extern "C" fn() -> HipError;
type HipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError;
type HipModuleUnload = unsafe extern "C" fn(HipModule) -> HipError;
type HipModuleGetFunction =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError;
type HipMalloc = unsafe extern "C" fn(*mut HipDeviceptr, usize) -> HipError;
type HipFree = unsafe extern "C" fn(HipDeviceptr) -> HipError;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipError;
type HipMemset = unsafe extern "C" fn(HipDeviceptr, c_int, usize) -> HipError;
type HipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipError;
type HipStreamCreate = unsafe extern "C" fn(*mut HipStream, u32) -> HipError;
type HipStreamDestroy = unsafe extern "C" fn(HipStream) -> HipError;
type HipStreamSynchronize = unsafe extern "C" fn(HipStream) -> HipError;
type HipStreamBeginCapture = unsafe extern "C" fn(HipStream, c_int) -> HipError;
type HipStreamEndCapture = unsafe extern "C" fn(HipStream, *mut HipGraph) -> HipError;
type HipGraphInstantiate = unsafe extern "C" fn(
    *mut HipGraphExec,
    HipGraph,
    *mut HipGraphNode,
    *mut c_char,
    usize,
) -> HipError;
type HipGraphLaunch = unsafe extern "C" fn(HipGraphExec, HipStream) -> HipError;
type HipGraphExecDestroy = unsafe extern "C" fn(HipGraphExec) -> HipError;
type HipGraphDestroy = unsafe extern "C" fn(HipGraph) -> HipError;
type HipGetErrorName = unsafe extern "C" fn(HipError) -> *const c_char;
type HipGetErrorString = unsafe extern "C" fn(HipError) -> *const c_char;

/// Resolved runtime entry points. The dlopen handle stays open.
pub struct Runtime {
    pub hip_init: HipInit,
    pub hip_get_device_count: HipGetDeviceCount,
    pub hip_set_device: HipSetDevice,
    pub hip_device_get_name: HipDeviceGetName,
    pub hip_device_synchronize: HipDeviceSynchronize,
    pub hip_module_load_data: HipModuleLoadData,
    pub hip_module_unload: HipModuleUnload,
    pub hip_module_get_function: HipModuleGetFunction,
    pub hip_malloc: HipMalloc,
    pub hip_free: HipFree,
    pub hip_memcpy: HipMemcpy,
    pub hip_memset: HipMemset,
    pub hip_module_launch_kernel: HipModuleLaunchKernel,
    pub hip_stream_create: HipStreamCreate,
    pub hip_stream_destroy: HipStreamDestroy,
    pub hip_stream_synchronize: HipStreamSynchronize,
    pub hip_stream_begin_capture: HipStreamBeginCapture,
    pub hip_stream_end_capture: HipStreamEndCapture,
    pub hip_graph_instantiate: HipGraphInstantiate,
    pub hip_graph_launch: HipGraphLaunch,
    pub hip_graph_exec_destroy: HipGraphExecDestroy,
    pub hip_graph_destroy: HipGraphDestroy,
    hip_get_error_name: HipGetErrorName,
    hip_get_error_string: HipGetErrorString,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        unsafe { $lib.function::<$ty>($name) }.map_err(|e| format!("HIP runtime: {e}"))?
    };
}

fn c_message(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl Runtime {
    fn open() -> Result<Self, String> {
        let mut names: Vec<String> = Vec::new();
        for name in [
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "/opt/rocm/lib/libamdhip64.so",
        ] {
            names.push(name.to_string());
        }
        let lib = Library::open_first(&names)?;
        Ok(Runtime {
            hip_init: sym!(lib, "hipInit", HipInit),
            hip_get_device_count: sym!(lib, "hipGetDeviceCount", HipGetDeviceCount),
            hip_set_device: sym!(lib, "hipSetDevice", HipSetDevice),
            hip_device_get_name: sym!(lib, "hipDeviceGetName", HipDeviceGetName),
            hip_device_synchronize: sym!(lib, "hipDeviceSynchronize", HipDeviceSynchronize),
            hip_module_load_data: sym!(lib, "hipModuleLoadData", HipModuleLoadData),
            hip_module_unload: sym!(lib, "hipModuleUnload", HipModuleUnload),
            hip_module_get_function: sym!(lib, "hipModuleGetFunction", HipModuleGetFunction),
            hip_malloc: sym!(lib, "hipMalloc", HipMalloc),
            hip_free: sym!(lib, "hipFree", HipFree),
            hip_memcpy: sym!(lib, "hipMemcpy", HipMemcpy),
            hip_memset: sym!(lib, "hipMemset", HipMemset),
            hip_module_launch_kernel: sym!(lib, "hipModuleLaunchKernel", HipModuleLaunchKernel),
            hip_stream_create: sym!(lib, "hipStreamCreateWithFlags", HipStreamCreate),
            hip_stream_destroy: sym!(lib, "hipStreamDestroy", HipStreamDestroy),
            hip_stream_synchronize: sym!(lib, "hipStreamSynchronize", HipStreamSynchronize),
            hip_stream_begin_capture: sym!(lib, "hipStreamBeginCapture", HipStreamBeginCapture),
            hip_stream_end_capture: sym!(lib, "hipStreamEndCapture", HipStreamEndCapture),
            hip_graph_instantiate: sym!(lib, "hipGraphInstantiate", HipGraphInstantiate),
            hip_graph_launch: sym!(lib, "hipGraphLaunch", HipGraphLaunch),
            hip_graph_exec_destroy: sym!(lib, "hipGraphExecDestroy", HipGraphExecDestroy),
            hip_graph_destroy: sym!(lib, "hipGraphDestroy", HipGraphDestroy),
            hip_get_error_name: sym!(lib, "hipGetErrorName", HipGetErrorName),
            hip_get_error_string: sym!(lib, "hipGetErrorString", HipGetErrorString),
        })
    }

    /// Turn a failed return code into an error naming `what`.
    pub fn check(&self, code: HipError, what: &str) -> Result<(), Error> {
        if code == HIP_SUCCESS {
            return Ok(());
        }
        let name = c_message(unsafe { (self.hip_get_error_name)(code) });
        let msg = c_message(unsafe { (self.hip_get_error_string)(code) });
        Err(Error::Rocm(format!("{what} failed: {name} ({code}) {msg}")))
    }
}

/// The process-wide HIP runtime, opened on first use.
pub fn runtime() -> Result<&'static Runtime, Error> {
    static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
    match RUNTIME.get_or_init(Runtime::open) {
        Ok(r) => Ok(r),
        Err(msg) => Err(Error::Rocm(format!("HIP runtime unavailable: {msg}"))),
    }
}
