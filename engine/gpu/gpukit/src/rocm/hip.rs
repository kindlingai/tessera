//! HIP runtime and hiprtc bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::dl::{message, once};
use crate::rtc::Rtc;
use std::ffi::{c_char, c_int, c_void};
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
/// `hipDeviceAttribute_t`, as HIP 6 and 7 number it.
pub const ATTR_INTEGRATED: c_int = 16;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 23;
pub const ATTR_MEMORY_BUS_WIDTH: c_int = 59;
pub const ATTR_MEMORY_CLOCK_RATE: c_int = 60;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 61;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
pub const HIP_STREAM_NON_BLOCKING: u32 = 1;
/// Other threads may still synchronize while a capture is open.
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

type HipInit = unsafe extern "C" fn(u32) -> HipError;
type HipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> HipError;
type HipDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> HipError;
type HipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipError;
type HipSetDevice = unsafe extern "C" fn(c_int) -> HipError;
type HipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipError;
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

crate::dl::library! {
    /// Resolved runtime entry points. The dlopen handle stays open.
    pub struct Runtime as "HIP runtime" in [
        "libamdhip64.so",
        "libamdhip64.so.7",
        "libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so",
    ] {
        pub hip_init: HipInit = "hipInit",
        pub hip_runtime_get_version: HipRuntimeGetVersion = "hipRuntimeGetVersion",
        pub hip_device_get_attribute: HipDeviceGetAttribute = "hipDeviceGetAttribute",
        pub hip_get_device_count: HipGetDeviceCount = "hipGetDeviceCount",
        pub hip_set_device: HipSetDevice = "hipSetDevice",
        pub hip_device_get_name: HipDeviceGetName = "hipDeviceGetName",
        pub hip_module_load_data: HipModuleLoadData = "hipModuleLoadData",
        pub hip_module_unload: HipModuleUnload = "hipModuleUnload",
        pub hip_module_get_function: HipModuleGetFunction = "hipModuleGetFunction",
        pub hip_malloc: HipMalloc = "hipMalloc",
        pub hip_free: HipFree = "hipFree",
        pub hip_memcpy: HipMemcpy = "hipMemcpy",
        pub hip_memset: HipMemset = "hipMemset",
        pub hip_module_launch_kernel: HipModuleLaunchKernel = "hipModuleLaunchKernel",
        pub hip_stream_create: HipStreamCreate = "hipStreamCreateWithFlags",
        pub hip_stream_destroy: HipStreamDestroy = "hipStreamDestroy",
        pub hip_stream_synchronize: HipStreamSynchronize = "hipStreamSynchronize",
        pub hip_stream_begin_capture: HipStreamBeginCapture = "hipStreamBeginCapture",
        pub hip_stream_end_capture: HipStreamEndCapture = "hipStreamEndCapture",
        pub hip_graph_instantiate: HipGraphInstantiate = "hipGraphInstantiate",
        pub hip_graph_launch: HipGraphLaunch = "hipGraphLaunch",
        pub hip_graph_exec_destroy: HipGraphExecDestroy = "hipGraphExecDestroy",
        pub hip_graph_destroy: HipGraphDestroy = "hipGraphDestroy",
        hip_get_error_name: HipGetErrorName = "hipGetErrorName",
        hip_get_error_string: HipGetErrorString = "hipGetErrorString",
    }
}

impl Runtime {
    /// Turn a failed return code into an error naming `what`.
    pub fn check(&self, code: HipError, what: &str) -> Result<(), Error> {
        if code == HIP_SUCCESS {
            return Ok(());
        }
        let name = message(unsafe { (self.hip_get_error_name)(code) });
        let msg = message(unsafe { (self.hip_get_error_string)(code) });
        Err(Error::Rocm(format!("{what} failed: {name} ({code}) {msg}")))
    }
}

/// The process-wide HIP runtime, opened on first use.
pub fn runtime() -> Result<&'static Runtime, Error> {
    static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
    once(&RUNTIME, Runtime::open, |msg| {
        Error::Rocm(format!("HIP runtime unavailable: {msg}"))
    })
}

/// hiprtc compiles the CUDA-dialect sources for AMD devices.
pub(crate) static HIPRTC: Rtc = Rtc {
    name: "hiprtc",
    prefix: "hiprtc",
    code: "Code",
    libraries: &[
        "libhiprtc.so",
        "libhiprtc.so.7",
        "libhiprtc.so.6",
        "/opt/rocm/lib/libhiprtc.so",
    ],
    install: "install ROCm",
    error: Error::Rocm,
    lib: OnceLock::new(),
};
