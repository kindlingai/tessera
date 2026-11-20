//! HIP runtime and hiprtc bindings, loaded at runtime with dlopen.

use crate::Error;
use crate::api::{LaunchFn, entry};
use crate::dl::once;
use crate::rtc::Rtc;
use sensai_dl::message;
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

pub type HipError = c_int;
pub(crate) type HipDeviceptr = u64;
pub(crate) type HipGraph = *mut c_void;
pub(crate) type HipGraphExec = *mut c_void;
pub(crate) type HipGraphNode = *mut c_void;

pub(crate) const HIP_SUCCESS: HipError = 0;
pub(crate) const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub(crate) const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub(crate) const HOST_MALLOC_MAPPED: u32 = 2;
/// `hipDeviceAttribute_t`, with HIP 6 and 7 numbering.
pub(crate) const ATTR_INTEGRATED: c_int = 16;
pub(crate) const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 23;
pub(crate) const ATTR_MEMORY_BUS_WIDTH: c_int = 59;
pub(crate) const ATTR_MEMORY_CLOCK_RATE: c_int = 60;
pub(crate) const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 61;
pub(crate) const ATTR_MULTIPROCESSOR_COUNT: c_int = 63;
pub(crate) const HIP_STREAM_NON_BLOCKING: u32 = 1;
/// Other threads may still synchronize while a capture is open.
pub(crate) const HIP_STREAM_CAPTURE_MODE_RELAXED: c_int = 2;

type HipSetDevice = unsafe extern "C" fn(c_int) -> HipError;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipError;
type HipMemset = unsafe extern "C" fn(HipDeviceptr, c_int, usize) -> HipError;
type HipGraphInstantiate = unsafe extern "C" fn(
    *mut HipGraphExec,
    HipGraph,
    *mut HipGraphNode,
    *mut c_char,
    usize,
) -> HipError;
type HipGetErrorName = unsafe extern "C" fn(HipError) -> *const c_char;
type HipGetErrorString = unsafe extern "C" fn(HipError) -> *const c_char;

sensai_dl::library! {
    /// Resolved runtime entry points. The dlopen handle stays open.
    pub struct Runtime as "HIP runtime" in [
        "libamdhip64.so",
        "libamdhip64.so.7",
        "libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so",
    ] {
        pub hip_init: entry::Init = "hipInit",
        pub hip_runtime_get_version: entry::Int = "hipRuntimeGetVersion",
        pub hip_device_get_attribute: entry::Attribute = "hipDeviceGetAttribute",
        pub hip_device_total_mem: entry::Total = "hipDeviceTotalMem",
        pub hip_get_device_count: entry::Int = "hipGetDeviceCount",
        pub hip_set_device: HipSetDevice = "hipSetDevice",
        pub hip_device_get_name: entry::Name = "hipDeviceGetName",
        pub hip_module_load_data: entry::Load = "hipModuleLoadData",
        pub hip_module_unload: entry::OnHandle = "hipModuleUnload",
        pub hip_module_get_function: entry::Function = "hipModuleGetFunction",
        pub hip_malloc: entry::Alloc = "hipMalloc",
        pub hip_free: entry::Free = "hipFree",
        pub hip_host_malloc: entry::HostAlloc = "hipHostMalloc",
        pub hip_host_get_device_pointer: entry::HostMap = "hipHostGetDevicePointer",
        pub hip_host_free: entry::HostFree = "hipHostFree",
        pub hip_memcpy: HipMemcpy = "hipMemcpy",
        pub hip_memset: HipMemset = "hipMemset",
        pub hip_module_launch_kernel: LaunchFn = "hipModuleLaunchKernel",
        pub hip_stream_create: entry::Stream = "hipStreamCreateWithFlags",
        pub hip_stream_destroy: entry::OnHandle = "hipStreamDestroy",
        pub hip_stream_synchronize: entry::OnHandle = "hipStreamSynchronize",
        pub hip_stream_begin_capture: entry::Capture = "hipStreamBeginCapture",
        pub hip_stream_end_capture: entry::EndCapture = "hipStreamEndCapture",
        pub hip_graph_instantiate: HipGraphInstantiate = "hipGraphInstantiate",
        pub hip_graph_launch: entry::Replay = "hipGraphLaunch",
        pub hip_graph_exec_destroy: entry::OnHandle = "hipGraphExecDestroy",
        pub hip_graph_destroy: entry::OnHandle = "hipGraphDestroy",
        pub hip_event_create: entry::Stream = "hipEventCreateWithFlags",
        pub hip_event_record: entry::Replay = "hipEventRecord",
        pub hip_event_elapsed_time: entry::Elapsed = "hipEventElapsedTime",
        pub hip_event_destroy: entry::OnHandle = "hipEventDestroy",
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
        let name = unsafe { message((self.hip_get_error_name)(code)) };
        let msg = unsafe { message((self.hip_get_error_string)(code)) };
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
