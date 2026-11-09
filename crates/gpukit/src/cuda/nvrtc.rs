//! NVRTC bindings loaded with dlopen; NVRTC ships with the toolkit.

use crate::Error;
use crate::dl::Library;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::OnceLock;

pub type NvrtcResult = c_int;
pub type NvrtcProgram = *mut c_void;

const NVRTC_SUCCESS: NvrtcResult = 0;
/// Environment variable that overrides the NVRTC library name.
const NVRTC_ENV: &str = "DGQ_NVRTC";

type NvrtcCreateProgram = unsafe extern "C" fn(
    *mut NvrtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> NvrtcResult;
type NvrtcCompileProgram =
    unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult;
type NvrtcGetPtxSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type NvrtcGetPtx = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type NvrtcGetProgramLogSize = unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult;
type NvrtcGetProgramLog = unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult;
type NvrtcDestroyProgram = unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult;

/// Resolved NVRTC entry points; the dlopen handle is never closed.
struct Nvrtc {
    create: NvrtcCreateProgram,
    compile: NvrtcCompileProgram,
    ptx_size: NvrtcGetPtxSize,
    ptx: NvrtcGetPtx,
    log_size: NvrtcGetProgramLogSize,
    log: NvrtcGetProgramLog,
    destroy: NvrtcDestroyProgram,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        unsafe { $lib.function::<$ty>($name) }.map_err(|e| format!("NVRTC: {e}"))?
    };
}

impl Nvrtc {
    fn open() -> Result<Self, String> {
        let mut names: Vec<String> = Vec::new();
        if let Ok(explicit) = std::env::var(NVRTC_ENV) {
            names.push(explicit);
        }
        for name in [
            "libnvrtc.so",
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so.11",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ] {
            names.push(name.to_string());
        }
        let lib = Library::open_first(&names)?;
        Ok(Self {
            create: sym!(lib, "nvrtcCreateProgram", NvrtcCreateProgram),
            compile: sym!(lib, "nvrtcCompileProgram", NvrtcCompileProgram),
            ptx_size: sym!(lib, "nvrtcGetPTXSize", NvrtcGetPtxSize),
            ptx: sym!(lib, "nvrtcGetPTX", NvrtcGetPtx),
            log_size: sym!(lib, "nvrtcGetProgramLogSize", NvrtcGetProgramLogSize),
            log: sym!(lib, "nvrtcGetProgramLog", NvrtcGetProgramLog),
            destroy: sym!(lib, "nvrtcDestroyProgram", NvrtcDestroyProgram),
        })
    }
}

fn library() -> Result<&'static Nvrtc, Error> {
    static LIB: OnceLock<Result<Nvrtc, String>> = OnceLock::new();
    match LIB.get_or_init(Nvrtc::open) {
        Ok(lib) => Ok(lib),
        Err(msg) => Err(Error::Cuda(format!(
            "no NVRTC ({msg}); install the CUDA toolkit or set {NVRTC_ENV}"
        ))),
    }
}

/// Compile `source` to NUL-terminated PTX for one compute capability.
pub fn compile_to_ptx(source: &str, major: i32, minor: i32) -> Result<Vec<u8>, Error> {
    let lib = library()?;
    let c_src =
        CString::new(source).map_err(|e| Error::Cuda(format!("kernel source has a NUL: {e}")))?;
    let c_name = CString::new("kernel.cu").expect("static name");
    let mut program: NvrtcProgram = std::ptr::null_mut();
    let code = unsafe {
        (lib.create)(
            &mut program,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if code != NVRTC_SUCCESS {
        return Err(Error::Cuda(format!("nvrtcCreateProgram failed ({code})")));
    }
    let result = compile_program(lib, program, major, minor);
    unsafe {
        (lib.destroy)(&mut program);
    }
    result
}

fn compile_program(
    lib: &Nvrtc,
    program: NvrtcProgram,
    major: i32,
    minor: i32,
) -> Result<Vec<u8>, Error> {
    let arch =
        CString::new(format!("--gpu-architecture=compute_{major}{minor}")).expect("arch option");
    let std_opt = CString::new("--std=c++17").expect("std option");
    let opts = [arch.as_ptr(), std_opt.as_ptr()];
    let code = unsafe { (lib.compile)(program, opts.len() as c_int, opts.as_ptr()) };
    if code != NVRTC_SUCCESS {
        let log = program_log(lib, program);
        return Err(Error::Cuda(format!(
            "nvrtcCompileProgram failed ({code}): {log}"
        )));
    }
    let mut size = 0usize;
    let code = unsafe { (lib.ptx_size)(program, &mut size) };
    if code != NVRTC_SUCCESS {
        return Err(Error::Cuda(format!("nvrtcGetPTXSize failed ({code})")));
    }
    let mut ptx = vec![0u8; size];
    let code = unsafe { (lib.ptx)(program, ptx.as_mut_ptr().cast()) };
    if code != NVRTC_SUCCESS {
        return Err(Error::Cuda(format!("nvrtcGetPTX failed ({code})")));
    }
    Ok(ptx)
}

fn program_log(lib: &Nvrtc, program: NvrtcProgram) -> String {
    let mut size = 0usize;
    if unsafe { (lib.log_size)(program, &mut size) } != NVRTC_SUCCESS || size == 0 {
        return "(no log)".to_string();
    }
    let mut buf = vec![0u8; size];
    if unsafe { (lib.log)(program, buf.as_mut_ptr().cast()) } != NVRTC_SUCCESS {
        return "(no log)".to_string();
    }
    String::from_utf8_lossy(&buf).trim().to_string()
}
