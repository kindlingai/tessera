//! hiprtc bindings, loaded with dlopen. hiprtc compiles the .cu sources.

use crate::Error;
use crate::dl::Library;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::OnceLock;

pub type HiprtcResult = c_int;
pub type HiprtcProgram = *mut c_void;

const HIPRTC_SUCCESS: HiprtcResult = 0;

type HiprtcCreateProgram = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> HiprtcResult;
type HiprtcCompileProgram =
    unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HiprtcResult;
type HiprtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult;
type HiprtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult;
type HiprtcGetProgramLogSize = unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult;
type HiprtcGetProgramLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult;
type HiprtcDestroyProgram = unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult;

/// Resolved hiprtc entry points. The dlopen handle stays open.
struct Hiprtc {
    create: HiprtcCreateProgram,
    compile: HiprtcCompileProgram,
    code_size: HiprtcGetCodeSize,
    code: HiprtcGetCode,
    log_size: HiprtcGetProgramLogSize,
    log: HiprtcGetProgramLog,
    destroy: HiprtcDestroyProgram,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        unsafe { $lib.function::<$ty>($name) }.map_err(|e| format!("hiprtc: {e}"))?
    };
}

impl Hiprtc {
    fn open() -> Result<Self, String> {
        let mut names: Vec<String> = Vec::new();
        for name in [
            "libhiprtc.so",
            "libhiprtc.so.7",
            "libhiprtc.so.6",
            "/opt/rocm/lib/libhiprtc.so",
        ] {
            names.push(name.to_string());
        }
        let lib = Library::open_first(&names)?;
        Ok(Self {
            create: sym!(lib, "hiprtcCreateProgram", HiprtcCreateProgram),
            compile: sym!(lib, "hiprtcCompileProgram", HiprtcCompileProgram),
            code_size: sym!(lib, "hiprtcGetCodeSize", HiprtcGetCodeSize),
            code: sym!(lib, "hiprtcGetCode", HiprtcGetCode),
            log_size: sym!(lib, "hiprtcGetProgramLogSize", HiprtcGetProgramLogSize),
            log: sym!(lib, "hiprtcGetProgramLog", HiprtcGetProgramLog),
            destroy: sym!(lib, "hiprtcDestroyProgram", HiprtcDestroyProgram),
        })
    }
}

fn library() -> Result<&'static Hiprtc, Error> {
    static LIB: OnceLock<Result<Hiprtc, String>> = OnceLock::new();
    match LIB.get_or_init(Hiprtc::open) {
        Ok(lib) => Ok(lib),
        Err(msg) => Err(Error::Rocm(format!(
            "no hiprtc ({msg}); install ROCm or put it on the library path"
        ))),
    }
}

/// Compile CUDA-dialect `source` for the current device.
pub fn compile_to_code(source: &str) -> Result<Vec<u8>, Error> {
    let lib = library()?;
    let c_src =
        CString::new(source).map_err(|e| Error::Rocm(format!("kernel source has a NUL: {e}")))?;
    let c_name = CString::new("kernel.cu").expect("static name");
    let mut program: HiprtcProgram = std::ptr::null_mut();
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
    if code != HIPRTC_SUCCESS {
        return Err(Error::Rocm(format!("hiprtcCreateProgram failed ({code})")));
    }
    let result = compile_program(lib, program);
    unsafe {
        (lib.destroy)(&mut program);
    }
    result
}

fn compile_program(lib: &Hiprtc, program: HiprtcProgram) -> Result<Vec<u8>, Error> {
    // With no --offload-arch, hiprtc targets the current device.
    let std_opt = CString::new("-std=c++17").expect("std option");
    let opts = [std_opt.as_ptr()];
    let code = unsafe { (lib.compile)(program, opts.len() as c_int, opts.as_ptr()) };
    if code != HIPRTC_SUCCESS {
        let log = program_log(lib, program);
        return Err(Error::Rocm(format!(
            "hiprtcCompileProgram failed ({code}): {log}"
        )));
    }
    let mut size = 0usize;
    let code = unsafe { (lib.code_size)(program, &mut size) };
    if code != HIPRTC_SUCCESS {
        return Err(Error::Rocm(format!("hiprtcGetCodeSize failed ({code})")));
    }
    let mut image = vec![0u8; size];
    let code = unsafe { (lib.code)(program, image.as_mut_ptr().cast()) };
    if code != HIPRTC_SUCCESS {
        return Err(Error::Rocm(format!("hiprtcGetCode failed ({code})")));
    }
    Ok(image)
}

fn program_log(lib: &Hiprtc, program: HiprtcProgram) -> String {
    let mut size = 0usize;
    if unsafe { (lib.log_size)(program, &mut size) } != HIPRTC_SUCCESS || size == 0 {
        return "(no log)".to_string();
    }
    let mut buf = vec![0u8; size];
    if unsafe { (lib.log)(program, buf.as_mut_ptr().cast()) } != HIPRTC_SUCCESS {
        return "(no log)".to_string();
    }
    String::from_utf8_lossy(&buf).trim().to_string()
}
