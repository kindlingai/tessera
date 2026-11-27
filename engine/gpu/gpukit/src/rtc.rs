//! Runtime compilers for CUDA-dialect source, loaded with dlopen.

use crate::Error;
use crate::dl::once;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::OnceLock;
use tessera_dl::Library;

type Program = *mut c_void;
type Create = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type Compile = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int;
type Size = unsafe extern "C" fn(Program, *mut usize) -> c_int;
type Read = unsafe extern "C" fn(Program, *mut c_char) -> c_int;
type Destroy = unsafe extern "C" fn(*mut Program) -> c_int;
type Version = unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int;

const SUCCESS: c_int = 0;

/// One runtime compiler: NVRTC or hiprtc.
pub(crate) struct Rtc {
    /// The compiler's name in messages.
    pub(crate) name: &'static str,
    /// The prefix on each symbol.
    pub(crate) prefix: &'static str,
    /// The name its symbols use for compiled code.
    pub(crate) code: &'static str,
    pub(crate) libraries: &'static [&'static str],
    /// The install hint when every library fails to load.
    pub(crate) install: &'static str,
    pub(crate) error: fn(String) -> Error,
    pub(crate) lib: OnceLock<Result<Fns, String>>,
}

/// Resolved entry points. The dlopen handle stays open.
pub(crate) struct Fns {
    create: Create,
    compile: Compile,
    code_size: Size,
    code: Read,
    log_size: Size,
    log: Read,
    destroy: Destroy,
    version: Option<Version>,
}

impl Rtc {
    fn sym<F: Copy>(&self, lib: &Library, name: &str) -> Result<F, String> {
        let name = format!("{}{name}", self.prefix);
        unsafe { lib.function::<F>(&name) }.map_err(|e| format!("{}: {e}", self.name))
    }

    fn open(&self) -> Result<Fns, String> {
        let names: Vec<String> = self.libraries.iter().map(|n| n.to_string()).collect();
        let lib = Library::open_first(&names)?;
        Ok(Fns {
            create: self.sym(&lib, "CreateProgram")?,
            compile: self.sym(&lib, "CompileProgram")?,
            code_size: self.sym(&lib, &format!("Get{}Size", self.code))?,
            code: self.sym(&lib, &format!("Get{}", self.code))?,
            log_size: self.sym(&lib, "GetProgramLogSize")?,
            log: self.sym(&lib, "GetProgramLog")?,
            destroy: self.sym(&lib, "DestroyProgram")?,
            version: self.sym(&lib, "Version").ok(),
        })
    }

    fn library(&'static self) -> Result<&'static Fns, Error> {
        once(
            &self.lib,
            || self.open(),
            |msg| {
                let (name, install) = (self.name, self.install);
                (self.error)(format!(
                    "no {name} ({msg}); {install} or put it on the library path"
                ))
            },
        )
    }

    fn fail(&self, call: &str, code: c_int) -> Error {
        (self.error)(format!("{}{call} failed ({code})", self.prefix))
    }

    /// Reports `(major, minor)` from the runtime compiler library.
    pub fn version(&'static self) -> Option<(i32, i32)> {
        let fns = self.library().ok()?;
        let version = fns.version?;
        let (mut major, mut minor) = (0, 0);
        let code = unsafe { version(&mut major, &mut minor) };
        (code == SUCCESS).then_some((major, minor))
    }

    /// Stable compiler version tag for cache keys.
    pub fn version_tag(&'static self) -> String {
        match self.version() {
            Some((major, minor)) => format!("{major}.{minor}"),
            None => "unknown".into(),
        }
    }

    /// Compiles `source` named `name` with `options` to code.
    pub fn compile(
        &'static self,
        name: &str,
        source: &str,
        options: &[&str],
    ) -> Result<Vec<u8>, Error> {
        let lib = self.library()?;
        let c_src = CString::new(source)
            .map_err(|e| (self.error)(format!("kernel source has a NUL: {e}")))?;
        let unit = if name.is_empty() { "kernel.cu" } else { name };
        let c_name =
            CString::new(unit).map_err(|e| (self.error)(format!("kernel name has a NUL: {e}")))?;
        let mut program: Program = std::ptr::null_mut();
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
        if code != SUCCESS {
            return Err(self.fail("CreateProgram", code));
        }
        let result = self.build(lib, program, options);
        unsafe {
            (lib.destroy)(&mut program);
        }
        result
    }

    fn build(&self, lib: &Fns, program: Program, options: &[&str]) -> Result<Vec<u8>, Error> {
        let mut c_opts = Vec::with_capacity(options.len());
        for opt in options {
            let c_opt = CString::new(*opt)
                .map_err(|e| (self.error)(format!("compiler option has a NUL: {e}")))?;
            c_opts.push(c_opt);
        }
        let opts: Vec<*const c_char> = c_opts.iter().map(|o| o.as_ptr()).collect();
        let code = unsafe { (lib.compile)(program, opts.len() as c_int, opts.as_ptr()) };
        if code != SUCCESS {
            let log = program_log(lib, program);
            let prefix = self.prefix;
            return Err((self.error)(format!(
                "{prefix}CompileProgram failed ({code}): {log}"
            )));
        }
        let mut size = 0usize;
        let code = unsafe { (lib.code_size)(program, &mut size) };
        if code != SUCCESS {
            return Err(self.fail(&format!("Get{}Size", self.code), code));
        }
        let mut image = vec![0u8; size];
        let code = unsafe { (lib.code)(program, image.as_mut_ptr().cast()) };
        if code != SUCCESS {
            return Err(self.fail(&format!("Get{}", self.code), code));
        }
        Ok(image)
    }
}

fn program_log(lib: &Fns, program: Program) -> String {
    let mut size = 0usize;
    if unsafe { (lib.log_size)(program, &mut size) } != SUCCESS || size == 0 {
        return "(no log)".to_string();
    }
    let mut buf = vec![0u8; size];
    if unsafe { (lib.log)(program, buf.as_mut_ptr().cast()) } != SUCCESS {
        return "(no log)".to_string();
    }
    String::from_utf8_lossy(&buf).trim().to_string()
}
