//! dlopen for the backends whose runtimes load at run time.

use std::ffi::{CStr, CString, c_char, c_int, c_void};

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

fn last_error() -> String {
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".to_string()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// A shared library that stays open until the process exits.
pub struct Library(*mut c_void);

impl Library {
    /// Open the first name that loads, else the last error.
    pub fn open_first(names: &[String]) -> Result<Self, String> {
        let mut last = String::from("no library names to try");
        for name in names {
            let Ok(c_name) = CString::new(name.as_str()) else {
                continue;
            };
            let lib = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW) };
            if !lib.is_null() {
                return Ok(Self(lib));
            }
            last = format!("{name}: {}", last_error());
        }
        Err(last)
    }

    /// Resolve `name`; `F` must be its true fn type.
    pub unsafe fn function<F: Copy>(&self, name: &str) -> Result<F, String> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        let c_name = CString::new(name).map_err(|e| format!("bad symbol name: {e}"))?;
        let raw = unsafe { dlsym(self.0, c_name.as_ptr()) };
        if raw.is_null() {
            return Err(format!("missing symbol {name}"));
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&raw) })
    }
}

/// FNV-1a: unlike `DefaultHasher`, stable across runs.
pub fn image_hash(image: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in image {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
