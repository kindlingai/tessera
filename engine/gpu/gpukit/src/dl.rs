//! dlopen for the backends whose runtimes load at run time.

use crate::Error;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

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
pub(crate) struct Library(*mut c_void);

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

    /// Resolve `name`. `F` must be its true fn type.
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

/// The C string at `p`, or empty when null.
pub fn message(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Opens a library once per process. `fail` words the error.
pub fn once<T>(
    cell: &'static OnceLock<Result<T, String>>,
    open: impl FnOnce() -> Result<T, String>,
    fail: impl FnOnce(&str) -> Error,
) -> Result<&'static T, Error> {
    cell.get_or_init(open).as_ref().map_err(|msg| fail(msg))
}

/// Entry points from the first library that loads.
macro_rules! library {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident as $what:literal in [$($file:literal),+ $(,)?] {
            $($fvis:vis $field:ident: $ty:ty = $sym:literal,)+
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $($fvis $field: $ty,)+
        }

        impl $name {
            fn open() -> Result<Self, String> {
                let lib = $crate::dl::Library::open_first(&[$($file.to_string()),+])?;
                Ok(Self {
                    $($field: unsafe { lib.function::<$ty>($sym) }
                        .map_err(|e| format!("{}: {e}", $what))?,)+
                })
            }
        }
    };
}
pub(crate) use library;

/// FNV-1a: unlike `DefaultHasher`, stable across runs.
pub(crate) fn image_hash(image: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in image {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
