//! Helpers for the backends whose runtimes load at run time.

/// Opens a library once per process. `fail` words the error.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn once<T>(
    cell: &'static std::sync::OnceLock<Result<T, String>>,
    open: impl FnOnce() -> Result<T, String>,
    fail: impl FnOnce(&str) -> crate::Error,
) -> Result<&'static T, crate::Error> {
    cell.get_or_init(open).as_ref().map_err(|msg| fail(msg))
}

/// FNV-1a: unlike `DefaultHasher`, stable across runs.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn image_hash(image: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in image {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
