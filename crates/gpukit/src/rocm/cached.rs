//! Process-global HIP context and kernel cache.

use super::{Context, ContextConfig, Kernel};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// The process-wide HIP context, opened on first use.
pub fn cached_context() -> Result<&'static Context, crate::Error> {
    static CTX: OnceLock<Result<Context, String>> = OnceLock::new();
    match CTX.get_or_init(|| Context::new(ContextConfig::default()).map_err(|e| e.to_string())) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(crate::Error::Rocm(msg.clone())),
    }
}

/// Resolve `entry` from a code object, cached by image address.
pub fn cached_kernel(image: &'static [u8], entry: &'static str) -> Result<Kernel, crate::Error> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Kernel>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (image.as_ptr() as usize, entry);
    if let Some(existing) = cache.lock().unwrap().get(&key) {
        return Ok(existing.clone());
    }
    let kernel = cached_context()?.load_module(image)?.function(entry)?;
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}

/// Compile `source` once per process and resolve `entry`.
pub fn cached_source_kernel(
    source: &'static str,
    entry: &'static str,
) -> Result<Kernel, crate::Error> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), Kernel>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Keyed by address: each embedded source compiles once.
    let key = (source.as_ptr() as usize, entry);
    if let Some(existing) = cache.lock().unwrap().get(&key) {
        return Ok(existing.clone());
    }
    let kernel = cached_context()?.compile_kernel(source, entry)?;
    cache.lock().unwrap().insert(key, kernel.clone());
    Ok(kernel)
}
