//! Atomic on-disk cache for compiled PTX and HSACO images.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tessera_engine_settings::Settings;

/// FNV-1a 128-bit offset basis and prime.
const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
const VERSION: &[u8] = b"tessera-rtc-v2";

static NEXT_TMP: AtomicU64 = AtomicU64::new(0);
static CURRENT: Mutex<Option<DiskCache>> = Mutex::new(None);

/// Disk directory and extra flags for runtime kernel compiles.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DiskCache {
    dir: Option<PathBuf>,
    flags: Vec<String>,
}

/// Sets the process-wide kernel cache directory and extra flags.
pub fn configure(settings: &Settings) {
    let cache = DiskCache::new(settings.kernel_cache.clone(), settings.nvrtc.clone());
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(cache);
}

/// Returns the active process-wide disk cache configuration.
pub(crate) fn current() -> DiskCache {
    CURRENT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

fn hash_bytes(mut state: u128, bytes: &[u8]) -> u128 {
    for &b in bytes {
        state ^= u128::from(b);
        state = state.wrapping_mul(PRIME);
    }
    state
}

fn hash_part(state: u128, bytes: &[u8]) -> u128 {
    let len = (bytes.len() as u64).to_le_bytes();
    hash_bytes(hash_bytes(state, &len), bytes)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let Some(dir) = path.parent() else {
        return Ok(());
    };
    std::fs::create_dir_all(dir)?;
    let stem = path.file_name().and_then(|n| n.to_str()).unwrap_or("img");
    let pid = std::process::id();
    let seq = NEXT_TMP.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".{stem}.{pid}.{seq}.tmp"));
    let written = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|mut file| file.write_all(bytes));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

impl DiskCache {
    pub(crate) fn new(dir: Option<PathBuf>, flags: Vec<String>) -> Self {
        Self { dir, flags }
    }

    /// Stable 128-bit hex key for one compile invocation.
    pub(crate) fn key(
        &self,
        backend: &str,
        compiler: &str,
        arch: &str,
        base: &[&str],
        source: &str,
    ) -> String {
        let mut h = hash_part(OFFSET, VERSION);
        h = hash_part(h, backend.as_bytes());
        h = hash_part(h, compiler.as_bytes());
        h = hash_part(h, arch.as_bytes());
        let count = (base.len() + self.flags.len()) as u64;
        h = hash_bytes(h, &count.to_le_bytes());
        for flag in base {
            h = hash_part(h, flag.as_bytes());
        }
        for flag in &self.flags {
            h = hash_part(h, flag.as_bytes());
        }
        h = hash_part(h, source.as_bytes());
        format!("{h:032x}")
    }

    /// Cache file path for one compile, if caching is enabled.
    pub(crate) fn path(
        &self,
        backend: &str,
        compiler: &str,
        arch: &str,
        ext: &str,
        base: &[&str],
        source: &str,
    ) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        let key = self.key(backend, compiler, arch, base, source);
        Some(dir.join(format!("{key}.{ext}")))
    }

    /// Loads a cached image, or builds and stores one atomically.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_or<E>(
        &self,
        backend: &str,
        compiler: &str,
        arch: &str,
        ext: &str,
        base: &[&str],
        source: &str,
        build: impl FnOnce(&[&str]) -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u8>, E> {
        let path = self.path(backend, compiler, arch, ext, base, source);
        if let Some(ref at) = path
            && let Ok(bytes) = std::fs::read(at)
            && !bytes.is_empty()
        {
            return Ok(bytes);
        }
        let mut opts: Vec<&str> = Vec::with_capacity(base.len() + self.flags.len());
        opts.extend_from_slice(base);
        for flag in &self.flags {
            opts.push(flag.as_str());
        }
        let image = build(&opts)?;
        if let Some(ref at) = path
            && !image.is_empty()
        {
            let _ = write_atomic(at, &image);
        }
        Ok(image)
    }
}
