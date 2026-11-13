//! On-disk `MTLBinaryArchive` cache, so cold starts skip recompiling.

use crate::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLPipelineOption,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

/// On-disk pipeline cache settings, supplied by the application.
#[derive(Clone)]
pub struct CacheConfig {
    /// `false` keeps an in-memory archive only (nothing touches disk).
    pub enabled: bool,
    /// Where archives live. `None` keeps the archive in memory.
    pub dir: Option<PathBuf>,
    pub namespace: &'static str,
    /// Change it with any shader source, or edits run stale.
    pub key: u64,
    /// Print cache load/save lines to stderr.
    pub verbose: bool,
}

impl CacheConfig {
    /// In-memory only, so namespace and key do nothing.
    pub const MEMORY: Self = Self {
        enabled: false,
        dir: None,
        namespace: "gpukit",
        key: 0,
        verbose: false,
    };

    fn root_dir(&self) -> Option<PathBuf> {
        let dir = self.dir.as_ref().filter(|_| self.enabled)?;
        Some(dir.join(self.namespace).join("metal-pipelines"))
    }
}

fn sanitize_device_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub struct PipelineArchiveCache {
    /// `MTLBinaryArchive` is not thread-safe: every use takes this lock.
    inner: Mutex<ArchiveInner>,
    cache_file: PathBuf,
    persist: bool,
    verbose: bool,
}

struct ArchiveInner {
    archive: Retained<ProtocolObject<dyn MTLBinaryArchive>>,
    /// Keyed by label, which must encode the full specialization.
    compiled: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    dirty: bool,
}

// SAFETY: the archive sits behind the lock. Pipelines are immutable.
unsafe impl Send for PipelineArchiveCache {}
unsafe impl Sync for PipelineArchiveCache {}

static GLOBAL_CACHE: OnceLock<Result<Arc<PipelineArchiveCache>, String>> = OnceLock::new();

impl PipelineArchiveCache {
    /// The process-global cache. The first call's `config` wins.
    pub fn shared(
        device: &ProtocolObject<dyn MTLDevice>,
        config: &CacheConfig,
    ) -> Result<Arc<Self>, Error> {
        match GLOBAL_CACHE.get_or_init(|| {
            Self::open(device, config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        }) {
            Ok(c) => Ok(Arc::clone(c)),
            Err(msg) => Err(Error::Cache(msg.clone())),
        }
    }

    pub fn flush_global() {
        if let Some(Ok(c)) = GLOBAL_CACHE.get() {
            c.flush_if_dirty();
        }
    }

    fn open(device: &ProtocolObject<dyn MTLDevice>, config: &CacheConfig) -> Result<Self, Error> {
        let open_started = std::time::Instant::now();
        let Some(cache_dir) = config.root_dir() else {
            return Self::open_ephemeral(device);
        };
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::Cache(format!("mkdir failed: {e}")))?;

        let device_name = sanitize_device_name(&device.name().to_string());
        let cache_file =
            cache_dir.join(format!("{device_name}-{:016x}.metallibarchive", config.key));
        let existed = cache_file.exists();

        let desc = MTLBinaryArchiveDescriptor::new();
        if existed && let Some(url) = NSURL::from_file_path(&cache_file) {
            desc.setUrl(Some(&url));
        }

        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|e| {
                Error::Cache(format!(
                    "MTLBinaryArchive open failed: {}",
                    e.localizedDescription()
                ))
            })?;

        if config.verbose {
            eprintln!(
                "metal pipeline cache: {} ({}, {:.2?})",
                cache_file.display(),
                if existed { "loaded" } else { "new" },
                open_started.elapsed()
            );
        }

        Ok(Self {
            inner: Mutex::new(ArchiveInner {
                archive,
                compiled: HashMap::new(),
                dirty: false,
            }),
            cache_file,
            persist: true,
            verbose: config.verbose,
        })
    }

    fn open_ephemeral(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&desc)
            .map_err(|_| Error::Gpu("MTLBinaryArchive create failed"))?;
        Ok(Self {
            inner: Mutex::new(ArchiveInner {
                archive,
                compiled: HashMap::new(),
                dirty: false,
            }),
            cache_file: PathBuf::new(),
            persist: false,
            verbose: false,
        })
    }

    pub fn compile_compute(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        label: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());

        if let Some(pipeline) = inner.compiled.get(label) {
            return Ok(pipeline.clone());
        }

        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function));
        desc.setLabel(Some(&NSString::from_str(label)));
        desc.setSupportIndirectCommandBuffers(true);
        let archives = NSArray::from_slice(&[&*inner.archive]);
        desc.setBinaryArchives(Some(&archives));

        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|_| Error::Gpu("Metal pipeline compile failed"))?;

        if self.persist
            && inner
                .archive
                .addComputePipelineFunctionsWithDescriptor_error(&desc)
                .is_ok()
        {
            inner.dirty = true;
        }

        inner.compiled.insert(label.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    pub fn flush_if_dirty(&self) {
        if !self.persist {
            return;
        }
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if !std::mem::replace(&mut inner.dirty, false) {
            return;
        }
        // Write then rename, so readers never see half an archive.
        let tmp = self
            .cache_file
            .with_extension(format!("tmp-{}", std::process::id()));
        let Some(url) = NSURL::from_file_path(&tmp) else {
            return;
        };
        let save_started = std::time::Instant::now();
        match inner.archive.serializeToURL_error(&url) {
            Ok(()) => {
                if let Err(e) = std::fs::rename(&tmp, &self.cache_file) {
                    let _ = std::fs::remove_file(&tmp);
                    eprintln!("warning: metal pipeline cache rename failed: {e}");
                    return;
                }
                if self.verbose {
                    eprintln!(
                        "metal pipeline cache: saved {} ({:.2?})",
                        self.cache_file.display(),
                        save_started.elapsed()
                    );
                }
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                eprintln!(
                    "warning: metal pipeline cache serialize failed: {}",
                    e.localizedDescription()
                );
            }
        }
    }
}
