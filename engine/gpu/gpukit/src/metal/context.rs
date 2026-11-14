use crate::metal::expand::expand;
use crate::metal::pipeline_cache::{CacheConfig, PipelineArchiveCache};
use crate::platform::{Asked, Platform};
use crate::{Error, Limits};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLLibrary,
};
use std::sync::Arc;

/// A Metal device, command queue and pipeline cache.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    cache: Arc<PipelineArchiveCache>,
}

impl Context {
    /// Open the default device. The first context fixes cache settings.
    pub fn new(cache: CacheConfig) -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Gpu("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Gpu("failed to create Metal command queue"))?;
        let cache = PipelineArchiveCache::shared(&device, &cache)?;
        Ok(Self {
            device,
            queue,
            cache,
        })
    }

    /// The device's marketing name.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// The device as asked, with its table row.
    pub fn platform(&self) -> Platform {
        let families = [
            (MTLGPUFamily::Apple9, "apple9"),
            (MTLGPUFamily::Apple8, "apple8"),
            (MTLGPUFamily::Apple7, "apple7"),
        ];
        let newest = families
            .iter()
            .find(|(f, _)| self.device.supportsFamily(*f));
        // The architecture names the chip's GPU, as `applegpu_g15p`.
        let arch = match self.device.architecture().name().to_string() {
            name if name.starts_with("applegpu_") => name,
            _ => newest.map_or("apple", |(_, n)| n).into(),
        };
        Platform::new(Asked {
            backend: "metal",
            arch,
            name: self.name(),
            driver: String::new(),
            limits: self.limits(),
            units: 0,
            bandwidth: 0.0,
            integrated: true,
        })
    }

    /// Apple GPUs run 32 lanes together. The rest is asked.
    pub fn limits(&self) -> Limits {
        Limits {
            lanes: 32,
            group_bytes: self.device.maxThreadgroupMemoryLength(),
            group_threads: self.device.maxThreadsPerThreadgroup().width,
            compiler: true,
            pdl: false,
        }
    }

    /// Compile a source string (quoted includes expanded) to a library.
    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        let ns_source = NSString::from_str(&expand(source, &[]));
        self.device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(compile_error)
    }

    /// Compile one kernel, cached by entry name across the process.
    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        self.compile_from_library(&library, entry)
    }

    /// As [`Self::compile_kernel`], for an already-compiled library.
    pub fn compile_from_library(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        entry: &str,
    ) -> Result<ComputePipeline, Error> {
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or(Error::Gpu("Metal kernel entry not found"))?;
        let pipeline = self.cache.compile_compute(&self.device, &function, entry)?;
        Ok(ComputePipeline { pipeline })
    }
}

pub struct ComputePipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Clone for ComputePipeline {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
        }
    }
}

fn compile_error(err: Retained<NSError>) -> Error {
    Error::Compile(err.localizedDescription().to_string())
}
