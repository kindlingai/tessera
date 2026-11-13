use crate::metal::expand::expand;
use crate::metal::pipeline_cache::{CacheConfig, PipelineArchiveCache};
use crate::{Error, Limits};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};
use std::sync::Arc;

/// FNV-1a of a shader source: unlike `DefaultHasher`, stable across runs.
pub fn source_hash(source: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in source.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Function-constant values; every one goes into the cache label.
#[derive(Clone, Default)]
pub struct FcValues {
    bools: Vec<(u32, bool)>,
    uints: Vec<(u32, u32)>,
    ulongs: Vec<(u32, u64)>,
    floats: Vec<(u32, f32)>,
}

impl FcValues {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_bool(&mut self, index: u32, value: bool) -> &mut Self {
        self.bools.push((index, value));
        self
    }

    pub fn set_uint(&mut self, index: u32, value: u32) -> &mut Self {
        self.uints.push((index, value));
        self
    }

    pub fn set_ulong(&mut self, index: u32, value: u64) -> &mut Self {
        self.ulongs.push((index, value));
        self
    }

    pub fn set_float(&mut self, index: u32, value: f32) -> &mut Self {
        self.floats.push((index, value));
        self
    }

    fn apply(&self, fc: &MTLFunctionConstantValues) {
        unsafe {
            for (index, value) in &self.bools {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::Bool,
                    *index as usize,
                );
            }
            for (index, value) in &self.uints {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::UInt,
                    *index as usize,
                );
            }
            for (index, value) in &self.ulongs {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::ULong,
                    *index as usize,
                );
            }
            for (index, value) in &self.floats {
                fc.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from_ref(value).cast(),
                    MTLDataType::Float,
                    *index as usize,
                );
            }
        }
    }

    fn label_suffix(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        for (index, value) in &self.bools {
            let _ = write!(s, "_b{index}={}", u8::from(*value));
        }
        for (index, value) in &self.uints {
            let _ = write!(s, "_u{index}={value}");
        }
        for (index, value) in &self.ulongs {
            let _ = write!(s, "_l{index}={value}");
        }
        for (index, value) in &self.floats {
            let _ = write!(s, "_f{index}={:08x}", value.to_bits());
        }
        s
    }
}

/// A Metal device, command queue and pipeline cache.
pub struct Context {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    includes: Vec<(&'static str, &'static str)>,
    cache: Arc<PipelineArchiveCache>,
}

impl Context {
    /// Open the default device; the first context fixes cache settings.
    pub fn new(cache: CacheConfig) -> Result<Self, Error> {
        let includes = crate::includes::include_table()?;
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Gpu("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Gpu("failed to create Metal command queue"))?;
        let cache = PipelineArchiveCache::shared(&device, &cache)?;
        Ok(Self {
            device,
            queue,
            includes,
            cache,
        })
    }

    /// The device's marketing name.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Apple GPUs run 32 lanes together. The rest is asked.
    pub fn limits(&self) -> Limits {
        Limits {
            lanes: 32,
            group_bytes: self.device.maxThreadgroupMemoryLength(),
            group_threads: self.device.maxThreadsPerThreadgroup().width,
        }
    }

    /// Compile a source string (quoted includes expanded) to a library.
    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Error> {
        let ns_source = NSString::from_str(&expand(source, &self.includes));
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

    /// Compile several entry points from one source.
    pub fn compile_kernels(
        &self,
        source: &str,
        entries: &[&str],
    ) -> Result<Vec<ComputePipeline>, Error> {
        let library = self.compile_library(source)?;
        entries
            .iter()
            .map(|entry| self.compile_from_library(&library, entry))
            .collect()
    }

    /// Compile a kernel specialized by function constants.
    pub fn compile_specialized(
        &self,
        source: &str,
        entry: &str,
        values: &FcValues,
        label_prefix: &str,
    ) -> Result<ComputePipeline, Error> {
        let library = self.compile_library(source)?;
        let fc = MTLFunctionConstantValues::new();
        values.apply(&fc);
        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName_constantValues_error(&name, &fc)
            .map_err(compile_error)?;
        // The prefix is for people; values and hash give uniqueness.
        let label = format!(
            "{label_prefix}{}_s{:x}",
            values.label_suffix(),
            source_hash(source)
        );
        let pipeline = self
            .cache
            .compile_compute(&self.device, &function, &label)?;
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
