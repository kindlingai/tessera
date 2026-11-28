//! Compiled pipelines by label, shared across the process.

use crate::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary,
    MTLPipelineOption,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Every pipeline and library compiled so far, shared process-wide.
#[derive(Default)]
pub(crate) struct Pipelines {
    /// Keyed by label. The label must encode the full specialization.
    compiled: Mutex<HashMap<String, Pipeline>>,
    /// Compiled Metal libraries keyed by expanded source text.
    libraries: Mutex<HashMap<String, Library>>,
}

// SAFETY: pipelines and libraries are immutable, and maps are locked.
unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

static GLOBAL: OnceLock<Arc<Pipelines>> = OnceLock::new();

impl Pipelines {
    /// The process-wide set.
    pub(crate) fn shared() -> Arc<Self> {
        Arc::clone(GLOBAL.get_or_init(Default::default))
    }

    pub(crate) fn get(&self, label: &str) -> Option<Pipeline> {
        self.compiled
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(label)
            .cloned()
    }

    pub(crate) fn get_library(&self, source: &str) -> Option<Library> {
        self.libraries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(source)
            .cloned()
    }

    pub(crate) fn insert_library(&self, source: String, library: Library) {
        self.libraries
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(source, library);
    }

    pub(crate) fn compile_compute(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        function: &ProtocolObject<dyn MTLFunction>,
        label: &str,
    ) -> Result<Pipeline, Error> {
        let mut compiled = self.compiled.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pipeline) = compiled.get(label) {
            return Ok(pipeline.clone());
        }
        let desc = MTLComputePipelineDescriptor::new();
        desc.setComputeFunction(Some(function));
        desc.setLabel(Some(&NSString::from_str(label)));
        let pipeline = device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &desc,
                MTLPipelineOption::empty(),
                None,
            )
            .map_err(|_| Error::Gpu("Metal pipeline compile failed"))?;
        compiled.insert(label.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}
