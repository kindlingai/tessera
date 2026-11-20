//! Compiled pipelines by label, shared across the process.

use crate::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice, MTLFunction,
    MTLPipelineOption,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Every pipeline compiled so far, by its label.
#[derive(Default)]
pub(crate) struct Pipelines {
    /// Keyed by label. The label must encode the full specialization.
    compiled: Mutex<HashMap<String, Pipeline>>,
}

// SAFETY: pipelines are immutable, and the map is locked.
unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

static GLOBAL: OnceLock<Arc<Pipelines>> = OnceLock::new();

impl Pipelines {
    /// The process-wide set.
    pub(crate) fn shared() -> Arc<Self> {
        Arc::clone(GLOBAL.get_or_init(Default::default))
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
