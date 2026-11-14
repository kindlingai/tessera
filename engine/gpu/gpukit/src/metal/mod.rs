//! Metal backend: context, specialization, pipeline cache, buffers, dispatch.

mod buffer;
mod cached;
mod context;
#[cfg(test)]
mod context_test;
mod dispatch;
mod expand;
#[cfg(test)]
mod expand_test;
mod pipeline_cache;
mod shared;

pub use buffer::BufferPool;
pub use cached::{cached_context, cached_pipeline};
pub use context::{ComputePipeline, Context};
pub use dispatch::{bind_buffer, dispatch_grid, div_up, set_bytes};
pub use expand::expand;
pub use pipeline_cache::{CacheConfig, PipelineArchiveCache};
pub use shared::Shared;
