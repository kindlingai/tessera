//! Per-thread Metal context and pipelines, since neither is Send.

use super::{CacheConfig, ComputePipeline, Context};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
    static PIPELINES: RefCell<HashMap<(&'static str, &'static str), Rc<ComputePipeline>>> =
        RefCell::new(HashMap::new());
}

/// This thread's Metal context; the first caller's `cache` wins.
pub fn cached_context(cache: CacheConfig) -> Result<Rc<Context>, crate::Error> {
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let ctx = Context::new(cache)?;
            *slot = Some(Rc::new(ctx));
        }
        Ok(Rc::clone(slot.as_ref().expect("just set")))
    })
}

/// The pipeline for (source, entry), compiled once per thread.
pub fn cached_pipeline(
    ctx: &Context,
    source: &'static str,
    entry: &'static str,
) -> Result<Rc<ComputePipeline>, crate::Error> {
    PIPELINES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(existing) = cache.get(&(source, entry)) {
            return Ok(Rc::clone(existing));
        }
        let compiled = Rc::new(ctx.compile_kernel(source, entry)?);
        cache.insert((source, entry), Rc::clone(&compiled));
        Ok(compiled)
    })
}
