//! Compiled kernels, shared by the GPU runners.

use crate::Error;
use std::collections::HashMap;
use tessera_codegen::Dialect;
use tessera_gpukit::backend::Backend;
use tessera_gpukit::launch::THREADS_PER_BLOCK;
use tessera_lower::{Launch, Shape, Source, Step};

/// The kernel text a backend compiles.
pub(crate) fn dialect(backend: Backend) -> Dialect {
    match backend {
        Backend::Metal => Dialect::Metal,
        Backend::Cuda | Backend::Rocm => Dialect::Cuda,
    }
}

/// A fixed kernel's text for `backend`, and its entry.
pub(crate) fn fixed_text(
    source: &Source,
    backend: Backend,
) -> Option<(&'static str, &'static str)> {
    let Source::Fixed {
        metal,
        cuda,
        hip,
        entry,
    } = *source
    else {
        return None;
    };
    let text = match backend {
        Backend::Metal => metal,
        Backend::Cuda => cuda,
        Backend::Rocm => hip,
    };
    Some((text, entry))
}

/// The largest group among the launches of `step`.
pub(crate) fn group(step: &Step) -> usize {
    let threads = |l: Launch| match l.shape {
        Shape::Grid { threads, .. } => threads,
        Shape::Line(_) | Shape::Runtime(_) => THREADS_PER_BLOCK as usize,
    };
    step.launches().into_iter().map(threads).max().unwrap_or(1)
}

/// Compiled kernels, by group size and entry.
pub(crate) struct Kernels<T> {
    made: HashMap<String, T>,
    /// With whether each was generated.
    known: HashMap<String, (T, bool)>,
}

impl<T> Default for Kernels<T> {
    fn default() -> Self {
        Self {
            made: HashMap::new(),
            known: HashMap::new(),
        }
    }
}

impl<T: Clone> Kernels<T> {
    /// `step`'s kernel, compiled once. True when generated.
    pub(crate) fn get(
        &mut self,
        step: &Step,
        backend: Backend,
        fixed: impl FnOnce(&'static str, &'static str) -> Result<T, Error>,
        made: impl FnOnce(&str, &str) -> Result<T, Error>,
    ) -> Result<(T, bool), Error> {
        // Printing a kernel is cheaper than writing its source.
        let key = format!("{:?}", step.kernel);
        if let Some(found) = self.known.get(&key) {
            return Ok(found.clone());
        }
        let source = step.kernel.source();
        let found = match source {
            Source::Fixed { .. } => {
                let (text, entry) = fixed_text(&source, backend).unwrap_or_default();
                (fixed(text, entry)?, false)
            }
            Source::Made(body) => {
                let group = group(step);
                let at = format!("{}/{group}", body.entry);
                match self.made.get(&at) {
                    Some(compiled) => (compiled.clone(), true),
                    None => {
                        let text = body.text().bounded(dialect(backend), Some(group));
                        let compiled = made(&text, &body.entry)?;
                        self.made.insert(at, compiled.clone());
                        (compiled, true)
                    }
                }
            }
        };
        self.known.insert(key, found.clone());
        Ok(found)
    }
}
