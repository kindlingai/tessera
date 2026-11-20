//! Compiled kernels, shared by the GPU runners.

use crate::Error;
use sensai_codegen::Dialect;
use sensai_gpukit::backend::Backend;
use sensai_lower::{Source, Step};
use std::collections::HashMap;

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

/// Compiled kernels, by how a kernel prints and by entry.
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
            Source::Made(body) => match self.made.get(&body.entry) {
                Some(compiled) => (compiled.clone(), true),
                None => {
                    let compiled = made(&body.text().render(dialect(backend)), &body.entry)?;
                    self.made.insert(body.entry.clone(), compiled.clone());
                    (compiled, true)
                }
            },
        };
        self.known.insert(key, found.clone());
        Ok(found)
    }
}
