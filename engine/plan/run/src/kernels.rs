//! Compiled kernels, shared by the GPU runners.

use codegen::Dialect;
use lower::{Error, Source, Step};
use std::collections::HashMap;

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
        dialect: Dialect,
        fixed: impl FnOnce(&'static str, &'static str) -> Result<T, Error>,
        made: impl FnOnce(&str, &str) -> Result<T, Error>,
    ) -> Result<(T, bool), Error> {
        // Printing a kernel is cheaper than writing its source.
        let key = format!("{:?}", step.kernel);
        if let Some(found) = self.known.get(&key) {
            return Ok(found.clone());
        }
        let found = match step.kernel.source() {
            Source::Fixed { metal, cuda, entry } => {
                let text = if dialect == Dialect::Metal {
                    metal
                } else {
                    cuda
                };
                (fixed(text, entry)?, false)
            }
            Source::Made(body) => match self.made.get(&body.entry) {
                Some(compiled) => (compiled.clone(), true),
                None => {
                    let compiled = made(&body.text().render(dialect), &body.entry)?;
                    self.made.insert(body.entry.clone(), compiled.clone());
                    (compiled, true)
                }
            },
        };
        self.known.insert(key, found.clone());
        Ok(found)
    }
}
