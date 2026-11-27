//! Compiled kernels, shared by the GPU runners.

use crate::Error;
use crate::overrides;
use crate::profile::format_attrs;
use crate::run::{Outcome, Reload};
use std::collections::{HashMap, HashSet};
use tessera_codegen::Dialect;
use tessera_gpukit::backend::Backend;
use tessera_gpukit::device::{Device, KernelAttrs};
use tessera_gpukit::launch::THREADS_PER_BLOCK;
use tessera_lower::{Launch, Plan, Shape, Source, Step};

/// A fixed kernel's text for `backend`, and its entry.
pub(crate) fn fixed_text(
    source: &Source,
    backend: Backend,
) -> Option<(&'static str, &'static str)> {
    let Source::Fixed { text, entry } = *source else {
        return None;
    };
    Some((text[backend], entry))
}

/// The largest group among the launches of `step`.
pub(crate) fn group(step: &Step) -> usize {
    let threads = |l: Launch| match l.shape {
        Shape::Grid { threads, .. } => threads,
        Shape::Line(_) | Shape::Runtime(_) => THREADS_PER_BLOCK as usize,
    };
    step.launches().into_iter().map(threads).max().unwrap_or(1)
}

/// Block width used when launching `step`.
fn launch_threads(step: &Step) -> u32 {
    let threads = |l: Launch| match l.shape {
        Shape::Grid { threads, .. } => threads as u32,
        Shape::Line(n) => THREADS_PER_BLOCK.min(n.max(1) as u32),
        Shape::Runtime(_) => THREADS_PER_BLOCK,
    };
    step.launches().into_iter().map(threads).max().unwrap_or(1)
}

/// Resource counts and active CTAs/SM for each step.
pub(crate) fn step_attrs<D: Device>(
    ctx: &D,
    plan: &Plan,
    kernels: &[D::Kernel],
    resident: &[bool],
) -> Vec<Option<KernelAttrs>> {
    let mut out = Vec::with_capacity(plan.steps.len());
    for i in 0..plan.steps.len() {
        let step = &plan.steps[i];
        let (threads, shared) = match step.kernel.resident().filter(|_| resident[i]) {
            Some(r) => (r.threads, r.shared),
            None => (
                launch_threads(step),
                u32::try_from(step.kernel.needs().dynamic_shared).unwrap_or(u32::MAX),
            ),
        };
        out.push(ctx.attrs(&kernels[i], threads, shared));
    }
    out
}

/// Logs one notice per spilling kernel entry.
pub(crate) fn log_spills(plan: &Plan, attrs: &[Option<KernelAttrs>], seen: &mut HashSet<String>) {
    for (step, attr) in plan.steps.iter().zip(attrs) {
        let Some(a) = attr.filter(|a| a.local_bytes > 0) else {
            continue;
        };
        let source = step.kernel.source();
        let entry = overrides::entry(&source);
        if seen.insert(entry.to_string()) {
            eprintln!("tessera: spill in {entry}: {}", format_attrs(a));
        }
    }
}

/// Logs a kernel reload, appending compiled attributes when available.
pub(crate) fn log_reload(
    reload: &Reload,
    entries: &[(String, usize)],
    attrs: &[Option<KernelAttrs>],
) {
    let active = matches!(reload.outcome, Outcome::Loaded | Outcome::Planned);
    let stem = reload.path.file_stem().and_then(|s| s.to_str());
    let attr = active
        .then(|| {
            entries
                .iter()
                .find(|(e, _)| Some(e.as_str()) == stem)
                .and_then(|(_, step)| attrs.get(*step).copied().flatten())
        })
        .flatten();
    match attr {
        Some(a) => eprintln!("tessera: {reload} ({})", format_attrs(a)),
        None => eprintln!("tessera: {reload}"),
    }
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

impl<T> Kernels<T> {
    pub(crate) fn is_empty(&self) -> bool {
        self.made.is_empty() && self.known.is_empty()
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
                        let text = body.text().bounded(Dialect::of(backend), Some(group));
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
