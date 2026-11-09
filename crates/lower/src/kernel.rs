//! The kernel trait: what a plan step launches.

use crate::abi::{Launch, Source};
use crate::plan::Step;
use std::fmt;
use std::sync::Arc;

/// One kernel with its sizes. Any crate may add one.
pub trait Kernel: fmt::Debug + Send + Sync {
    /// Names its cost curve, so it must be unique.
    fn name(&self) -> &'static str;
    fn source(&self) -> Source;
    /// The launches that run `step`, usually one.
    fn launches(&self, step: &Step) -> Vec<Launch>;
    /// Multiply-adds, or elements touched, writing `len` elements.
    fn work(&self, len: usize) -> f64 {
        len as f64
    }
    /// Seconds per unit of work, until it is measured.
    fn guess(&self) -> f64 {
        1e-10
    }
    /// Whether it writes over its first input.
    fn overwrites(&self) -> bool {
        false
    }
    /// Whether equal launches on equal inputs may share a result.
    fn shared(&self) -> bool {
        false
    }
}

/// A kernel a plan can hold, share and compare.
#[derive(Clone)]
pub struct KernelRef(Arc<dyn Kernel>);

impl KernelRef {
    pub fn new(kernel: impl Kernel + 'static) -> Self {
        Self(Arc::new(kernel))
    }
}

impl std::ops::Deref for KernelRef {
    type Target = dyn Kernel;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl fmt::Debug for KernelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Equal when they print the same: name and every size.
impl PartialEq for KernelRef {
    fn eq(&self, other: &Self) -> bool {
        format!("{self:?}") == format!("{other:?}")
    }
}
