//! The kernel trait: what a plan step launches.

use crate::abi::{Launch, Source};
use crate::form::Form;
use crate::plan::Step;
use gpukit::Platform;
use graph::Order;
use sim::{Load, ZoneKind};
use std::fmt;
use std::sync::Arc;

/// Seconds per unit of work for an unmeasured kernel.
pub const UNPROVEN: f64 = 1e-9;

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
    /// Sizes its time depends on. Near sizes cost alike.
    fn sizes(&self) -> Vec<usize> {
        Vec::new()
    }
    /// Seconds per unit of work, until it is measured.
    fn guess(&self) -> f64 {
        1e-10
    }
    /// How firmly it orders its sums.
    fn order(&self) -> Order {
        Order::Fixed
    }
    /// Its work and traffic when writing `len` elements.
    fn load(&self, _: usize) -> Option<Load> {
        None
    }
    /// Whether it writes over its first input.
    fn overwrites(&self) -> bool {
        false
    }
    /// Whether equal launches on equal inputs may share a result.
    fn shared(&self) -> bool {
        false
    }
    /// The hardware it runs on.
    fn runs_on(&self) -> ZoneKind {
        ZoneKind::Gpu
    }
    /// Whether it reads input `input` stored as `form`.
    fn accepts(&self, input: usize, form: &dyn Form) -> bool {
        let _ = input;
        !form.narrow()
    }
    /// What it asks of a thread group. Zero asks nothing.
    fn needs(&self) -> Needs {
        Needs::default()
    }
}

/// What a kernel asks of the device to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Exactly this many lanes together, or zero for any.
    pub lanes: usize,
    pub group_bytes: usize,
    pub group_threads: usize,
    /// Hand-written text, for a backend with a compiler.
    pub compiler: bool,
    /// Tiles on a matrix unit.
    pub tile: bool,
}

impl Needs {
    /// A SIMD group per row, in groups of 256 threads.
    pub fn lanes() -> Self {
        Needs {
            lanes: codegen::LANES,
            group_threads: 256,
            ..Needs::default()
        }
    }

    /// Hand-written kernels ask for a compiler, nothing else.
    pub fn text() -> Self {
        Needs {
            compiler: true,
            ..Needs::default()
        }
    }

    pub fn fits(&self, platform: &Platform) -> bool {
        let limits = &platform.limits;
        (self.lanes == 0 || self.lanes == limits.lanes)
            && self.group_bytes <= limits.group_bytes
            && self.group_threads <= limits.group_threads
            && (!self.compiler || limits.compiler)
            && (!self.tile || platform.tile().is_some())
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
