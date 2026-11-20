//! The kernel trait: what a plan step launches.

use crate::abi::{Launch, Source};
use crate::plan::Step;
use sensai_codegen::Dialect;
use sensai_graph::Order;
use sensai_platform::Platform;
use sensai_sim::ZoneKind;
use sensai_work::Load;
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
    fn accepts(&self, input: usize, form: &dyn crate::form::Load) -> bool {
        let _ = input;
        !form.narrow()
    }
    /// Its needs from a thread group. Zero needs nothing.
    fn needs(&self) -> Needs {
        Needs::default()
    }
}

/// A kernel's needs from the device to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Exactly this many lanes together, or zero for any.
    pub lanes: usize,
    pub group_bytes: usize,
    pub group_threads: usize,
    /// Hand-written text, for a backend with a compiler.
    pub compiler: bool,
    /// A dialect the hand-written text leaves out.
    pub lacks: Option<Dialect>,
    /// Tiles on a matrix unit.
    pub tile: bool,
}

impl Needs {
    /// A SIMD group per row, in groups of 256 threads.
    pub fn lanes() -> Self {
        Needs {
            lanes: sensai_codegen::LANES,
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
            && (self.lacks.is_none() || self.lacks != speaks(platform))
            && (!self.tile || platform.tile().is_some())
    }
}

/// The dialect `platform` compiles, if it compiles one.
fn speaks(platform: &Platform) -> Option<Dialect> {
    match platform.backend.as_str() {
        "metal" => Some(Dialect::Metal),
        "cuda" | "rocm" => Some(Dialect::Cuda),
        _ => None,
    }
}

/// A kernel a plan can hold, share and compare.
#[derive(Clone, derive_more::Deref, derive_more::Debug)]
#[deref(forward)]
#[debug("{_0:?}")]
pub struct KernelRef(Arc<dyn Kernel>);

impl KernelRef {
    pub fn new(kernel: impl Kernel + 'static) -> Self {
        Self(Arc::new(kernel))
    }
}

/// Equal when they print the same: name and every size.
impl PartialEq for KernelRef {
    fn eq(&self, other: &Self) -> bool {
        format!("{self:?}") == format!("{other:?}")
    }
}
