//! A plan: kernel launches over buffers that stay on device.

use crate::abi::{Arg, Launch};
use crate::form::{F32_FORM, Form};
use crate::kernel::KernelRef;
use graph::{NodeId, Order};
use std::sync::atomic::{AtomicU64, Ordering};

/// `len` elements of one buffer, `offset` elements in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub buffer: u32,
    pub offset: usize,
    pub len: usize,
    pub form: &'static dyn Form,
}

impl Slot {
    /// Elements from the start of a buffer, as f32.
    pub fn floats(buffer: u32, len: usize) -> Self {
        Self {
            buffer,
            offset: 0,
            len,
            form: F32_FORM,
        }
    }

    /// The first byte's offset.
    pub fn start(&self) -> usize {
        self.form.offset(self.offset)
    }

    pub fn bytes(&self) -> usize {
        self.form.bytes(self.len)
    }

    /// The byte offset of element `skip`.
    pub fn at(&self, skip: usize) -> usize {
        self.form.offset(self.offset + skip)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kernel: KernelRef,
    /// The rule that proposed it.
    pub rule: &'static str,
    pub reads: Vec<Slot>,
    pub write: Slot,
    /// The graph nodes this launch computes, or helps compute.
    pub covers: Vec<NodeId>,
    /// The zone it runs in.
    pub zone: u32,
}

impl Step {
    pub fn launches(&self) -> Vec<Launch> {
        self.kernel.launches(self)
    }

    /// The `i`th input's buffer, from its start.
    pub fn read(&self, i: usize) -> Arg {
        Arg::Buffer(self.reads[i], 0)
    }

    pub fn out(&self) -> Arg {
        Arg::Buffer(self.write, 0)
    }
}

/// A graph weight and where it loads, with its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub weight: u32,
    pub slot: Slot,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub slot: Slot,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Tells plans apart without comparing them. Zero means unknown.
    pub id: u64,
    /// Bytes per buffer. Values that never overlap share one.
    pub buffers: Vec<usize>,
    /// The zone each buffer lies in. Empty means zone 0.
    pub zones: Vec<u32>,
    pub steps: Vec<Step>,
    /// Each used graph weight.
    pub weights: Vec<Loaded>,
    /// Each kept-row buffer, holding this launch's rows too.
    pub kept: Vec<(u32, Slot)>,
    /// One per input port of the entry.
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Output>,
}

static IDS: AtomicU64 = AtomicU64::new(1);

/// Equal plans run alike, whatever their ids.
impl PartialEq for Plan {
    fn eq(&self, other: &Self) -> bool {
        self.buffers == other.buffers
            && self.zones == other.zones
            && self.steps == other.steps
            && self.weights == other.weights
            && self.kept == other.kept
            && self.inputs == other.inputs
            && self.outputs == other.outputs
    }
}

impl Plan {
    /// An id no other plan has had in this process.
    pub fn fresh() -> u64 {
        IDS.fetch_add(1, Ordering::Relaxed)
    }

    /// The weakest order any of its kernels sums in.
    pub fn order(&self) -> Order {
        let orders = self.steps.iter().map(|s| s.kernel.order());
        orders.max().unwrap_or(Order::Fixed)
    }

    pub fn zone(&self, buffer: u32) -> u32 {
        self.zones.get(buffer as usize).copied().unwrap_or(0)
    }

    /// Whether any buffer lies outside zone 0.
    pub fn zoned(&self) -> bool {
        self.zones.iter().any(|z| *z != 0)
    }

    /// Bytes in buffers no weight, input or kept row owns.
    pub fn scratch(&self) -> usize {
        let named = self
            .weights
            .iter()
            .map(|l| &l.slot)
            .chain(self.kept.iter().map(|(_, s)| s));
        let owned = |s: &Slot| self.buffers[s.buffer as usize];
        let held: usize = named.chain(&self.inputs).map(owned).sum();
        self.buffers.iter().sum::<usize>() - held
    }
}
