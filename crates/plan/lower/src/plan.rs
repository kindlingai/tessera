//! A plan: kernel launches over buffers that stay on device.

use crate::abi::{Arg, Launch};
use crate::form::Form;
use crate::kernel::KernelRef;
use graph::{NodeId, Order};

/// Leading elements of one buffer. Every element is four bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slot {
    pub buffer: u32,
    pub len: usize,
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

/// One graph weight, where it loads and in what form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub weight: u32,
    pub slot: Slot,
    pub form: Form,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub slot: Slot,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// Elements per buffer. Values that never overlap share one.
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

impl Plan {
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

    /// Elements in buffers no weight, input or kept row owns.
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
