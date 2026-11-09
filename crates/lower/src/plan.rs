//! A plan: kernel launches over buffers that stay on device.

use crate::abi::{Arg, Launch};
use crate::kernel::KernelRef;
use graph::NodeId;

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

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub slot: Slot,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// Elements per buffer. Values that never overlap share one.
    pub buffers: Vec<usize>,
    pub steps: Vec<Step>,
    /// Each used graph weight, and where it loads.
    pub weights: Vec<(u32, Slot)>,
    /// One per input port of the entry.
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Output>,
}

impl Plan {
    /// Elements in buffers that hold no weight or input.
    pub fn scratch(&self) -> usize {
        let fixed = self.weights.iter().map(|(_, s)| s).chain(&self.inputs);
        let held: usize = fixed.map(|s| s.len).sum();
        self.buffers.iter().sum::<usize>() - held
    }
}
