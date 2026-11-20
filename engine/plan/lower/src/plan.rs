//! A plan: kernel launches over buffers that stay on device.

use crate::abi::{Arg, Launch};
use crate::form::{F32_FORM, Load};
use crate::kernel::KernelRef;
use sensai_codegen::kept::Layout;
use sensai_graph::{NodeId, Order};
use std::sync::atomic::{AtomicU64, Ordering};

/// `len` elements of one buffer, `offset` elements in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub buffer: u32,
    pub offset: usize,
    pub len: usize,
    pub form: &'static dyn Load,
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
    /// The proposing rule.
    pub rule: &'static str,
    pub reads: Vec<Slot>,
    pub write: Slot,
    /// The graph nodes this launch computes, or helps compute.
    pub covers: Vec<NodeId>,
    /// The zone it runs in.
    pub zone: u32,
    /// The plan's page table, when kept rows are paged.
    pub table: Option<Slot>,
}

impl Step {
    pub fn launches(&self) -> Vec<Launch> {
        self.kernel.launches(self)
    }

    /// The `i`th input's buffer, from its start.
    pub fn read(&self, i: usize) -> Arg {
        Arg::at(self.reads[i], 0)
    }

    pub fn out(&self) -> Arg {
        Arg::at(self.write, 0)
    }

    /// The page table, if `layout` reads one.
    pub fn kept(&self, layout: &dyn Layout) -> Option<Arg> {
        let table = || self.table.expect("a paged plan has a table");
        layout.page().map(|_| Arg::at(table(), 0))
    }
}

/// A rank's part of a split weight, `count` from `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Share {
    pub axis: u32,
    pub first: usize,
    pub count: usize,
    /// The axis's whole length, and the elements after it.
    pub whole: usize,
    pub inner: usize,
}

impl Share {
    /// This rank's part of `bytes`, packed as `form`.
    pub fn of(&self, form: &dyn sensai_mem::form::Form, bytes: &[u8]) -> Option<Vec<u8>> {
        let outer = form.elements(bytes.len()) / (self.whole * self.inner).max(1);
        form.share(
            bytes,
            [outer, self.whole, self.inner],
            [self.first, self.count],
        )
    }
}

/// A graph weight and where it loads, with its form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub weight: u32,
    pub slot: Slot,
    /// Set when each rank loads its own slice.
    pub share: Option<Share>,
    /// This slot's part of a split file.
    pub part: Option<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub slot: Slot,
    pub shape: Vec<usize>,
}

/// Equal plans run alike, whatever their ids.
#[derive(Clone, Debug, Default, derive_more::PartialEq)]
pub struct Plan {
    /// Tells plans apart without comparing them. Zero means unknown.
    #[partial_eq(skip)]
    pub id: u64,
    /// Bytes per buffer. Values that never overlap share one.
    pub buffers: Vec<usize>,
    /// The zone each buffer lies in. Empty means zone 0.
    pub zones: Vec<u32>,
    /// Ranks that run together. Zero means one.
    pub ranks: u32,
    /// Which of them this plan is for.
    pub rank: u32,
    pub steps: Vec<Step>,
    /// Each used graph weight.
    pub weights: Vec<Loaded>,
    /// Each kept-row buffer, holding this launch's rows too.
    pub kept: Vec<(u32, Slot)>,
    /// Each slot's pages, when kept rows are paged.
    pub table: Option<Slot>,
    /// One per input port of the entry.
    pub inputs: Vec<Slot>,
    pub outputs: Vec<Output>,
}

static IDS: AtomicU64 = AtomicU64::new(1);

impl Plan {
    /// An id unique among plans in this process.
    pub fn fresh() -> u64 {
        IDS.fetch_add(1, Ordering::Relaxed)
    }

    /// This rank's first-axis rows of `weight`, if split there.
    pub fn rows(&self, weight: u32) -> Option<(usize, usize)> {
        let l = self.weights.iter().find(|l| l.weight == weight)?;
        let share = l.share.filter(|s| s.axis == 0 && l.part.is_none())?;
        Some((share.first, share.count))
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
            .chain(self.kept.iter().map(|(_, s)| s))
            .chain(&self.table);
        let owned = |s: &Slot| self.buffers[s.buffer as usize];
        let held: usize = named.chain(&self.inputs).map(owned).sum();
        self.buffers.iter().sum::<usize>() - held
    }
}
