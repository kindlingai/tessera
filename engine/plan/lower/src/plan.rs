//! A plan: kernel launches over buffers that stay on device.

use crate::abi::{Arg, Launch};
use crate::form::{F32_FORM, Load};
use crate::kernel::KernelRef;
use std::sync::atomic::{AtomicU64, Ordering};
use tessera_codegen::kept::Layout;
use tessera_graph::{NodeId, Order};

/// `len` elements of one buffer, `offset` bytes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub buffer: u32,
    pub offset: usize,
    pub len: usize,
    pub form: &'static dyn Load,
    pub dims: u8,
    pub shape: [u32; 4],
}

/// Packs up to four tensor axes.
pub(crate) fn pack_shape(axes: &[usize], len: usize) -> (u8, [u32; 4]) {
    if axes.is_empty() || axes.len() > 4 {
        return match u32::try_from(len) {
            Ok(n) if len > 0 => (1, [n, 0, 0, 0]),
            _ => (0, [0; 4]),
        };
    }
    let mut shape = [0u32; 4];
    for (dst, &src) in shape.iter_mut().zip(axes) {
        let Ok(n) = u32::try_from(src) else {
            return (0, [0; 4]);
        };
        *dst = n;
    }
    (axes.len() as u8, shape)
}

impl Slot {
    /// Elements from the start of a buffer, as f32.
    pub fn floats(buffer: u32, len: usize) -> Self {
        let (dims, shape) = pack_shape(&[len], len);
        Self {
            buffer,
            offset: 0,
            len,
            form: F32_FORM,
            dims,
            shape,
        }
    }

    /// The first byte's offset.
    pub fn start(&self) -> usize {
        self.offset
    }

    pub fn bytes(&self) -> usize {
        self.form.bytes(self.len)
    }

    /// The byte offset of element `skip`.
    pub fn at(&self, skip: usize) -> usize {
        self.offset + self.form.offset(skip)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub kernel: KernelRef,
    /// The proposing rule.
    pub rule: &'static str,
    pub reads: Vec<Slot>,
    pub write: Slot,
    /// Extra outputs this launch writes.
    pub also: Vec<Slot>,
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

    /// Bytes it reads and writes.
    pub fn bytes(&self) -> usize {
        let read: usize = self.reads.iter().map(|r| r.bytes()).sum();
        read + self.write.bytes() + self.also.iter().map(|w| w.bytes()).sum::<usize>()
    }

    /// The `i`th input's buffer, from its start.
    pub fn read(&self, i: usize) -> Arg {
        Arg::at(self.reads[i], 0)
    }

    pub fn out(&self) -> Arg {
        Arg::at(self.write, 0)
    }

    /// Output `i`: `0` is `write`, `1..` are `also`.
    pub fn out_at(&self, i: usize) -> Arg {
        match i {
            0 => self.out(),
            n => Arg::at(self.also[n - 1], 0),
        }
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
    pub fn of(&self, form: &dyn tessera_mem::form::Form, bytes: &[u8]) -> Option<Vec<u8>> {
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

/// Geometry of one kept-row buffer in `Plan::kept`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeptInfo {
    pub slots: u32,
    pub context: u32,
    pub width: u32,
    pub page_rows: u32,
    pub interleaved: bool,
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
    /// Steps after the forward ones that replay accepted rows.
    pub commits: usize,
    /// Trailing steps that copy slot 0's spare into its state.
    pub promotes: usize,
    /// Each used graph weight.
    pub weights: Vec<Loaded>,
    /// Each kept-row buffer, holding this launch's rows too.
    pub kept: Vec<(u32, Slot)>,
    /// Geometry for each entry of `kept`, in order.
    pub kept_info: Vec<KeptInfo>,
    /// Each slot's pages, when kept rows are paged.
    pub table: Option<Slot>,
    /// Physical pages in the kept pool, when paged.
    pub pool_pages: usize,
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

    /// This rank's `(axis, first, count)` slice of `weight`, if split.
    pub fn share(&self, weight: u32) -> Option<(u32, usize, usize)> {
        let l = self.weights.iter().find(|l| l.weight == weight)?;
        let s = l.share.filter(|_| l.part.is_none())?;
        Some((s.axis, s.first, s.count))
    }

    /// This rank's first-axis rows of `weight`, if split there.
    pub fn rows(&self, weight: u32) -> Option<(usize, usize)> {
        let (axis, first, count) = self.share(weight)?;
        (axis == 0).then_some((first, count))
    }

    /// The steps every launch runs.
    pub fn forward(&self) -> std::ops::Range<usize> {
        0..self.steps.len() - self.commits - self.promotes
    }

    /// The steps that replay a launch's leading rows.
    pub fn commit(&self) -> std::ops::Range<usize> {
        let first = self.forward().end;
        first..first + self.commits
    }

    /// The steps that keep the state after every row.
    pub fn promote(&self) -> std::ops::Range<usize> {
        self.commit().end..self.steps.len()
    }

    /// The weakest order any of its kernels sums in.
    pub fn order(&self) -> Order {
        let orders = self.steps.iter().map(|s| s.kernel.order());
        orders.max().unwrap_or(Order::Fixed)
    }

    pub fn zone(&self, buffer: u32) -> u32 {
        self.zones.get(buffer as usize).copied().unwrap_or(0)
    }

    /// Bytes of every weight the plan reads.
    pub fn weight_bytes(&self) -> usize {
        self.weights.iter().map(|w| w.slot.bytes()).sum()
    }

    /// Whether `step` reads a weight.
    pub fn reads_weight(&self, step: &Step) -> bool {
        let weight = |r: &Slot| self.weights.iter().any(|w| w.slot.buffer == r.buffer);
        step.reads.iter().any(weight)
    }

    /// Whether any buffer lies outside zone 0.
    pub fn zoned(&self) -> bool {
        self.zones.iter().any(|z| *z != 0)
    }

    /// Bytes in buffers no weight, kept row or table owns.
    pub fn scratch(&self) -> usize {
        let mut owned = vec![false; self.buffers.len()];
        for l in &self.weights {
            owned[l.slot.buffer as usize] = true;
        }
        for (_, s) in &self.kept {
            owned[s.buffer as usize] = true;
        }
        if let Some(s) = self.table {
            owned[s.buffer as usize] = true;
        }
        let mut total = 0;
        for (&bytes, held) in self.buffers.iter().zip(owned) {
            if !held {
                total += bytes;
            }
        }
        total
    }
}
