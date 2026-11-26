//! Steps one persistent grid runs, with its sums inside.

use crate::Error;
use crate::profile::Marks;
use tessera_lower::{Arg, Extent, KernelRef, Launch as Call, Plan, Slot, Step};
use tessera_mem::form::Read;

/// Role of one table item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemKind {
    Input,
    Output,
    Scratch,
}

/// Companion metadata for a packed storage form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormMeta {
    pub name: &'static str,
    pub rows: u32,
    pub cols: u32,
    pub suh: Option<u64>,
    pub svh: Option<u64>,
}

/// One table item: address, role, shape, share and form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub at: u64,
    pub bytes: usize,
    pub kind: ItemKind,
    pub elem: &'static str,
    pub dims: u32,
    pub shape: [u32; 4],
    pub share: [u32; 4],
    pub form: Option<FormMeta>,
}

/// One kept-row descriptor for the KV convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeptDesc {
    pub base: u64,
    pub pages: Option<u64>,
    pub len: Option<u64>,
    pub slots: u32,
    pub context: u32,
    pub pages_per_slot: u32,
    pub page_rows: u32,
    pub row_bytes: u32,
    pub elem: &'static str,
    pub interleaved: bool,
}

/// Table counts a grid opens for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layout {
    pub items: usize,
    pub forms: usize,
    pub kept: usize,
    pub pages: usize,
}

/// Runs resident steps. One launch per step, ended each time.
pub trait Grid {
    /// Readies `kernel` for launches matching `layout`.
    fn open(&mut self, kernel: &KernelRef, layout: Layout) -> Result<(), Error>;
    /// One launch of `kernel`. It returns once the grid ends.
    fn launch(
        &mut self,
        kernel: &KernelRef,
        items: &[Item],
        kept: &[KeptDesc],
        values: &[u32],
    ) -> Result<(), Error>;
    /// Enables or disables resident profiling telemetry and marks.
    fn profiling(&mut self, _on: bool) {}
    /// Replaces `entry`'s source with `source`, or restores planned text.
    fn replace(&mut self, _entry: &str, _source: Option<String>) {}
    /// Drains mark samples gathered by recent launches.
    fn take_marks(&mut self) -> Vec<(String, Marks)> {
        Vec::new()
    }
}

/// Matrix rows and columns from a packed shape.
fn matrix_dims(dims: u32, shape: [u32; 4]) -> (u32, u32) {
    match dims {
        0 => (0, 0),
        1 => (1, shape[0]),
        d => (shape[d as usize - 2], shape[d as usize - 1]),
    }
}

/// Companion `suh` and `svh` device addresses for a rotated weight.
fn companions(
    plan: &Plan,
    weight: u32,
    address: &impl Fn(Slot) -> u64,
) -> (Option<u64>, Option<u64>) {
    let mut suh = None;
    let mut svh = None;
    for l in &plan.weights {
        if l.weight != weight {
            continue;
        }
        let at = address(l.slot) + l.slot.start() as u64;
        match l.part {
            Some(1) => suh = Some(at),
            Some(2) => svh = Some(at),
            _ => {}
        }
    }
    (suh, svh)
}

/// A bound launch as items and values, by argument order.
pub(crate) fn table(
    plan: &Plan,
    step: &Step,
    launch: &Call,
    address: &impl Fn(Slot) -> u64,
) -> Result<(Vec<Item>, Vec<u32>), Error> {
    let (mut items, mut values) = (Vec::new(), Vec::new());
    let mut buf_index = 0usize;
    for arg in &launch.args {
        match arg {
            Arg::Buffer(slot, Extent::Fixed(skip)) => {
                let kind = if buf_index >= step.reads.len()
                    && (*slot == step.write || step.also.contains(slot))
                {
                    ItemKind::Output
                } else {
                    ItemKind::Input
                };
                let at = address(*slot) + slot.at(*skip) as u64;
                let bytes = slot.bytes() - (slot.at(*skip) - slot.start());
                let (dims, shape) = if *skip == 0 && slot.dims > 0 {
                    (u32::from(slot.dims), slot.shape)
                } else {
                    (1, [bytes as u32, 0, 0, 0])
                };
                let loaded = plan
                    .weights
                    .iter()
                    .find(|l| l.slot.buffer == slot.buffer && l.slot.offset == slot.offset);
                let share = match loaded.and_then(|l| l.share) {
                    Some(s) => [s.axis, s.first as u32, s.count as u32, s.whole as u32],
                    None => [0; 4],
                };
                let form = if slot.form.read() == Read::Decode {
                    let (rows, cols) = matrix_dims(dims, shape);
                    let (suh, svh) = match loaded.filter(|l| l.part == Some(0)) {
                        Some(l) => companions(plan, l.weight, address),
                        None => (None, None),
                    };
                    Some(FormMeta {
                        name: slot.form.name(),
                        rows,
                        cols,
                        suh,
                        svh,
                    })
                } else {
                    None
                };
                items.push(Item {
                    at,
                    bytes,
                    kind,
                    elem: slot.form.name(),
                    dims,
                    shape,
                    share,
                    form,
                });
                buf_index += 1;
            }
            Arg::KeptHost(_) => values.push(0),
            arg => values.push(arg.u32().ok_or(tessera_lower::Error::Unbound)?),
        }
    }
    Ok((items, values))
}

/// Kept-row table descriptors for the KV convention.
pub(crate) fn kept_tables(plan: &Plan, address: &impl Fn(Slot) -> u64) -> Vec<KeptDesc> {
    let pages = plan.table.map(|t| address(t) + t.start() as u64);
    let mut out = Vec::with_capacity(plan.kept.len());
    for ((_, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
        let page_rows = info.page_rows.max(1);
        out.push(KeptDesc {
            base: address(*slot) + slot.start() as u64,
            pages,
            len: None,
            slots: info.slots,
            context: info.context,
            pages_per_slot: info.context.div_ceil(page_rows),
            page_rows,
            row_bytes: slot.form.bytes(info.width as usize) as u32,
            elem: slot.form.name(),
            interleaved: info.interleaved,
        });
    }
    out
}

/// Control table counts for a resident step.
pub(crate) fn step_layout(plan: &Plan, step: &Step) -> Layout {
    let mut items = 0usize;
    let mut forms = 0usize;
    for launch in step.launches() {
        let mut n_items = 0;
        let mut n_forms = 0;
        for arg in &launch.args {
            if let Arg::Buffer(slot, _) = arg {
                n_items += 1;
                if slot.form.read() == Read::Decode {
                    n_forms += 1;
                }
            }
        }
        items = items.max(n_items);
        forms = forms.max(n_forms);
    }
    let kept = plan.kept.len().min(plan.kept_info.len());
    let pages = match (plan.table, plan.kept_info.first()) {
        (None, Some(info)) if info.interleaved => (info.slots * info.context) as usize,
        (None, Some(info)) => info.slots as usize,
        _ => 0,
    };
    Layout {
        items,
        forms,
        kept,
        pages,
    }
}
