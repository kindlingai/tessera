//! Packs values into contiguous arenas, for one op order.

use crate::kernel::KernelRef;
use crate::kernels::{Map, crosses};
use crate::plan::{KeptInfo, Loaded, Output, Plan, Share, Slot, Step, pack_shape};
use crate::select::{Home, Selected, Val, Value};
use std::collections::BTreeMap;
use tessera_graph::DType;

/// A last use that never comes: weights, inputs and outputs.
const FOREVER: usize = usize::MAX;

/// Byte alignment for every sub-allocated slot.
const ALIGN: usize = 256;

/// Aligned byte span of a slot.
fn span(bytes: usize) -> usize {
    bytes.max(1).next_multiple_of(ALIGN)
}

/// Free byte ranges in one zone's scratch arena.
struct Scratch {
    buffer: u32,
    size: usize,
    /// Disjoint, sorted, coalesced `[start, end)` ranges.
    free: Vec<(usize, usize)>,
}

impl Scratch {
    fn new(buffer: u32) -> Self {
        Self {
            buffer,
            size: 0,
            free: Vec::new(),
        }
    }

    /// Best-fit free range, else tail extension or growth.
    fn alloc(&mut self, bytes: usize) -> usize {
        let need = span(bytes);
        let mut best: Option<(usize, usize)> = None;
        for (i, &(lo, hi)) in self.free.iter().enumerate() {
            let len = hi - lo;
            if len >= need && best.is_none_or(|(_, b)| len < b) {
                best = Some((i, len));
            }
        }
        if let Some((i, len)) = best {
            let (lo, hi) = self.free[i];
            if len == need {
                self.free.remove(i);
            } else {
                self.free[i] = (lo + need, hi);
            }
            return lo;
        }
        if let Some(&(lo, hi)) = self.free.last()
            && hi == self.size
        {
            self.free.pop();
            self.size = lo + need;
            return lo;
        }
        let lo = self.size;
        self.size += need;
        lo
    }

    /// A range past every other, free or used.
    fn fresh(&mut self, bytes: usize) -> usize {
        let lo = self.size;
        self.size += span(bytes);
        lo
    }

    /// Returns a range and coalesces adjacent holes.
    fn free(&mut self, start: usize, bytes: usize) {
        let mut lo = start;
        let mut hi = start + span(bytes);
        let mut pos = self.free.partition_point(|&(s, _)| s < lo);
        if pos > 0 && self.free[pos - 1].1 == lo {
            pos -= 1;
            lo = self.free[pos].0;
            self.free.remove(pos);
        }
        if pos < self.free.len() && self.free[pos].0 == hi {
            hi = self.free[pos].1;
            self.free.remove(pos);
        }
        self.free.insert(pos, (lo, hi));
    }
}

struct Buffers<'a> {
    s: &'a Selected,
    plan: Plan,
    slots: Vec<Option<Slot>>,
    /// Per value: the position of its last reader.
    last: Vec<usize>,
    /// Execution lane for each step in `order`.
    lanes: Vec<u8>,
    /// Temp values freed at a later join step.
    deferred: Vec<Vec<Val>>,
    /// One scratch arena per zone, made on first use.
    scratch: BTreeMap<u32, Scratch>,
}

/// Buffers and steps for `s`, its ops in `order`.
pub fn assign(s: &Selected, order: &[usize]) -> Plan {
    let mut last = vec![0; s.values.len()];
    for (at, &i) in order.iter().enumerate() {
        for v in &s.ops[i].reads {
            last[v.0 as usize] = at;
        }
    }
    // Keep overlapped comm inputs live until compute joins.
    let mut lanes = vec![0u8; order.len()];
    let mut extended = vec![false; s.values.len()];
    for (at, &i) in order.iter().enumerate() {
        let op = &s.ops[i];
        if !crosses(op.kernel.name()) {
            continue;
        }
        let join = order[at + 1..]
            .iter()
            .enumerate()
            .find_map(|(d, &j)| {
                let later = &s.ops[j];
                let reads_out = later.reads.contains(&op.write)
                    || op.also.iter().any(|w| later.reads.contains(w));
                (!crosses(later.kernel.name()) && reads_out).then_some(at + 1 + d)
            })
            .unwrap_or(FOREVER);
        let end = join.min(order.len());
        if !order[at + 1..end]
            .iter()
            .any(|&j| !crosses(s.ops[j].kernel.name()))
        {
            continue;
        }
        lanes[at] = 1;
        for v in &op.reads {
            if join > last[v.0 as usize] {
                last[v.0 as usize] = join;
                extended[v.0 as usize] = true;
            }
        }
    }
    for (v, value) in s.values.iter().enumerate() {
        if value.home != Home::Temp {
            last[v] = FOREVER;
        }
    }
    for (v, _) in &s.outputs {
        last[v.0 as usize] = FOREVER;
    }
    // A commit reads its step's inputs again.
    for op in s.ops.iter().filter(|op| op.kernel.commit().is_some()) {
        for v in &op.reads {
            last[v.0 as usize] = FOREVER;
        }
    }
    let mut deferred = vec![Vec::new(); order.len()];
    for (v, &end) in last.iter().enumerate() {
        let val = Val(v as u32);
        if extended[v]
            && end < order.len()
            && s.values[v].home == Home::Temp
            && !s.ops[order[end]].reads.contains(&val)
        {
            deferred[end].push(val);
        }
    }
    let plan = Plan {
        id: Plan::fresh(),
        ranks: s.ranks,
        rank: s.rank,
        ..Plan::default()
    };
    let mut b = Buffers {
        s,
        plan,
        slots: vec![None; s.values.len()],
        last,
        lanes,
        deferred,
        scratch: BTreeMap::new(),
    };
    b.pack_weights();
    b.pack_kept();
    for v in &s.inputs {
        let slot = b.alloc_scratch(*v);
        b.plan.inputs.push(slot);
    }
    for (at, &i) in order.iter().enumerate() {
        b.step(at, i);
    }
    b.commits();
    for (v, shape) in &s.outputs {
        let (slot, shape) = (b.slot(*v), shape.clone());
        b.plan.outputs.push(Output { slot, shape });
    }
    for arena in b.scratch.values() {
        b.plan.buffers[arena.buffer as usize] = arena.size;
    }
    b.plan
}

impl Buffers<'_> {
    fn buffer(&mut self, bytes: usize, zone: u32) -> u32 {
        self.plan.buffers.push(bytes);
        self.plan.zones.push(zone);
        self.plan.buffers.len() as u32 - 1
    }

    fn slot(&self, v: Val) -> Slot {
        self.slots[v.0 as usize].expect("written before it is read")
    }

    /// Packs weights into one contiguous arena per zone.
    fn pack_weights(&mut self) {
        let mut items: Vec<(u32, u32, Option<u8>, usize)> = Vec::new();
        for (v, value) in self.s.values.iter().enumerate() {
            match value.home {
                Home::Weight(w) => items.push((value.zone, w, None, v)),
                Home::Part(w, p) => items.push((value.zone, w, Some(p), v)),
                Home::Kept { .. } | Home::Input(_) | Home::Temp => {}
            }
        }
        items.sort_unstable();
        let mut arenas: BTreeMap<u32, (u32, usize)> = BTreeMap::new();
        type Key = (u32, u32, Option<u8>, &'static str, usize, Option<Share>);
        let mut seen: Vec<(Key, Slot)> = Vec::new();
        for (zone, weight, part, v) in items {
            let value = self.s.values[v];
            let key: Key = (
                zone,
                weight,
                part,
                value.form.name(),
                value.len,
                value.share,
            );
            if let Some((_, existing)) = seen.iter().find(|(k, _)| *k == key) {
                self.slots[v] = Some(*existing);
                continue;
            }
            let (buffer, cursor) = match arenas.get_mut(&zone) {
                Some(entry) => entry,
                None => {
                    let buf = self.buffer(0, zone);
                    arenas.entry(zone).or_insert((buf, 0))
                }
            };
            let offset = *cursor;
            *cursor += span(value.form.bytes(value.len));
            let slot = Slot {
                buffer: *buffer,
                offset,
                len: value.len,
                form: value.form,
                dims: value.dims,
                shape: value.shape,
            };
            self.slots[v] = Some(slot);
            seen.push((key, slot));
            self.plan.weights.push(Loaded {
                weight,
                slot,
                share: value.share,
                part,
            });
        }
        for (buffer, size) in arenas.into_values() {
            self.plan.buffers[buffer as usize] = size;
        }
    }

    /// Packs kept rows into one contiguous arena per zone.
    fn pack_kept(&mut self) {
        let mut items: Vec<(u32, u32, usize, KeptInfo, usize)> = Vec::new();
        for (v, value) in self.s.values.iter().enumerate() {
            if let Home::Kept { index, room, info } = value.home {
                items.push((value.zone, index, room, info, v));
            }
        }
        items.sort_unstable();
        let mut arenas: BTreeMap<u32, (u32, usize)> = BTreeMap::new();
        for (zone, index, room, info, v) in items {
            let value = self.s.values[v];
            let (buffer, cursor) = match arenas.get_mut(&zone) {
                Some(entry) => entry,
                None => {
                    let buf = self.buffer(0, zone);
                    arenas.entry(zone).or_insert((buf, 0))
                }
            };
            let offset = *cursor;
            *cursor += span(value.form.bytes(room));
            let slot = Slot {
                buffer: *buffer,
                offset,
                len: value.len,
                form: value.form,
                dims: value.dims,
                shape: value.shape,
            };
            self.slots[v] = Some(slot);
            self.plan.kept.push((index, slot));
            self.plan.kept_info.push(info);
            // One table beside kept rows names every lane's pages.
            if self.s.table > 0 && self.plan.table.is_none() {
                let form = tessera_codegen::form::of(DType::U32);
                let split = self.s.index_table > 0;
                let total = self.s.table + usize::from(split) * self.s.index_table;
                let buffer = self.buffer(form.bytes(total), zone);
                let (dims, shape) = pack_shape(&[self.s.table], self.s.table);
                self.plan.table = Some(Slot {
                    buffer,
                    offset: 0,
                    len: self.s.table,
                    form,
                    dims,
                    shape,
                });
                self.plan.pool_pages = self.s.pool_pages;
                if split {
                    let (idims, ishape) = pack_shape(&[self.s.index_table], self.s.index_table);
                    self.plan.index_table = Some(Slot {
                        buffer,
                        offset: form.bytes(self.s.table),
                        len: self.s.index_table,
                        form,
                        dims: idims,
                        shape: ishape,
                    });
                    self.plan.index_pool_pages = self.s.index_pool_pages;
                }
            }
        }
        for (buffer, size) in arenas.into_values() {
            self.plan.buffers[buffer as usize] = size;
        }
    }

    /// Sub-allocates `v` in its zone's scratch arena.
    fn alloc_scratch(&mut self, v: Val) -> Slot {
        let Value {
            len,
            zone,
            form,
            dims,
            shape,
            ..
        } = self.s.values[v.0 as usize];
        let bytes = form.bytes(len);
        if !self.scratch.contains_key(&zone) {
            let buffer = self.buffer(0, zone);
            self.scratch.insert(zone, Scratch::new(buffer));
        }
        let arena = self.scratch.get_mut(&zone).expect("inserted above");
        let offset = arena.alloc(bytes);
        let slot = Slot {
            buffer: arena.buffer,
            offset,
            len,
            form,
            dims,
            shape,
        };
        self.slots[v.0 as usize] = Some(slot);
        slot
    }

    /// Gives each peeking step a spare, commit and promotion.
    fn commits(&mut self) {
        let mut promotes = Vec::new();
        for at in 0..self.plan.steps.len() {
            let Some(kernel) = self.plan.steps[at].kernel.commit() else {
                continue;
            };
            let mut step = self.plan.steps[at].clone();
            let state = *step.reads.last().expect("a kept state");
            let zone = self.plan.zone(step.write.buffer);
            let arena = self.scratch.get_mut(&zone).expect("written in scratch");
            // Later forward steps must not reuse the spare.
            let mut spare = state;
            spare.buffer = arena.buffer;
            spare.offset = arena.fresh(state.bytes());
            self.plan.steps[at].also = vec![spare];
            // A replay writes its outputs to scratch of its own.
            step.write.offset = arena.alloc(step.write.bytes());
            step.kernel = kernel;
            // A scan stores only the lanes of slots it read.
            let kept = self.plan.kept.iter().position(|(_, s)| *s == state);
            let lanes = self.plan.kept_info[kept.expect("a kept state")].slots as usize;
            let lane = |mut slot: Slot| {
                slot.len /= lanes;
                (slot.dims, slot.shape) = pack_shape(&[slot.len], slot.len);
                slot
            };
            promotes.push(Step {
                kernel: KernelRef::new(Map::copy(spare.form, state.form)),
                reads: vec![lane(spare)],
                write: lane(state),
                covers: Vec::new(),
                ..step.clone()
            });
            self.plan.steps.push(step);
            self.plan.commits += 1;
        }
        self.plan.promotes = promotes.len();
        self.plan.steps.extend(promotes);
    }

    fn free_scratch(&mut self, slot: Slot) {
        let zone = self.plan.zone(slot.buffer);
        let arena = self.scratch.get_mut(&zone).expect("allocated in scratch");
        arena.free(slot.offset, slot.bytes());
    }

    fn step(&mut self, at: usize, i: usize) {
        let op = &self.s.ops[i];
        let mut reads: Vec<Slot> = op.reads.iter().map(|v| self.slot(*v)).collect();
        let dying: Vec<bool> = op
            .reads
            .iter()
            .map(|v| self.last[v.0 as usize] == at)
            .collect();
        let mut kept = None;
        let write = if op.kernel.overwrites() {
            let first = op.reads[0];
            if dying[0] {
                // Nothing else reads it, so rotate where it lies.
                kept = Some(first);
                self.slots[op.write.0 as usize] = Some(reads[0]);
            } else {
                let copy = self.alloc_scratch(op.write);
                let covers = op.covers.clone();
                self.plan.steps.push(Step {
                    kernel: KernelRef::new(Map::copy(reads[0].form, copy.form)),
                    rule: op.rule,
                    site: op.site.clone(),
                    reads: vec![reads[0]],
                    write: copy,
                    also: Vec::new(),
                    covers,
                    zone: op.zone,
                    lane: 0,
                    table: self.plan.table,
                    index_table: self.plan.index_table,
                });
                reads[0] = copy;
            }
            reads[0]
        } else {
            // Kept rows have their slot already.
            let held = self.slots[op.write.0 as usize];
            held.unwrap_or_else(|| self.alloc_scratch(op.write))
        };
        let also: Vec<Slot> = op
            .also
            .iter()
            .map(|&v| {
                let held = self.slots[v.0 as usize];
                held.unwrap_or_else(|| self.alloc_scratch(v))
            })
            .collect();
        self.plan.steps.push(Step {
            kernel: op.kernel.clone(),
            rule: op.rule,
            site: op.site.clone(),
            reads,
            write,
            also,
            covers: op.covers.clone(),
            zone: op.zone,
            lane: self.lanes[at],
            table: self.plan.table,
            index_table: self.plan.index_table,
        });
        for (k, v) in op.reads.iter().enumerate() {
            let repeat = op.reads[..k].contains(v);
            if dying[k] && Some(*v) != kept && !repeat {
                self.free_scratch(self.slot(*v));
            }
        }
        // Unread temporary outputs free immediately after this step.
        if self.last[op.write.0 as usize] <= at
            && self.s.values[op.write.0 as usize].home == Home::Temp
        {
            self.free_scratch(write);
        }
        for &v in &op.also {
            if self.last[v.0 as usize] <= at && self.s.values[v.0 as usize].home == Home::Temp {
                self.free_scratch(self.slot(v));
            }
        }
        let deferred = std::mem::take(&mut self.deferred[at]);
        for v in deferred {
            if let Some(slot) = self.slots[v.0 as usize] {
                self.free_scratch(slot);
            }
        }
    }
}
