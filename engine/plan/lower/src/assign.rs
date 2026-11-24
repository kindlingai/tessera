//! Packs values into contiguous arenas, for one op order.

use crate::kernel::KernelRef;
use crate::kernels::Map;
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
    for (v, value) in s.values.iter().enumerate() {
        if value.home != Home::Temp {
            last[v] = FOREVER;
        }
    }
    for (v, _) in &s.outputs {
        last[v.0 as usize] = FOREVER;
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
                let buffer = self.buffer(form.bytes(self.s.table), zone);
                let (dims, shape) = pack_shape(&[self.s.table], self.s.table);
                self.plan.table = Some(Slot {
                    buffer,
                    offset: 0,
                    len: self.s.table,
                    form,
                    dims,
                    shape,
                });
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
                    reads: vec![reads[0]],
                    write: copy,
                    also: Vec::new(),
                    covers,
                    zone: op.zone,
                    table: self.plan.table,
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
            reads,
            write,
            also,
            covers: op.covers.clone(),
            zone: op.zone,
            table: self.plan.table,
        });
        for (k, v) in op.reads.iter().enumerate() {
            let repeat = op.reads[..k].contains(v);
            if dying[k] && Some(*v) != kept && !repeat {
                self.free_scratch(self.slot(*v));
            }
        }
    }
}
