//! Gives each value a buffer, for one op order.

use crate::kernel::KernelRef;
use crate::kernels::Map;
use crate::plan::{Loaded, Output, Plan, Slot, Step};
use crate::select::{Home, Selected, Val, Value};

/// A last use that never comes: weights, inputs and outputs.
const FOREVER: usize = usize::MAX;

struct Buffers<'a> {
    s: &'a Selected,
    plan: Plan,
    slots: Vec<Option<Slot>>,
    /// Per value: the position of its last reader.
    last: Vec<usize>,
    free: Vec<u32>,
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
        ..Plan::default()
    };
    let mut b = Buffers {
        s,
        plan,
        slots: vec![None; s.values.len()],
        last,
        free: Vec::new(),
    };
    for (v, value) in s.values.iter().enumerate() {
        match value.home {
            Home::Weight(weight) => {
                let slot = b.alloc(Val(v as u32));
                b.plan.weights.push(Loaded { weight, slot });
            }
            Home::Kept { index, room } => {
                // The longest context fits, so the buffer never moves.
                let buffer = b.buffer(value.form.bytes(room), value.zone);
                let slot = Slot {
                    buffer,
                    offset: 0,
                    len: value.len,
                    form: value.form,
                };
                b.slots[v] = Some(slot);
                b.plan.kept.push((index, slot));
            }
            Home::Input(_) | Home::Temp => {}
        }
    }
    for v in &s.inputs {
        let slot = b.alloc(*v);
        b.plan.inputs.push(slot);
    }
    for (at, &i) in order.iter().enumerate() {
        b.step(at, i);
    }
    for (v, shape) in &s.outputs {
        let (slot, shape) = (b.slot(*v), shape.clone());
        b.plan.outputs.push(Output { slot, shape });
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

    /// Reuses a free buffer of the value's size, if any.
    fn alloc(&mut self, v: Val) -> Slot {
        let Value {
            len, zone, form, ..
        } = self.s.values[v.0 as usize];
        let bytes = form.bytes(len);
        let fits = |b: &u32| self.plan.buffers[*b as usize] == bytes && self.plan.zone(*b) == zone;
        let buffer = match self.free.iter().position(fits) {
            Some(at) => self.free.swap_remove(at),
            None => self.buffer(bytes, zone),
        };
        let slot = Slot {
            buffer,
            offset: 0,
            len,
            form,
        };
        self.slots[v.0 as usize] = Some(slot);
        slot
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
                let copy = self.alloc(op.write);
                let covers = op.covers.clone();
                self.plan.steps.push(Step {
                    kernel: KernelRef::new(Map(Vec::new())),
                    rule: op.rule,
                    reads: vec![reads[0]],
                    write: copy,
                    covers,
                    zone: op.zone,
                });
                reads[0] = copy;
            }
            reads[0]
        } else {
            // Kept rows have their buffer already.
            let held = self.slots[op.write.0 as usize];
            held.unwrap_or_else(|| self.alloc(op.write))
        };
        self.plan.steps.push(Step {
            kernel: op.kernel.clone(),
            rule: op.rule,
            reads,
            write,
            covers: op.covers.clone(),
            zone: op.zone,
        });
        for (k, v) in op.reads.iter().enumerate() {
            let repeat = op.reads[..k].contains(v);
            if dying[k] && Some(*v) != kept && !repeat {
                self.free.push(self.slot(*v).buffer);
            }
        }
    }
}
