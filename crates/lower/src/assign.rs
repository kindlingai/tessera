//! Gives each value a buffer, for one op order.

use crate::plan::{Kernel, Output, Plan, Slot, Step};
use crate::select::{Home, Selected, Val};

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
    let mut b = Buffers {
        s,
        plan: Plan::default(),
        slots: vec![None; s.values.len()],
        last,
        free: Vec::new(),
    };
    for (v, value) in s.values.iter().enumerate() {
        match value.home {
            Home::Weight(w) => {
                let slot = b.alloc(Val(v as u32));
                b.plan.weights.push((w, slot));
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
    fn slot(&self, v: Val) -> Slot {
        self.slots[v.0 as usize].expect("written before it is read")
    }

    /// Reuses a free buffer of the value's size, if any.
    fn alloc(&mut self, v: Val) -> Slot {
        let len = self.s.values[v.0 as usize].len;
        let fits = |b: &u32| self.plan.buffers[*b as usize] == len;
        let buffer = match self.free.iter().position(fits) {
            Some(at) => self.free.swap_remove(at),
            None => {
                self.plan.buffers.push(len);
                self.plan.buffers.len() as u32 - 1
            }
        };
        let slot = Slot { buffer, len };
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
                    kernel: Kernel::Map(Vec::new()),
                    reads: vec![reads[0]],
                    write: copy,
                    covers,
                });
                reads[0] = copy;
            }
            reads[0]
        } else {
            self.alloc(op.write)
        };
        self.plan.steps.push(Step {
            kernel: op.kernel.clone(),
            reads,
            write,
            covers: op.covers.clone(),
        });
        for (k, v) in op.reads.iter().enumerate() {
            let repeat = op.reads[..k].contains(v);
            if dying[k] && Some(*v) != kept && !repeat {
                self.free.push(self.slot(*v).buffer);
            }
        }
    }
}
