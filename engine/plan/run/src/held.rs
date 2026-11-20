//! Buffers a runner keeps across plans.

use crate::Error;
use sensai_lower::{Plan, Slot};
use sensai_mem::{Buffer, View, ViewMut};
use std::collections::HashMap;

/// What owns a buffer that outlives a plan.
#[derive(Clone, Copy)]
enum Owner {
    Weight((u32, Option<u8>)),
    Kept(u32),
    Table,
}

/// Buffers that outlive one plan.
#[derive(Default)]
pub struct Held {
    /// Every live buffer.
    all: Vec<Buffer>,
    /// By weight, and part when a file splits.
    weights: HashMap<(u32, Option<u8>), usize>,
    kept: HashMap<u32, usize>,
    /// The page table, when kept rows are paged.
    table: Option<usize>,
    /// Where each of the plan's buffers lies in `all`.
    bound: Vec<usize>,
}

impl Held {
    pub fn get(&self, slot: Slot) -> &Buffer {
        &self.all[self.bound[slot.buffer as usize]]
    }

    /// These buffers' bytes, each named once, for the host.
    pub fn mapped_mut(&mut self, buffers: &[u32]) -> Vec<&mut [u8]> {
        let wanted: Vec<usize> = buffers.iter().map(|b| self.bound[*b as usize]).collect();
        let mut out: Vec<Option<&mut [u8]>> = (0..wanted.len()).map(|_| None).collect();
        for (at, buffer) in self.all.iter_mut().enumerate() {
            if let Some(i) = wanted.iter().position(|w| *w == at) {
                out[i] = buffer.backing_mut().mapped_mut();
            }
        }
        out.into_iter().map(|o| o.expect("host memory")).collect()
    }

    pub fn view(&self, slot: Slot) -> View<'_> {
        self.get(slot)
            .view(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// `bytes`, packed in the form, into the slot.
    pub fn upload(&mut self, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        let at = self.bound[slot.buffer as usize];
        Ok(self.all[at].backing_mut().upload(slot.start(), bytes)?)
    }

    pub fn view_mut(&mut self, slot: Slot) -> ViewMut<'_> {
        let at = self.bound[slot.buffer as usize];
        self.all[at].view_mut(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// Binds `plan`'s buffers. Weights, kept rows and tables stay.
    pub fn bind(
        &mut self,
        plan: &Plan,
        mut alloc: impl FnMut(u32, usize) -> Result<Buffer, Error>,
    ) -> Result<(), Error> {
        let old = std::mem::take(&mut self.all).into_iter().map(Some);
        let mut old: Vec<Option<Buffer>> = old.collect();
        let named = self.weights.values_mut().chain(self.kept.values_mut());
        for at in named.chain(&mut self.table) {
            self.all.push(old[*at].take().expect("each is held once"));
            *at = self.all.len() - 1;
        }
        // The last plan's scratch buffers.
        let mut spare: Vec<Buffer> = old.into_iter().flatten().collect();
        let mut owner = vec![None; plan.buffers.len()];
        for l in &plan.weights {
            owner[l.slot.buffer as usize] = Some(Owner::Weight((l.weight, l.part)));
        }
        for (index, slot) in &plan.kept {
            owner[slot.buffer as usize] = Some(Owner::Kept(*index));
        }
        if let Some(slot) = plan.table {
            owner[slot.buffer as usize] = Some(Owner::Table);
        }
        self.bound.clear();
        for (b, (&bytes, owner)) in plan.buffers.iter().zip(owner).enumerate() {
            let zone = plan.zone(b as u32);
            // Named buffers fit exactly. Scratch may be larger.
            let fits = |buffer: &Buffer| buffer.zone == zone && buffer.bytes() >= bytes;
            let held = match owner {
                Some(Owner::Kept(index)) => self.kept.get(&index).copied(),
                Some(Owner::Weight(key)) => self.weights.get(&key).copied(),
                Some(Owner::Table) => self.table,
                None => spare.iter().position(fits).map(|at| {
                    self.all.push(spare.swap_remove(at));
                    self.all.len() - 1
                }),
            };
            let same = |buffer: &Buffer| buffer.zone == zone && buffer.bytes() == bytes;
            let at = match held.filter(|at| owner.is_none() || same(&self.all[*at])) {
                Some(at) => at,
                None => {
                    // Scratch grows in steps, so a growing context reuses it.
                    let want = match owner {
                        Some(_) => bytes,
                        None => bytes.next_power_of_two(),
                    };
                    self.all.push(alloc(zone, want)?);
                    self.all.len() - 1
                }
            };
            match owner {
                Some(Owner::Kept(index)) => _ = self.kept.insert(index, at),
                Some(Owner::Weight(key)) => _ = self.weights.insert(key, at),
                Some(Owner::Table) => self.table = Some(at),
                None => {}
            }
            self.bound.push(at);
        }
        Ok(())
    }
}
