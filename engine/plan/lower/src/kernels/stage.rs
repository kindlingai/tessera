//! Host bytes copied to the device ahead of their reader.

use crate::abi::{Launch, Source};
use crate::hazard::{self, earliest};
use crate::kernels::Map;
use crate::plan::{Plan, Slot, Step};
use crate::{Kernel, KernelRef};
use std::collections::HashMap;

/// Copies host bytes for the weights its slots name.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub bytes: usize,
    /// The runner's copy list for this step.
    pub batch: u32,
}

impl Stage {
    /// Its kernel name. Runners find copies by it.
    pub const NAME: &'static str = "stage";

    /// A step on the copy lane that stages `weight`.
    pub fn step(self, weight: Slot, evicted: Vec<Slot>) -> Step {
        Step {
            kernel: KernelRef::new(self),
            rule: Self::NAME,
            site: None,
            reads: Vec::new(),
            write: weight,
            also: evicted,
            covers: Vec::new(),
            zone: 0,
            lane: hazard::COPY_LANE,
            table: None,
            index_table: None,
        }
    }
}

impl Kernel for Stage {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    // The runner copies, so nothing compiles.
    fn source(&self) -> Source {
        Map::plain(Vec::new()).source()
    }

    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }

    fn work(&self, _: usize) -> f64 {
        self.bytes as f64
    }

    fn staged(&self) -> Option<u32> {
        Some(self.batch)
    }
}

/// VRAM for staged experts: `parts` of `slots` each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ring {
    pub parts: usize,
    pub slots: usize,
}

/// One copy step's work: experts into a ring part.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Batch {
    /// The paged weight, by pager entry.
    pub weight: usize,
    pub experts: Vec<usize>,
    pub part: usize,
    /// The batch whose experts go back to host memory first.
    pub evicts: Option<usize>,
}

/// Every batch a plan has asked for, each made once.
#[derive(Default)]
pub struct Batches {
    list: Vec<Batch>,
    ids: HashMap<Batch, usize>,
}

impl Batches {
    fn id(&mut self, batch: Batch) -> usize {
        if let Some(&id) = self.ids.get(&batch) {
            return id;
        }
        self.list.push(batch.clone());
        self.ids.insert(batch, self.list.len() - 1);
        self.list.len() - 1
    }

    pub fn list(&self) -> &[Batch] {
        &self.list
    }
}

fn same(a: Slot, b: Slot) -> bool {
    a.buffer == b.buffer && a.offset == b.offset
}

/// Adds copy steps for the experts `pick` names.
pub fn place(
    plan: &mut Plan,
    // Each entry's slot and bytes per expert.
    weights: &[Option<(Slot, usize)>],
    ring: Ring,
    batches: &mut Batches,
    // Copies wait for cross-rank steps.
    turns: bool,
    mut pick: impl FnMut(usize) -> Vec<usize>,
) {
    if ring.parts == 0 || ring.slots == 0 {
        return;
    }
    let entry = |read: Slot| {
        weights
            .iter()
            .position(|w| w.is_some_and(|(slot, _)| same(slot, read)))
    };
    let mut order = Vec::new();
    for step in &plan.steps[plan.forward()] {
        for &read in &step.reads {
            if let Some(w) = entry(read).filter(|w| !order.contains(w)) {
                order.push(w);
            }
        }
    }
    let slot_of = |w: usize| weights[w].expect("an ordered weight has a slot");
    let (mut steps, mut ends) = (plan.steps.clone(), plan.forward().end);
    let mut last = vec![None::<usize>; ring.parts];
    let mut staged = 0;
    for w in order {
        let mut experts = pick(w);
        experts.truncate(ring.slots);
        if experts.is_empty() {
            continue;
        }
        let part = staged % ring.parts;
        staged += 1;
        let (slot, stride) = slot_of(w);
        let evicts = last[part];
        let bytes = experts.len() * stride;
        let id = batches.id(Batch {
            weight: w,
            experts,
            part,
            evicts,
        });
        last[part] = Some(id);
        let evicted = evicts.map(|e| slot_of(batches.list[e].weight).0);
        let batch = id as u32;
        let step = Stage { bytes, batch }.step(slot, evicted.into_iter().collect());
        let reads = |s: &Step| s.reads.iter().any(|r| same(*r, slot));
        let reader = steps.iter().position(reads).expect("read above");
        let at = earliest(&steps, &step, reader, turns);
        steps.insert(at, step);
        ends += 1;
    }
    // Each part's last weight returns to host memory.
    for evicts in last.into_iter().flatten() {
        let (weight, part) = (batches.list[evicts].weight, batches.list[evicts].part);
        let id = batches.id(Batch {
            weight,
            experts: Vec::new(),
            part,
            evicts: Some(evicts),
        });
        let batch = id as u32;
        let step = Stage { bytes: 0, batch }.step(slot_of(weight).0, Vec::new());
        steps.insert(ends, step);
        ends += 1;
    }
    // Callers keep their step numbers, so copies sit apart.
    plan.staged.clear();
    let mut before = 0;
    for step in steps {
        match step.kernel.staged() {
            Some(_) => plan.staged.push((before, step)),
            None => before += 1,
        }
    }
}
