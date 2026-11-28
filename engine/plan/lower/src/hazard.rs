//! Which steps must keep their order, on any lane.

use crate::kernels::{Stage, crosses};
use crate::plan::{Plan, Slot, Step};

/// The lane cross-rank steps run on beside compute.
pub(crate) const COMM_LANE: u8 = 1;
/// The lane host copies run on beside compute.
pub const COPY_LANE: u8 = 2;

/// True when `a` and `b` overlap in one buffer.
pub fn overlaps(a: Slot, b: Slot) -> bool {
    if a.buffer != b.buffer {
        return false;
    }
    let a_end = a.offset + a.bytes().max(1);
    let b_end = b.offset + b.bytes().max(1);
    a.offset < b_end && b.offset < a_end
}

fn writes(step: &Step) -> impl Iterator<Item = Slot> + '_ {
    std::iter::once(step.write).chain(step.also.iter().copied())
}

/// Whether `copy` names a weight `other` touches.
fn stages(copy: &Step, other: &Step) -> bool {
    let same = |a: Slot, b: Slot| a.buffer == b.buffer && a.offset == b.offset;
    let touched = || other.reads.iter().copied().chain(writes(other));
    writes(copy).any(|w| touched().any(|t| same(w, t)))
}

/// Whether `later` waits for `earlier`. `turns`: copies wait for sums.
pub fn hazards(earlier: &Step, later: &Step, turns: bool) -> bool {
    if crosses(earlier.kernel.name()) && crosses(later.kernel.name()) {
        return true;
    }
    if earlier.kernel.name() == Stage::NAME {
        return (turns && crosses(later.kernel.name())) || stages(earlier, later);
    }
    if later.kernel.name() == Stage::NAME {
        return (turns && crosses(earlier.kernel.name())) || stages(later, earlier);
    }
    let raw = writes(earlier).any(|w| later.reads.iter().any(|&r| overlaps(w, r)));
    let war = earlier
        .reads
        .iter()
        .any(|&r| writes(later).any(|w| overlaps(r, w)));
    let waw = writes(earlier).any(|w1| writes(later).any(|w2| overlaps(w1, w2)));
    raw || war || waw
}

/// The earliest place up to `at` for `step`.
pub fn earliest(steps: &[Step], step: &Step, at: usize, turns: bool) -> usize {
    let mut place = at;
    // Steps on one lane keep their order.
    while place > 0
        && steps[place - 1].lane != step.lane
        && !hazards(&steps[place - 1], step, turns)
    {
        place -= 1;
    }
    place
}

/// Tracks concurrent compute waves and when a barrier is required.
pub struct Waves {
    active: bool,
    kept: Vec<bool>,
    start: usize,
}

impl Waves {
    pub fn new(plan: &Plan, active: bool) -> Self {
        let mut kept = vec![false; if active { plan.buffers.len() } else { 0 }];
        for (_, slot) in &plan.kept {
            if let Some(flag) = kept.get_mut(slot.buffer as usize) {
                *flag = true;
            }
        }
        Self {
            active,
            kept,
            start: 0,
        }
    }

    pub fn active(&self) -> bool {
        self.active
    }

    /// True when `steps[at]` needs a barrier before running.
    pub fn before(&mut self, steps: &[&Step], at: usize, special: bool) -> bool {
        if !self.active || at <= self.start {
            return false;
        }
        let step = steps[at];
        let hit = special
            || (self.start..at).any(|j| {
                let prev = steps[j];
                hazards(prev, step, false)
                    || prev.reads.iter().any(|&a| {
                        self.kept.get(a.buffer as usize).copied().unwrap_or(false)
                            && step.reads.iter().any(|&b| overlaps(a, b))
                    })
            });
        if hit {
            self.start = at;
        }
        hit
    }

    /// Starts a new wave at `next` when active.
    pub fn reset(&mut self, next: usize) -> bool {
        if self.active {
            self.start = next;
        }
        self.active
    }
}
