//! Which steps must keep their order, on any lane.

use crate::kernels::{Stage, crosses};
use crate::plan::{Slot, Step};

/// The lane cross-rank steps run on beside compute.
pub(crate) const COMM_LANE: u8 = 1;
/// The lane host copies run on beside compute.
pub const COPY_LANE: u8 = 2;

fn overlaps(a: Slot, b: Slot) -> bool {
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

/// Whether `later` must wait for `earlier`.
pub fn hazards(earlier: &Step, later: &Step) -> bool {
    if crosses(earlier.kernel.name()) && crosses(later.kernel.name()) {
        return true;
    }
    if earlier.kernel.name() == Stage::NAME {
        return stages(earlier, later);
    }
    if later.kernel.name() == Stage::NAME {
        return stages(later, earlier);
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
pub fn earliest(steps: &[Step], step: &Step, at: usize) -> usize {
    let mut place = at;
    // Steps on one lane keep their order.
    while place > 0 && steps[place - 1].lane != step.lane && !hazards(&steps[place - 1], step) {
        place -= 1;
    }
    place
}
