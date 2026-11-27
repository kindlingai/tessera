//! Which drafter a dual selector uses next.

use super::DualSelector;

/// Chooses the drafter. Ranks name a choice by its index.
pub(super) trait Switch: Sync {
    /// Whether the next round drafts with MTP.
    fn want_mtp(&self, selector: &DualSelector) -> bool;
}

/// Every choice, in wire order.
pub(super) static SWITCHES: [&dyn Switch; 1] = [&Dwell];
pub(super) const DWELL: usize = 0;

/// Keeps a drafter for `dwell` rounds, then needs `margin`.
struct Dwell;

impl Switch for Dwell {
    fn want_mtp(&self, selector: &DualSelector) -> bool {
        if selector.since < selector.dwell {
            return selector.use_mtp;
        }
        let (mtp, dflash) = selector.rates();
        let gain = 1.0 + selector.margin;
        match selector.use_mtp {
            true => mtp * gain >= dflash,
            false => mtp > dflash * gain,
        }
    }
}
