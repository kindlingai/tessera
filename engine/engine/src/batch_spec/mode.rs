//! When a dual batch drafts one answer alone.

use super::BatchMtp;

/// A batch's answers in decode and in prefill.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flight {
    pub active: usize,
    pub prefilling: usize,
}

/// Chooses solo or batched drafting. Ranks must agree.
pub(crate) trait Mode: Send {
    /// Whether `admitted` requests start in solo mode.
    fn solo(&self, flight: Flight, admitted: usize) -> bool;
    /// Whether the answer in `flight` moves to solo mode.
    fn promote(&self, flight: Flight, left: usize) -> bool;
    /// Whether to keep DFlash taps for a later promotion.
    fn tail(&self, answers: usize) -> bool;
}

/// Solo mode for one answer with tokens left to write.
pub(crate) struct Lone {
    /// Fewest tokens left that earn a promotion.
    pub least: usize,
    /// Most answers that still keep DFlash taps.
    pub near: usize,
}

impl Default for Lone {
    fn default() -> Self {
        Self { least: 8, near: 2 }
    }
}

impl Mode for Lone {
    fn solo(&self, flight: Flight, admitted: usize) -> bool {
        flight.active + flight.prefilling == 0 && admitted == 1
    }

    fn promote(&self, flight: Flight, left: usize) -> bool {
        flight.active == 1 && flight.prefilling == 0 && left >= self.least
    }

    fn tail(&self, answers: usize) -> bool {
        answers <= self.near
    }
}

impl BatchMtp<'_> {
    pub(super) fn flight(&self) -> Flight {
        Flight {
            active: self.active.len(),
            prefilling: self.prefilling.len(),
        }
    }
}
