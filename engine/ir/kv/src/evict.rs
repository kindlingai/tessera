//! Which retained run a full pool ends.

/// Chooses the retained run to end, by slot.
pub trait Evict: Send {
    /// The run at `at` became retained.
    fn retained(&mut self, at: usize);
    /// A new run shares the rows of `at`.
    fn used(&mut self, at: usize);
    /// The run at `at` is no longer retained.
    fn gone(&mut self, at: usize);
    /// The retained run to end.
    fn victim(&mut self) -> Option<usize>;
}

/// Ends the run retained first.
#[derive(Default)]
pub struct Oldest {
    /// Each slot's retirement number. Zero: not retained.
    stamps: Vec<u64>,
    count: u64,
}

impl Evict for Oldest {
    fn retained(&mut self, at: usize) {
        if self.stamps.len() <= at {
            self.stamps.resize(at + 1, 0);
        }
        self.count += 1;
        self.stamps[at] = self.count;
    }

    fn used(&mut self, _at: usize) {}

    fn gone(&mut self, at: usize) {
        if let Some(stamp) = self.stamps.get_mut(at) {
            *stamp = 0;
        }
    }

    fn victim(&mut self) -> Option<usize> {
        let held = self.stamps.iter().enumerate().filter(|(_, s)| **s > 0);
        held.min_by_key(|(_, s)| **s).map(|(at, _)| at)
    }
}
