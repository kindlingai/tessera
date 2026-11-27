//! How an idle loop waits for work.

use std::time::{Duration, Instant};

/// The next action of an idle loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    Spin,
    Poll,
    /// Poll, then sleep this long.
    Nap(Duration),
}

/// Paces a loop that looks for work.
pub(crate) trait Pace {
    /// The loop did work.
    fn worked(&mut self);
    /// The loop found none. `pending`: more arrives soon.
    fn idle(&mut self, pending: bool) -> Step;
}

/// Spins, polls at intervals, and naps after a quiet time.
pub(crate) struct Spinning {
    /// Idle turns between polls.
    pub every: u32,
    /// Quiet time before the first nap.
    pub awake: Duration,
    pub nap: Duration,
    spins: u32,
    busy: Instant,
}

impl Default for Spinning {
    fn default() -> Self {
        Self {
            every: 1024,
            awake: Duration::from_millis(200),
            nap: Duration::from_micros(200),
            spins: 0,
            busy: Instant::now(),
        }
    }
}

impl Pace for Spinning {
    fn worked(&mut self) {
        // Reads the clock only after a poll.
        if self.spins >= self.every {
            self.busy = Instant::now();
        }
        self.spins = 0;
    }

    fn idle(&mut self, pending: bool) -> Step {
        self.spins = self.spins.wrapping_add(1);
        if !self.spins.is_multiple_of(self.every) {
            return Step::Spin;
        }
        if pending {
            self.busy = Instant::now();
        } else if self.busy.elapsed() > self.awake {
            return Step::Nap(self.nap);
        }
        Step::Poll
    }
}
