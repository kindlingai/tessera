//! When a batch runs prefill between decode rounds.

/// A batch's work at one point in a round.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Work {
    /// Requests with prompt rows left.
    pub prefilling: usize,
    /// Requests that decode.
    pub active: usize,
    /// The fewest prompt rows a request lacks.
    pub least: usize,
}

/// Chooses the rounds that run prefill. Ranks must agree.
pub(crate) trait Interleave: Send {
    /// Whether this round runs a prefill step.
    fn prefill(&mut self, work: Work) -> bool;
    /// A prefill step fed `rows` and left `work`.
    fn fed(&mut self, rows: usize, work: Work);
}

/// Decode rounds follow each large prefill chunk.
pub(crate) struct Cooldown {
    /// A prompt this near its end skips the wait.
    pub quick: usize,
    /// Prefill rows that earn one decode round.
    pub rows: usize,
    /// Most decode rounds after one chunk.
    pub most: usize,
    wait: usize,
}

impl Default for Cooldown {
    fn default() -> Self {
        Self {
            quick: 512,
            rows: 1024,
            most: 4,
            wait: 0,
        }
    }
}

impl Interleave for Cooldown {
    fn prefill(&mut self, work: Work) -> bool {
        if work.prefilling == 0 {
            self.wait = 0;
            return false;
        }
        if work.active == 0 || self.wait == 0 || work.least <= self.quick {
            return true;
        }
        self.wait -= 1;
        false
    }

    fn fed(&mut self, rows: usize, work: Work) {
        self.wait = match work.prefilling > 0 && work.active > 0 {
            true => (rows / self.rows).min(self.most),
            false => 0,
        };
    }
}
