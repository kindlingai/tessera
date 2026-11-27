//! Whether a batch takes one more request.

use tessera_kv::Stats;

/// The pool's room and the next request's need.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ask {
    /// The batch has other requests.
    pub busy: bool,
    /// Rows outside the pages of live answers.
    pub free: usize,
    /// Rows the prompt adds.
    pub prompt: usize,
}

impl Ask {
    pub(crate) fn of(pool: Stats, busy: bool, prompt: usize) -> Self {
        Self {
            busy,
            free: pool.size.saturating_sub(pool.active),
            prompt,
        }
    }
}

/// Chooses on rank 0. Other ranks replay the result.
pub(crate) trait Admit: Send {
    fn admit(&mut self, ask: Ask) -> bool;
}

/// A busy batch keeps spare rows for its answers.
pub(crate) struct Headroom {
    pub rows: usize,
}

impl Default for Headroom {
    fn default() -> Self {
        Self { rows: 2048 }
    }
}

impl Admit for Headroom {
    fn admit(&mut self, ask: Ask) -> bool {
        !ask.busy || ask.free >= ask.prompt + self.rows
    }
}
