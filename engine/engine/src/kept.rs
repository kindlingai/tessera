//! Finished answers' rows, kept for prompts that continue them.

use sensai_kv::{Pool, Prefixes, RunId, Tag};

const TAG: Tag = Tag(0);

/// Each slot is one page of `context` rows.
pub(crate) struct Kept {
    pool: Pool,
    index: Prefixes,
    context: usize,
    /// Each slot's live run.
    runs: Vec<Option<RunId>>,
}

impl Kept {
    pub fn new(slots: usize, context: usize) -> Self {
        Self {
            pool: Pool::new(slots, context),
            index: Prefixes::new(0),
            context,
            runs: vec![None; slots],
        }
    }

    /// A slot for `prompt`, and the rows it already holds.
    pub fn admit(&mut self, prompt: &[u32]) -> Option<(u32, usize)> {
        // The last prompt row is always fed.
        let limit = prompt.len() - 1;
        let hit = self.index.find(&self.pool, TAG, prompt, limit);
        let adopted = hit.and_then(|(run, skip)| {
            let run = self.pool.adopt(run, skip, self.context).ok()?;
            Some((run, skip))
        });
        let (run, skip) = match adopted {
            Some(found) => found,
            None => (self.pool.reserve(TAG, 1, self.context).ok()?, 0),
        };
        let slot = self.pool.pages(run).ok()?[0];
        self.runs[slot as usize] = Some(run);
        Some((slot, skip))
    }

    /// Keeps a slot's rows under the tokens they hold.
    pub fn finish(&mut self, slot: u32, history: Vec<u32>) {
        let Some(run) = self.runs[slot as usize].take() else {
            return;
        };
        let rows = history.len();
        let grown = rows > 0 && self.pool.grow(run, rows).is_ok();
        if grown && self.pool.retain(run).is_ok() {
            self.index.keep(run, history, rows);
        } else {
            let _ = self.pool.release(run);
        }
    }
}
