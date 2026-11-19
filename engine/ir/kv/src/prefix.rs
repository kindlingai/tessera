//! Which retained run a prompt continues.

use crate::{Pool, RunId, Tag, begin};

struct Entry {
    run: RunId,
    history: Vec<u32>,
    valid: usize,
}

/// The tokens of retained runs, oldest first.
pub struct Prefixes {
    grid: usize,
    kept: Vec<Entry>,
}

impl Prefixes {
    /// Skips come in whole `grid` steps.
    pub fn new(grid: usize) -> Self {
        let kept = Vec::new();
        Self { grid, kept }
    }

    /// Records the tokens behind a retained run's `valid` rows.
    pub fn keep(&mut self, run: RunId, history: Vec<u32>, valid: usize) {
        let entry = Entry {
            run,
            history,
            valid,
        };
        self.kept.push(entry);
    }

    /// The newest run with the longest skip.
    pub fn find(
        &mut self,
        pool: &Pool,
        tag: Tag,
        prompt: &[u32],
        limit: usize,
    ) -> Option<(RunId, usize)> {
        self.kept.retain(|e| pool.rows(e.run).is_ok());
        let mut best = None;
        for e in &self.kept {
            if pool.tag(e.run) != Ok(tag) {
                continue;
            }
            let skip = begin(&e.history, e.valid, prompt, limit, self.grid).skip;
            if skip > 0 && best.is_none_or(|(_, most)| skip >= most) {
                best = Some((e.run, skip));
            }
        }
        best
    }
}
