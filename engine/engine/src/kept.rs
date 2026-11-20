//! Finished answers' rows, kept for prompts that continue them.

use crate::Error;
use sensai_kv::{Pool, Prefixes, RunId, Stats, Tag};

const TAG: Tag = Tag(0);

crate::metrics::registry!(KV = "kv");
crate::metrics::metric!(SIZE in KV = Gauge "vllm:kv_cache_size_tokens", "Rows the kept-row pool holds.");
crate::metrics::metric!(ACTIVE in KV = Gauge "vllm:kv_active_tokens", "Rows in pages live answers hold.");
crate::metrics::metric!(WORST in KV = Gauge "tessera:kv_worst_tokens", "Rows live answers may grow to.");
crate::metrics::metric!(LIVE in KV = Gauge "tessera:kv_live_runs", "Answers that hold kept rows.");
crate::metrics::metric!(RETAINED in KV = Gauge "tessera:kv_kept_runs", "Finished answers whose rows stay.");
crate::metrics::metric!(HITS in KV = Counter "tessera:kv_hits_total", "Prompts that started on kept rows.");
crate::metrics::metric!(MISSES in KV = Counter "tessera:kv_misses_total", "Prompts that started with no rows.");

/// The kept rows' pool, and how prompts met it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub pool: Stats,
    /// Admitted prompts that started on kept rows.
    pub hits: u64,
    /// Admitted prompts that started with no rows.
    pub misses: u64,
}

/// Dense: each slot is one page of `context` rows.
pub(crate) struct Kept {
    pool: Pool,
    index: Prefixes,
    context: usize,
    /// Each slot's live run.
    runs: Vec<Option<RunId>>,
    /// Paged: the free slots.
    free: Vec<u32>,
    /// Paged: the pages each slot's lane names.
    laned: Option<Vec<usize>>,
    hits: u64,
    misses: u64,
}

impl Kept {
    pub fn new(slots: usize, context: usize) -> Self {
        Self {
            pool: Pool::new(slots, context),
            index: Prefixes::new(0),
            context,
            runs: vec![None; slots],
            free: Vec::new(),
            laned: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Slots share whole pages of `page` rows.
    pub fn paged(slots: usize, context: usize, pages: usize, page: usize) -> Self {
        Self {
            pool: Pool::new(pages, page),
            index: Prefixes::new(page),
            free: (0..slots as u32).rev().collect(),
            laned: Some(vec![0; slots]),
            ..Self::new(slots, context)
        }
    }

    /// A slot for `prompt`, and the rows it already holds.
    pub fn admit(&mut self, prompt: &[u32]) -> Option<(u32, usize)> {
        // The last prompt row is always fed.
        let limit = prompt.len() - 1;
        let hit = self.index.find(&self.pool, TAG, prompt, limit);
        let worst = self.context;
        let Some(laned) = &mut self.laned else {
            let adopted = hit.and_then(|(run, skip)| {
                let run = self.pool.adopt(run, skip, worst).ok()?;
                Some((run, skip))
            });
            let (run, skip) = match adopted {
                Some(found) => found,
                None => (self.pool.reserve(TAG, 1, worst).ok()?, 0),
            };
            let slot = self.pool.pages(run).ok()?[0];
            self.runs[slot as usize] = Some(run);
            self.count(skip);
            return Some((slot, skip));
        };
        let slot = *self.free.last()?;
        let forked = hit.and_then(|(run, skip)| {
            let run = self.pool.fork(run, skip, worst).ok()?;
            Some((run, skip))
        });
        let (run, skip) = match forked {
            Some(found) => found,
            None => (self.pool.reserve(TAG, 0, worst).ok()?, 0),
        };
        self.free.pop();
        self.runs[slot as usize] = Some(run);
        laned[slot as usize] = 0;
        self.count(skip);
        Some((slot, skip))
    }

    fn count(&mut self, skip: usize) {
        match skip {
            0 => (self.misses += 1, MISSES.add(1)),
            _ => (self.hits += 1, HITS.add(1)),
        };
    }

    /// Sets the pool's gauges.
    pub fn publish(&self) {
        let stats = self.pool.stats();
        SIZE.set(stats.size as u64);
        ACTIVE.set(stats.active as u64);
        WORST.set(stats.worst as u64);
        LIVE.set(stats.live_runs as u64);
        RETAINED.set(stats.kept_runs as u64);
    }

    pub fn totals(&self) -> Totals {
        Totals {
            pool: self.pool.stats(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Paged: the slot's pages, if its lane lacks some.
    pub fn grow(&mut self, slot: u32, rows: usize) -> Result<Option<&[u32]>, Error> {
        let Some(lanes) = &mut self.laned else {
            return Ok(None);
        };
        let run = self.runs[slot as usize].ok_or(Error::RowPlace)?;
        // Runs share only whole pages, so no row moves.
        if !self.pool.grow(run, rows)?.is_empty() {
            return Err(Error::SharedPage);
        }
        let pages = self.pool.pages(run)?;
        let laned = &mut lanes[slot as usize];
        let more = pages.len() != *laned;
        *laned = pages.len();
        Ok(more.then_some(pages))
    }

    /// Keeps a slot's rows under the tokens they hold.
    pub fn finish(&mut self, slot: u32, history: Vec<u32>) {
        let Some(run) = self.runs[slot as usize].take() else {
            return;
        };
        if self.laned.is_some() {
            self.free.push(slot);
        }
        let rows = history.len();
        let grown = rows > 0 && self.pool.grow(run, rows).is_ok();
        if grown && self.pool.retain(run).is_ok() {
            self.index.keep(run, history, rows);
        } else {
            let _ = self.pool.release(run);
        }
    }
}
