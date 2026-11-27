//! Finished answers' rows, kept for follow-up prompts to reuse.

use crate::Error;
use tessera_kv::{EVICT_COLD, EVICT_PINNED, Oldest, Pool, Prefixes, RunId, Spot, Stats, Tag};

const TAG: Tag = Tag(0);

tessera_engine_settings::declare_registry!(KV = "kv");
tessera_engine_settings::declare_metric!(SIZE in KV = Gauge "vllm:kv_cache_size_tokens", "Rows in the kept-row pool.");
tessera_engine_settings::declare_metric!(ACTIVE in KV = Gauge "vllm:kv_active_tokens", "Rows in pages live answers hold.");
tessera_engine_settings::declare_metric!(WORST in KV = Gauge "tessera:kv_worst_tokens", "Rows live answers may grow to.");
tessera_engine_settings::declare_metric!(LIVE in KV = Gauge "tessera:kv_live_runs", "Answers that hold kept rows.");
tessera_engine_settings::declare_metric!(RETAINED in KV = Gauge "tessera:kv_kept_runs", "Finished answers whose rows stay.");
tessera_engine_settings::declare_metric!(HITS in KV = Counter "tessera:kv_hits_total", "Prompts that started on kept rows.");
tessera_engine_settings::declare_metric!(MISSES in KV = Counter "tessera:kv_misses_total", "Prompts that started with no rows.");

/// The kept rows' pool statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub pool: Stats,
    /// Admitted prompts that started on kept rows.
    pub hits: u64,
    /// Admitted prompts that started with no rows.
    pub misses: u64,
}

/// Dense: each slot is one page of `context` rows.
pub struct Kept {
    pool: Pool,
    index: Prefixes,
    context: usize,
    /// Each slot's live run.
    runs: Vec<Option<RunId>>,
    /// Paged: the free slots.
    free: Vec<u32>,
    /// Paged: the pages each slot's lane names.
    laned: Option<Vec<usize>>,
    /// Paged: slots share fewer physical pages than full capacity.
    shared: bool,
    hits: u64,
    misses: u64,
}

impl Kept {
    pub fn new(slots: usize, context: usize) -> Self {
        Self {
            pool: Pool::new(slots, context, Box::new(Oldest::default())),
            index: Prefixes::new(0),
            context,
            runs: vec![None; slots],
            free: Vec::new(),
            laned: None,
            shared: false,
            hits: 0,
            misses: 0,
        }
    }

    /// Slots share whole pages of `page` rows.
    pub fn paged(slots: usize, context: usize, pages: usize, page: usize) -> Self {
        Self {
            pool: Pool::new(pages, page, Box::new(Oldest::default())),
            index: Prefixes::new(page),
            free: (0..slots as u32).rev().collect(),
            laned: Some(vec![0; slots]),
            shared: pages < slots * context.div_ceil(page),
            ..Self::new(slots, context)
        }
    }

    /// A slot for `prompt`, and at most `limit` kept rows.
    pub fn admit(&mut self, prompt: &[u32], limit: usize) -> Option<(u32, usize)> {
        self.admit_drop(prompt, limit, 0)
    }

    /// As `admit`, dropping trailing rows for shifted MTP KV.
    pub fn admit_shifted(
        &mut self,
        prompt: &[u32],
        limit: usize,
        page: usize,
    ) -> Option<(u32, usize)> {
        let drop = if self.laned.is_some() { page } else { 1 };
        self.admit_drop(prompt, limit, drop)
    }

    fn admit_drop(&mut self, prompt: &[u32], limit: usize, drop: usize) -> Option<(u32, usize)> {
        let hit = self.index.find(&self.pool, TAG, prompt, limit);
        let worst = self.context;
        let Some(laned) = &mut self.laned else {
            let adopted = hit.and_then(|(run, found)| {
                let skip = found.saturating_sub(drop);
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
        let need = match self.shared {
            true => prompt.len().max(1).min(worst),
            false => worst,
        };
        let forked = hit.and_then(|(run, found)| {
            let skip = found.saturating_sub(drop);
            (skip > 0).then(|| self.pool.fork(run, skip, need).ok().map(|r| (r, skip)))?
        });
        let (run, skip) = match forked {
            Some(found) => found,
            None => (self.pool.reserve(TAG, 0, need).ok()?, 0),
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

    /// Paged: grows each slot in `rows` once.
    #[cfg(test)]
    pub fn grow_rows(
        &mut self,
        rows: &[crate::Row],
        lanes: &mut crate::lanes::Lanes,
    ) -> Result<(), Error> {
        self.grow_rows_tiered(rows, lanes, &mut |_, _| Ok(false))
    }

    /// Paged: grows each slot in `rows` and calls `on_set`.
    pub fn grow_rows_with(
        &mut self,
        rows: &[crate::Row],
        mut on_set: impl FnMut(u32, &[u32]),
    ) -> Result<(), Error> {
        for (slot, _) in self.grow_peaks(rows)? {
            if let Some(pages) = self.pages(slot) {
                on_set(slot, pages);
            }
        }
        Ok(())
    }

    /// Grows each slot once, returning old lane lengths.
    fn grow_peaks(&mut self, rows: &[crate::Row]) -> Result<Vec<(u32, usize)>, Error> {
        let mut peaks: Vec<(u32, usize)> = Vec::new();
        for r in rows {
            let need = r.position as usize + 1;
            if let Some((_, best)) = peaks.last_mut().filter(|(s, _)| *s == r.slot) {
                *best = (*best).max(need);
            } else if let Some((_, best)) = peaks.iter_mut().find(|(s, _)| *s == r.slot) {
                *best = (*best).max(need);
            } else {
                peaks.push((r.slot, need));
            }
        }
        let mut grown_slots = Vec::new();
        for (slot, need) in peaks {
            let prev = self.laned.as_ref().map_or(0, |l| l[slot as usize]);
            if self.grow(slot, need)?.is_some() {
                grown_slots.push((slot, prev));
            }
        }
        Ok(grown_slots)
    }

    /// Paged: grows slots for `rows`, syncing `TIER_HOST` KV pages.
    pub fn grow_rows_tiered(
        &mut self,
        rows: &[crate::Row],
        lanes: &mut crate::lanes::Lanes,
        move_pages: &mut impl FnMut(bool, &[(u32, u32)]) -> Result<bool, Error>,
    ) -> Result<(), Error> {
        if !lanes.tiered() {
            return self.grow_rows_with(rows, |slot, pages| lanes.set(slot, pages));
        }
        let grown_slots = self.grow_peaks(rows)?;
        let freed = self.pool.take_freed();
        lanes.release_freed(&freed);
        let page = lanes.page;
        let mut alloc_pages = Vec::new();
        for &(slot, prev) in &grown_slots {
            if let Some(pages) = self.pages(slot) {
                alloc_pages.extend_from_slice(&pages[prev.min(pages.len())..]);
            }
        }
        let mut write_pages = Vec::new();
        let mut priority = self.page_priorities(lanes);
        for r in rows {
            if let Some(pages) = self.pages(r.slot)
                && let Some(&p) = pages.get(r.position as usize / page)
            {
                if !write_pages.contains(&p) {
                    write_pages.push(p);
                }
                if let Some(pr) = priority.get_mut(p as usize) {
                    *pr = EVICT_PINNED;
                }
            }
        }
        let moved = lanes.ensure_vram(&alloc_pages, &write_pages, &priority, move_pages)?;
        if moved {
            for slot in 0..self.runs.len() {
                if let Some(pages) = self.pages(slot as u32) {
                    lanes.set(slot as u32, pages);
                }
            }
        } else {
            for (slot, _) in grown_slots {
                if let Some(pages) = self.pages(slot) {
                    lanes.set(slot, pages);
                }
            }
        }
        Ok(())
    }

    /// Classifies each pool page for `TIER_HOST` eviction priority.
    fn page_priorities(&self, lanes: &crate::lanes::Lanes) -> Vec<u8> {
        let mut priority = vec![EVICT_COLD; lanes.pages()];
        let live = self.runs.iter().filter(|r| r.is_some()).count();
        for slot in 0..self.runs.len() {
            let Some(pages) = self.pages(slot as u32) else {
                continue;
            };
            let len = pages.len();
            for (at, &p) in pages.iter().enumerate() {
                let holders = self.pool.live_holders(p);
                let cls = lanes.class(Spot { at, len, holders }, live);
                if let Some(pr) = priority.get_mut(p as usize) {
                    *pr = (*pr).max(cls);
                }
            }
        }
        priority
    }

    /// Paged: the slot's pages, if its lane lacks some.
    pub fn grow(&mut self, slot: u32, rows: usize) -> Result<Option<&[u32]>, Error> {
        let Some(lanes) = &mut self.laned else {
            return Ok(None);
        };
        let run = self.runs[slot as usize].ok_or(Error::RowPlace)?;
        if self.shared {
            self.pool.widen(run, rows)?;
        }
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

    /// Physical pages currently assigned to `slot`, if paged.
    pub fn pages(&self, slot: u32) -> Option<&[u32]> {
        self.laned.as_ref()?;
        let run = self.runs[slot as usize]?;
        self.pool.pages(run).ok()
    }

    /// Keeps a slot's rows under their tokens.
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
