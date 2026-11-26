//! A pool of pages that runs of kept rows share.

use std::collections::BTreeSet;
use thiserror::Error;

/// A run's place in the pool. Stale ids fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunId {
    at: u32,
    generation: u32,
}

/// Runs share pages only with runs of the same tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tag(pub u64);

/// Rows to copy between pages before the launch writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCopy {
    pub from: u32,
    pub to: u32,
    pub rows: usize,
}

/// Every error leaves the pool as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum Error {
    #[error("{0} more pages do not fit beside every worst case")]
    NoRoom(usize),
    #[error("{rows} rows pass the run's worst case of {worst}")]
    PastWorst { rows: usize, worst: usize },
    #[error("a run takes at most its source's {0} rows")]
    PastSource(usize),
    #[error("{0:?} is live")]
    Live(RunId),
    #[error("{0:?} is retained")]
    Retained(RunId),
    #[error("{0:?} is not a run")]
    Unknown(RunId),
    #[error("pages do not fit the VRAM and host tiers")]
    TierFull,
}

struct Run {
    tag: Tag,
    pages: Vec<u32>,
    rows: usize,
    worst: usize,
    /// Retirement number, if retained. Lowest goes first.
    retired: Option<u64>,
}

#[derive(Default)]
struct Slot {
    generation: u32,
    run: Option<Run>,
}

/// A pool's totals. Row counts are whole pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// The pool's row count.
    pub size: usize,
    /// Rows in pages of live runs.
    pub active: usize,
    /// Rows live runs may grow to.
    pub worst: usize,
    pub live_runs: usize,
    pub kept_runs: usize,
}

/// Pages and their holders. Equal calls give equal pools.
pub struct Pool {
    page: usize,
    /// Each page's holders: live runs, then retained runs.
    live: Vec<u32>,
    kept: Vec<u32>,
    /// Pages held by at least one live run.
    active: usize,
    /// Pages held only by retained runs.
    reclaimable: usize,
    free: BTreeSet<u32>,
    freed: Vec<u32>,
    slots: Vec<Slot>,
    retirements: u64,
}

impl Pool {
    /// `pages` pages of `page` rows each.
    pub fn new(pages: usize, page: usize) -> Self {
        Self {
            page,
            live: vec![0; pages],
            kept: vec![0; pages],
            active: 0,
            reclaimable: 0,
            free: (0..pages as u32).collect(),
            freed: Vec::new(),
            slots: Vec::new(),
            retirements: 0,
        }
    }

    /// Drains physical pages freed since the previous call.
    pub fn take_freed(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.freed)
    }

    /// Live runs currently holding `page`.
    pub fn live_holders(&self, page: u32) -> u32 {
        self.live.get(page as usize).copied().unwrap_or(0)
    }

    pub fn pages(&self, run: RunId) -> Result<&[u32], Error> {
        Ok(&self.run(run)?.pages)
    }

    pub fn rows(&self, run: RunId) -> Result<usize, Error> {
        Ok(self.run(run)?.rows)
    }

    pub fn tag(&self, run: RunId) -> Result<Tag, Error> {
        Ok(self.run(run)?.tag)
    }

    /// Free pages, and pages that only retained runs hold.
    pub fn spare(&self) -> usize {
        self.free.len() + self.reclaimable
    }

    /// Pages live runs may take before their worst case.
    pub fn promised(&self) -> usize {
        let live = self.slots.iter().filter_map(|s| s.run.as_ref());
        live.filter(|r| r.retired.is_none())
            .map(|r| self.need(r))
            .sum()
    }

    pub fn stats(&self) -> Stats {
        let mut stats = Stats {
            size: self.live.len() * self.page,
            active: self.active * self.page,
            ..Stats::default()
        };
        for run in self.slots.iter().filter_map(|s| s.run.as_ref()) {
            if run.retired.is_some() {
                stats.kept_runs += 1;
                continue;
            }
            stats.live_runs += 1;
            stats.worst += self.span(run.worst) * self.page;
        }
        stats
    }

    /// A live run of `rows` that may grow to `worst`.
    pub fn reserve(&mut self, tag: Tag, rows: usize, worst: usize) -> Result<RunId, Error> {
        if rows > worst {
            return Err(Error::PastWorst { rows, worst });
        }
        self.fits(self.span(worst), 0)?;
        let pages = (0..self.span(rows)).map(|_| self.take()).collect();
        let run = Run {
            tag,
            pages,
            rows,
            worst,
            retired: None,
        };
        Ok(self.add(run))
    }

    /// A live run that shares the first `rows` of `source`.
    pub fn fork(&mut self, source: RunId, rows: usize, worst: usize) -> Result<RunId, Error> {
        let src = self.run(source)?;
        if rows > src.rows {
            return Err(Error::PastSource(src.rows));
        }
        if rows > worst {
            return Err(Error::PastWorst { rows, worst });
        }
        let shared = &src.pages[..self.span(rows)];
        let lost = shared
            .iter()
            .filter(|p| self.live[**p as usize] == 0)
            .count();
        // A shared partial page costs each live holder one copy.
        let mine = !rows.is_multiple_of(self.page);
        let theirs = src.retired.is_none()
            && !src.rows.is_multiple_of(self.page)
            && shared.len() == src.pages.len()
            && shared.last().is_some_and(|p| self.holders(*p) == 1);
        let need = self.span(worst) - shared.len() + mine as usize + theirs as usize;
        self.fits(need, lost)?;
        let (tag, pages) = (src.tag, shared.to_vec());
        for &p in &pages {
            self.hold_live(p);
        }
        let run = Run {
            tag,
            pages,
            rows,
            worst,
            retired: None,
        };
        Ok(self.add(run))
    }

    /// Raises a live run's worst-case row limit when room allows.
    pub fn widen(&mut self, run: RunId, worst: usize) -> Result<(), Error> {
        let at = self.live_at(run)?;
        let cur = self.run_at(at).worst;
        if worst <= cur {
            return Ok(());
        }
        let extra = self.span(worst).saturating_sub(self.span(cur));
        if extra > 0 {
            self.fits(extra, 0)?;
        }
        self.run_mut(at).worst = worst;
        debug_assert!(self.consistent());
        Ok(())
    }

    /// Grows a live run. Make the copies before the launch.
    pub fn grow(&mut self, run: RunId, rows: usize) -> Result<Vec<PageCopy>, Error> {
        let at = self.live_at(run)?;
        let (held, worst) = (self.run_at(at).rows, self.run_at(at).worst);
        if rows > worst {
            return Err(Error::PastWorst { rows, worst });
        }
        let mut copies = Vec::new();
        if rows <= held {
            return Ok(copies);
        }
        let written = held % self.page;
        let last = self.run_mut(at).pages.last().copied();
        if let Some(from) = last.filter(|p| written > 0 && self.holders(*p) > 1) {
            let to = self.take();
            self.unhold(from, false);
            *self.run_mut(at).pages.last_mut().expect("a last page") = to;
            copies.push(PageCopy {
                from,
                to,
                rows: written,
            });
        }
        for _ in self.run_mut(at).pages.len()..self.span(rows) {
            let page = self.take();
            self.run_mut(at).pages.push(page);
        }
        self.run_mut(at).rows = rows;
        debug_assert!(self.consistent());
        Ok(copies)
    }

    /// Drops a live run's rows past `rows`.
    pub fn truncate(&mut self, run: RunId, rows: usize) -> Result<(), Error> {
        let at = self.live_at(run)?;
        let r = self.run_at(at);
        if rows >= r.rows {
            return Ok(());
        }
        let keep = self.span(rows);
        // The run may grow back. Shared pages stay held.
        let regrow = r.pages.len() - keep + self.copies(rows, &r.pages[..keep]);
        let freed = r.pages[keep..]
            .iter()
            .filter(|p| self.live[**p as usize] == 1);
        let owed = freed.count() + self.copies(r.rows, &r.pages);
        self.fits(regrow.saturating_sub(owed), 0)?;
        let dropped = self.run_mut(at).pages.split_off(keep);
        self.run_mut(at).rows = rows;
        for page in dropped {
            self.unhold(page, false);
        }
        debug_assert!(self.consistent());
        Ok(())
    }

    /// Makes a live run reclaimable. Forks may share its rows.
    pub fn retain(&mut self, run: RunId) -> Result<(), Error> {
        let at = self.live_at(run)?;
        self.retirements += 1;
        let retired = self.retirements;
        self.run_mut(at).retired = Some(retired);
        for i in 0..self.run_at(at).pages.len() {
            let p = self.run_at(at).pages[i];
            self.live_to_kept(p);
        }
        debug_assert!(self.consistent());
        Ok(())
    }

    /// Makes a retained run live again, with its first `rows`.
    pub fn adopt(&mut self, run: RunId, rows: usize, worst: usize) -> Result<RunId, Error> {
        let r = self.run(run)?;
        if r.retired.is_none() {
            return Err(Error::Live(run));
        }
        if rows > r.rows {
            return Err(Error::PastSource(r.rows));
        }
        if rows > worst {
            return Err(Error::PastWorst { rows, worst });
        }
        let keep = self.span(rows);
        let held = &r.pages[..keep];
        let lost = held.iter().filter(|p| self.live[**p as usize] == 0);
        let need = self.span(worst) - keep + self.copies(rows, held);
        self.fits(need, lost.count())?;
        let at = run.at as usize;
        for page in self.run_mut(at).pages.split_off(keep) {
            self.unhold(page, true);
        }
        let r = self.run_mut(at);
        (r.rows, r.worst, r.retired) = (rows, worst, None);
        for i in 0..self.run_at(at).pages.len() {
            let p = self.run_at(at).pages[i];
            self.kept_to_live(p);
        }
        // The retained id is now stale.
        self.slots[at].generation += 1;
        debug_assert!(self.consistent());
        Ok(RunId {
            at: run.at,
            generation: self.slots[at].generation,
        })
    }

    /// Ends a run, live or retained.
    pub fn release(&mut self, run: RunId) -> Result<(), Error> {
        self.run(run)?;
        self.end(run.at as usize);
        debug_assert!(self.consistent());
        Ok(())
    }

    /// A hash of each page's holders and each run.
    pub fn digest(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        let mut eat = |x: u64| {
            for b in x.to_le_bytes() {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
            }
        };
        for (l, k) in self.live.iter().zip(&self.kept) {
            eat((u64::from(*l) << 32) | u64::from(*k));
        }
        for slot in &self.slots {
            eat(u64::from(slot.generation));
            let Some(r) = &slot.run else { continue };
            for x in [r.tag.0, r.rows as u64, r.worst as u64] {
                eat(x);
            }
            eat(r.retired.unwrap_or(0));
            r.pages.iter().for_each(|p| eat(u64::from(*p)));
        }
        hash
    }

    fn span(&self, rows: usize) -> usize {
        rows.div_ceil(self.page)
    }

    fn holders(&self, page: u32) -> u32 {
        self.live[page as usize] + self.kept[page as usize]
    }

    /// Pages to reach the worst case, copies included.
    fn need(&self, r: &Run) -> usize {
        self.span(r.worst) - r.pages.len() + self.copies(r.rows, &r.pages)
    }

    /// One if the next write lands in a shared page.
    fn copies(&self, rows: usize, pages: &[u32]) -> usize {
        let partial = !rows.is_multiple_of(self.page);
        let shared = pages.last().is_some_and(|p| self.holders(*p) > 1);
        (partial && shared) as usize
    }

    /// Refuses unless every live run could reach its worst case.
    fn fits(&self, need: usize, lost: usize) -> Result<(), Error> {
        match self.promised() + need + lost <= self.spare() {
            true => Ok(()),
            false => Err(Error::NoRoom(need)),
        }
    }

    /// The lowest free page, after evicting retained runs if needed.
    fn take(&mut self) -> u32 {
        let page = loop {
            if let Some(page) = self.free.pop_first() {
                break page;
            }
            let retained = self.slots.iter().enumerate().filter_map(|(at, s)| {
                let retired = s.run.as_ref()?.retired?;
                Some((retired, at))
            });
            let (_, at) = retained.min().expect("the room check promised a page");
            self.end(at);
        };
        self.live[page as usize] = 1;
        self.active += 1;
        page
    }

    fn end(&mut self, at: usize) {
        let slot = &mut self.slots[at];
        slot.generation += 1;
        let r = slot.run.take().expect("a run to end");
        for page in r.pages {
            self.unhold(page, r.retired.is_some());
        }
    }

    fn hold_live(&mut self, page: u32) {
        let at = page as usize;
        if self.live[at] == 0 {
            self.active += 1;
            if self.kept[at] > 0 {
                self.reclaimable -= 1;
            }
        }
        self.live[at] += 1;
    }

    fn live_to_kept(&mut self, page: u32) {
        let at = page as usize;
        self.live[at] -= 1;
        self.kept[at] += 1;
        if self.live[at] == 0 {
            self.active -= 1;
            self.reclaimable += 1;
        }
    }

    fn kept_to_live(&mut self, page: u32) {
        let at = page as usize;
        let was_only_kept = self.live[at] == 0;
        self.kept[at] -= 1;
        self.live[at] += 1;
        if was_only_kept {
            self.active += 1;
            self.reclaimable -= 1;
        }
    }

    fn unhold(&mut self, page: u32, retained: bool) {
        let at = page as usize;
        if retained {
            self.kept[at] -= 1;
            if self.live[at] == 0 && self.kept[at] == 0 {
                self.reclaimable -= 1;
                self.free.insert(page);
                self.freed.push(page);
            }
            return;
        }
        self.live[at] -= 1;
        if self.live[at] == 0 {
            self.active -= 1;
            if self.kept[at] > 0 {
                self.reclaimable += 1;
            } else {
                self.free.insert(page);
                self.freed.push(page);
            }
        }
    }

    fn add(&mut self, run: Run) -> RunId {
        let at = match self.slots.iter().position(|s| s.run.is_none()) {
            Some(at) => at,
            None => {
                self.slots.push(Slot::default());
                self.slots.len() - 1
            }
        };
        self.slots[at].run = Some(run);
        debug_assert!(self.consistent());
        RunId {
            at: at as u32,
            generation: self.slots[at].generation,
        }
    }

    fn run(&self, id: RunId) -> Result<&Run, Error> {
        let slot = self.slots.get(id.at as usize);
        let slot = slot.filter(|s| s.generation == id.generation);
        slot.and_then(|s| s.run.as_ref()).ok_or(Error::Unknown(id))
    }

    fn live_at(&self, id: RunId) -> Result<usize, Error> {
        match self.run(id)?.retired {
            Some(_) => Err(Error::Retained(id)),
            None => Ok(id.at as usize),
        }
    }

    fn run_at(&self, at: usize) -> &Run {
        self.slots[at].run.as_ref().expect("a live slot")
    }

    fn run_mut(&mut self, at: usize) -> &mut Run {
        self.slots[at].run.as_mut().expect("a live slot")
    }

    /// Holder counts match the runs, and every promise fits.
    fn consistent(&self) -> bool {
        let (mut live, mut kept) = (vec![0; self.live.len()], vec![0; self.kept.len()]);
        for r in self.slots.iter().filter_map(|s| s.run.as_ref()) {
            let counts = if r.retired.is_some() {
                &mut kept
            } else {
                &mut live
            };
            r.pages.iter().for_each(|p| counts[*p as usize] += 1);
        }
        let free =
            (0..self.live.len() as u32).filter(|p| live[*p as usize] + kept[*p as usize] == 0);
        let active = live.iter().filter(|l| **l > 0).count();
        let reclaimable = live
            .iter()
            .zip(&kept)
            .filter(|&(l, k)| *l == 0 && *k > 0)
            .count();
        live == self.live
            && kept == self.kept
            && active == self.active
            && reclaimable == self.reclaimable
            && free.eq(self.free.iter().copied())
            && self.promised() <= self.spare()
    }
}
