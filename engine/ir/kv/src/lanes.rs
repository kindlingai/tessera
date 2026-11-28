//! Host page tables and two-tier sparse KV mapping.

use crate::Error;
use crate::flight::{Flights, Move};
use crate::tiering::{EVICT_COLD, Load, Room, Spot, Tiering};
use std::ops::Range;

/// Page-table bit selecting pinned host RAM (`TIER_HOST = 1` in `kept.h`).
const TIER_HOST_BIT: u32 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    Vram(u32),
    Host(u32),
    /// Reads use this VRAM page while it copies out.
    Leaving(u32),
}

impl Loc {
    fn entry(self) -> u32 {
        match self {
            Self::Vram(p) | Self::Leaving(p) => p,
            Self::Host(p) => TIER_HOST_BIT | p,
        }
    }
}

/// Levels of the VRAM pages outside `busy` that may leave.
fn levels<'a>(
    loc: &'a [Option<Loc>],
    busy: [&'a [u32]; 2],
    priority: &'a [u8],
) -> impl Fn(u32) -> Option<u8> + 'a {
    move |p| match loc.get(p as usize).copied().flatten() {
        Some(Loc::Vram(_)) if !busy.iter().any(|b| b.contains(&p)) => {
            Some(priority.get(p as usize).copied().unwrap_or(EVICT_COLD))
        }
        _ => None,
    }
}

/// Maps logical pool pages to VRAM or host KV pages.
struct TierMap {
    loc: Vec<Option<Loc>>,
    vram_free: Vec<u32>,
    host_free: Vec<u32>,
    flights: Flights,
    policy: Box<dyn Tiering>,
    vram_cap: usize,
    host_cap: usize,
}

impl TierMap {
    fn new(
        index_pages: usize,
        vram_pages: usize,
        host_pages: usize,
        policy: Box<dyn Tiering>,
    ) -> Self {
        Self {
            loc: vec![None; index_pages],
            vram_free: (0..vram_pages as u32).rev().collect(),
            host_free: (0..host_pages as u32).rev().collect(),
            flights: Flights::default(),
            policy,
            vram_cap: vram_pages,
            host_cap: host_pages,
        }
    }

    fn reset(&mut self) {
        self.loc.fill(None);
        self.vram_free = (0..self.vram_cap as u32).rev().collect();
        self.host_free = (0..self.host_cap as u32).rev().collect();
        self.flights = Flights::default();
        self.policy.clear();
    }

    fn release(&mut self, freed: &[u32]) {
        for &p in freed {
            if let Some(loc) = self.loc.get_mut(p as usize).and_then(Option::take) {
                match loc {
                    Loc::Vram(v) => self.vram_free.push(v),
                    Loc::Host(h) => self.host_free.push(h),
                    Loc::Leaving(_) => self.flights.free(p),
                }
            }
        }
        self.policy.freed(freed);
    }

    fn at(&self, p: u32) -> Option<Loc> {
        self.loc.get(p as usize).copied().flatten()
    }

    fn entry(&self, p: u32) -> u32 {
        self.at(p).map_or(0, Loc::entry)
    }

    /// Moves each copied page to its host page.
    fn land<E: From<Error>>(
        &mut self,
        move_pages: &mut impl FnMut(Move, &[(u32, u32)]) -> Result<bool, E>,
    ) -> Result<bool, E> {
        if self.flights.is_empty() {
            return Ok(false);
        }
        move_pages(Move::Settle, &[])?;
        for leaving in self.flights.land() {
            self.vram_free.push(leaving.vram);
            match leaving.live {
                true => self.loc[leaving.page as usize] = Some(Loc::Host(leaving.host)),
                false => self.host_free.push(leaving.host),
            }
        }
        Ok(true)
    }

    /// The policy's victims among VRAM pages outside `busy`.
    fn victims(&mut self, short: usize, busy: [&[u32]; 2], priority: &[u8]) -> Vec<u32> {
        let level = levels(&self.loc, busy, priority);
        self.policy.victims(short, self.host_free.len(), &level)
    }

    /// Starts demotes of `picked` pages beside the next step.
    fn depart<E: From<Error>>(
        &mut self,
        picked: Vec<u32>,
        move_pages: &mut impl FnMut(Move, &[(u32, u32)]) -> Result<bool, E>,
    ) -> Result<(), E> {
        let mut left = Vec::with_capacity(picked.len());
        let mut pairs = Vec::with_capacity(picked.len());
        for p in picked {
            if let Some(Loc::Vram(v)) = self.at(p)
                && let Some(h) = self.host_free.pop()
            {
                left.push(p);
                pairs.push((v, h));
            }
        }
        if pairs.is_empty() {
            return Ok(());
        }
        if !move_pages(Move::Depart, &pairs)? {
            return Err(Error::TierFull.into());
        }
        for (&p, &(v, h)) in left.iter().zip(&pairs) {
            self.loc[p as usize] = Some(Loc::Leaving(v));
            self.flights.depart(p, v, h);
        }
        Ok(())
    }

    fn ensure_vram<E: From<Error>>(
        &mut self,
        alloc_pages: &[u32],
        write_pages: &[u32],
        priority: &[u8],
        move_pages: &mut impl FnMut(Move, &[(u32, u32)]) -> Result<bool, E>,
    ) -> Result<bool, E> {
        let mut moved = self.land(move_pages)?;
        let mut fresh = Vec::new();
        let mut to_promote = Vec::new();
        for &p in alloc_pages.iter().chain(write_pages) {
            if self.at(p).is_none() && !fresh.contains(&p) {
                fresh.push(p);
            }
        }
        for &p in write_pages {
            if let Some(Loc::Host(h)) = self.at(p)
                && !to_promote.iter().any(|&(q, _)| q == p)
            {
                to_promote.push((p, h));
            }
        }
        let keep = self.policy.reserve(fresh.len() + to_promote.len());
        let rank = |p: u32| priority.get(p as usize).copied().unwrap_or(EVICT_COLD);
        let mut host = Vec::new();
        for (idx, at) in self.loc.iter().enumerate() {
            let p = idx as u32;
            if matches!(at, Some(Loc::Host(_))) && !write_pages.contains(&p) {
                host.push((p, rank(p)));
            }
        }
        // Promoted pages leave the reserve free.
        let room = Room {
            need: fresh.len() + to_promote.len() + keep,
            vram_free: self.vram_free.len(),
            host_free: self.host_free.len(),
        };
        let promoted = {
            let level = levels(&self.loc, [write_pages, &[]], priority);
            self.policy.promotes(room, &level, &host)
        };
        for p in promoted {
            if let Some(Loc::Host(h)) = self.at(p) {
                to_promote.push((p, h));
            }
        }
        let need = fresh.len() + to_promote.len();
        if self.vram_free.len() < need {
            let short = need - self.vram_free.len();
            let picked = self.victims(short, [write_pages, &[]], priority);
            let mut victims = Vec::with_capacity(picked.len());
            for p in picked {
                if let Some(Loc::Vram(v)) = self.at(p) {
                    victims.push((p, v));
                }
            }
            if self.vram_free.len() + victims.len() < need {
                return Err(Error::TierFull.into());
            }
            if !victims.is_empty() {
                let mut pairs = Vec::with_capacity(victims.len());
                for &(_, v) in &victims {
                    let h = self.host_free.pop().ok_or(Error::TierFull)?;
                    pairs.push((v, h));
                }
                if !move_pages(Move::Demote, &pairs)? {
                    return Err(Error::TierFull.into());
                }
                for (&(p, v), &(_, h)) in victims.iter().zip(&pairs) {
                    self.loc[p as usize] = Some(Loc::Host(h));
                    self.vram_free.push(v);
                }
                moved = true;
            }
        }
        for p in fresh {
            let v = self.vram_free.pop().ok_or(Error::TierFull)?;
            self.loc[p as usize] = Some(Loc::Vram(v));
            self.policy.placed(p);
        }
        if !to_promote.is_empty() {
            let mut pairs = Vec::with_capacity(to_promote.len());
            for &(_, h) in &to_promote {
                let v = self.vram_free.pop().ok_or(Error::TierFull)?;
                pairs.push((v, h));
            }
            if !move_pages(Move::Promote, &pairs)? {
                return Err(Error::TierFull.into());
            }
            for (&(p, _), &(v, _)) in to_promote.iter().zip(&pairs) {
                self.loc[p as usize] = Some(Loc::Vram(v));
                self.policy.placed(p);
            }
            moved = true;
        }
        let short = keep.saturating_sub(self.vram_free.len());
        if short > 0 {
            // This step uses its new and promoted pages.
            let mut placed = alloc_pages.to_vec();
            placed.extend(to_promote.iter().map(|&(p, _)| p));
            let picked = self.victims(short, [write_pages, &placed], priority);
            self.depart(picked, move_pages)?;
        }
        // Slower ranks still read the promoted host pages.
        self.host_free.extend(to_promote.iter().map(|&(_, h)| h));
        Ok(moved)
    }
}

/// One page table with a dirty entry range per slot.
struct Table {
    entries: Vec<u32>,
    dirty: Vec<Range<usize>>,
    width: usize,
}

impl Table {
    fn new(entries: Vec<u32>, slots: usize) -> Self {
        let width = entries.len() / slots;
        Self {
            entries,
            dirty: vec![0..width; slots],
            width,
        }
    }

    fn mark(&mut self, s: usize, add: Range<usize>) {
        let cur = &mut self.dirty[s];
        if cur.start >= cur.end {
            *cur = add;
        } else {
            *cur = cur.start.min(add.start)..cur.end.max(add.end);
        }
    }

    /// Sets each entry to `want(index)` and marks changes.
    fn fill(&mut self, want: impl Fn(usize) -> u32) {
        for s in 0..self.dirty.len() {
            let base = s * self.width;
            let mut lo = self.width;
            let mut hi = 0;
            for i in 0..self.width {
                let value = want(base + i);
                if self.entries[base + i] != value {
                    self.entries[base + i] = value;
                    lo = lo.min(i);
                    hi = i + 1;
                }
            }
            if lo < hi {
                self.mark(s, lo..hi);
            }
        }
    }

    fn set(&mut self, s: usize, pages: &[u32]) {
        let lane = &mut self.entries[s * self.width..][..pages.len()];
        let mut lo = pages.len();
        let mut hi = 0;
        for (i, (&old, &new)) in lane.iter().zip(pages).enumerate() {
            if old != new {
                lo = lo.min(i);
                hi = i + 1;
            }
        }
        if lo < hi {
            lane[lo..hi].copy_from_slice(&pages[lo..hi]);
            self.mark(s, lo..hi);
        }
    }

    fn pending<'a>(&'a self, base: usize, out: &mut Vec<(u32, &'a [u32])>) {
        for (s, range) in self.dirty.iter().enumerate() {
            if range.start < range.end {
                let start = s * self.width + range.start;
                let end = s * self.width + range.end;
                out.push(((base + start) as u32, &self.entries[start..end]));
            }
        }
    }

    fn apply(&mut self, start: usize, values: &[u32]) {
        let end = start + values.len();
        if values.is_empty() || end > self.entries.len() || self.entries[start..end] == values[..] {
            return;
        }
        self.entries[start..end].copy_from_slice(values);
        let mut pos = start;
        while pos < end {
            let s = pos / self.width;
            let lo = pos % self.width;
            let stop = ((s + 1) * self.width).min(end);
            self.mark(s, lo..lo + (stop - pos));
            pos = stop;
        }
    }

    fn take_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        let mut out = Vec::new();
        for (s, range) in self.dirty.iter_mut().enumerate() {
            if range.start >= range.end {
                continue;
            }
            let r = std::mem::replace(range, 0..0);
            let start = s * self.width + r.start;
            let end = s * self.width + r.end;
            let bytes: Vec<u8> = self.entries[start..end]
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect();
            out.push((start * 4, bytes));
        }
        out
    }
}

/// Each slot's lane names its pages, in order.
pub struct Lanes {
    /// Rows per page.
    pub page: usize,
    /// Physical pages in the KV VRAM pool.
    pool: usize,
    /// Physical pages in the split index VRAM pool, or zero.
    index_pool: usize,
    table: Table,
    index_table: Option<Table>,
    tier: Option<TierMap>,
}

impl Lanes {
    /// Page table for `slots`, backed by `pool` physical pages.
    pub fn new(page: usize, entries: usize, slots: usize, pool: usize) -> Self {
        let pool = pool.max(1);
        let table = match pool < entries {
            true => vec![0; entries],
            false => (0..entries as u32).collect(),
        };
        Self {
            page,
            pool,
            index_pool: 0,
            table: Table::new(table, slots),
            index_table: None,
            tier: None,
        }
    }

    /// Split page tables for sparse KV and dense index.
    pub fn new_split(
        page: usize,
        entries: usize,
        slots: usize,
        kv_pool: usize,
        index_pool: usize,
    ) -> Self {
        let mut out = Self::new(page, entries, slots, kv_pool);
        out.index_pool = index_pool.max(1);
        out.index_table = Some(Table::new(vec![0; entries], slots));
        out
    }

    /// Whether dense index keys have a separate VRAM pool.
    pub fn split(&self) -> bool {
        self.index_table.is_some()
    }

    /// Whether `TIER_HOST` mapping is active.
    pub fn tiered(&self) -> bool {
        self.tier.is_some()
    }

    /// Enables `TIER_HOST` mapping backed by `host_pages` pinned host pages.
    pub fn enable_tier(&mut self, host_pages: usize, policy: Box<dyn Tiering>) {
        if self.index_pool > 0 && host_pages > 0 {
            let cap = self.tier_cap(host_pages);
            self.tier = Some(TierMap::new(cap, self.pool, host_pages, policy));
        }
    }

    /// The tier policy's level for a lane's page.
    pub fn class(&self, spot: Spot, slots: usize) -> u8 {
        let load = Load {
            pages: self.pool,
            slots,
        };
        self.tier
            .as_ref()
            .map_or(EVICT_COLD, |t| t.policy.class(spot, load))
    }

    fn tier_cap(&self, host_pages: usize) -> usize {
        if host_pages >= self.index_pool {
            self.index_pool
        } else {
            let slots = self.table.dirty.len();
            self.index_pool
                .min(self.pool + host_pages.saturating_sub(slots))
        }
    }

    /// Physical pages in the KV VRAM pool.
    pub fn kv_pages(&self) -> usize {
        self.pool
    }

    /// Physical pages in the split index VRAM pool.
    pub fn index_pages(&self) -> usize {
        self.index_pool
    }

    /// Physical pages in the pinned host KV pool, or zero.
    pub fn host_pages(&self) -> usize {
        self.tier.as_ref().map_or(0, |t| t.host_cap)
    }

    /// Physical pages in the logical pool.
    pub fn pages(&self) -> usize {
        match &self.tier {
            Some(t) => self.tier_cap(t.host_cap),
            None => self.pool,
        }
    }

    /// Whether slots share fewer physical pages than full capacity.
    pub fn shared(&self) -> bool {
        self.pool < self.table.entries.len() || self.index_table.is_some()
    }

    /// Resets the table to its initial page map.
    pub fn reset(&mut self) {
        if let Some(t) = &mut self.tier {
            t.reset();
        }
        if let Some(idx) = &mut self.index_table {
            idx.fill(|_| 0);
        }
        let shared = self.shared();
        self.table.fill(|i| if shared { 0 } else { i as u32 });
    }

    /// Maps slot zero linearly across the pool. Returns usable rows.
    pub fn map_solo(&mut self) -> usize {
        self.reset();
        let count = self.pool.min(self.table.width);
        let pages: Vec<u32> = (0..count as u32).collect();
        self.set(0, &pages);
        count * self.page
    }

    /// Releases freed pool pages from the host-tier map.
    pub fn release_freed(&mut self, freed: &[u32]) {
        if let Some(t) = &mut self.tier {
            t.release(freed);
        }
    }

    /// Ensures written pages live in VRAM, demoting lower-priority pages.
    pub fn ensure_vram<E: From<Error>>(
        &mut self,
        alloc_pages: &[u32],
        write_pages: &[u32],
        priority: &[u8],
        move_pages: &mut impl FnMut(Move, &[(u32, u32)]) -> Result<bool, E>,
    ) -> Result<bool, E> {
        let Some(t) = &mut self.tier else {
            return Ok(false);
        };
        t.ensure_vram(alloc_pages, write_pages, priority, move_pages)
    }

    /// Writes a slot's pages into KV and index tables.
    pub fn set(&mut self, slot: u32, pages: &[u32]) {
        let s = slot as usize;
        if let Some(idx) = &mut self.index_table {
            idx.set(s, pages);
        }
        match &self.tier {
            Some(t) => {
                let entries: Vec<u32> = pages.iter().map(|&p| t.entry(p)).collect();
                self.table.set(s, &entries);
            }
            None => self.table.set(s, pages),
        }
    }

    /// Dirty patches to broadcast. Index table entries follow KV entries.
    pub fn pending(&self) -> Vec<(u32, &[u32])> {
        let mut out = Vec::new();
        self.table.pending(0, &mut out);
        if let Some(idx) = &self.index_table {
            idx.pending(self.table.entries.len(), &mut out);
        }
        out
    }

    /// Applies rank 0's patches to the local tables.
    pub fn apply(&mut self, patches: &[(u32, Vec<u32>)]) {
        let len = self.table.entries.len();
        for (offset, values) in patches {
            let start = *offset as usize;
            match (start.checked_sub(len), &mut self.index_table) {
                (None, _) => self.table.apply(start, values),
                (Some(at), Some(idx)) => idx.apply(at, values),
                (Some(_), None) => {}
            }
        }
    }

    /// Dirty KV table `(byte_offset, bytes)` slices to upload.
    pub fn take_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        self.table.take_dirty()
    }

    /// Dirty index table `(byte_offset, bytes)` slices to upload.
    pub fn take_index_dirty(&mut self) -> Vec<(usize, Vec<u8>)> {
        self.index_table
            .as_mut()
            .map_or_else(Vec::new, Table::take_dirty)
    }
}
