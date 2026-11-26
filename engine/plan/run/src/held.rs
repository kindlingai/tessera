//! Buffers a runner keeps across plans.

use crate::Error;
use std::borrow::Cow;
use std::collections::HashMap;
use tessera_lower::{Loaded, Plan, Slot};
use tessera_mem::{Buffer, View, ViewMut};

/// Bytes a buffer holds at least, as zero-byte allocations fail.
const LEAST: usize = 4;

type WeightKey = (u32, Option<u8>);

/// One weight or kept entry's span inside a held buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Place {
    buffer: usize,
    offset: usize,
    bytes: usize,
}

/// Copies `len` bytes from `src` at `src_at` into `dst`.
fn copy_span(
    src: &Buffer,
    src_at: usize,
    dst: &mut Buffer,
    dst_at: usize,
    len: usize,
) -> Result<(), Error> {
    if len == 0 {
        return Ok(());
    }
    if let Some(bytes) = src.backing().mapped() {
        return Ok(dst
            .backing_mut()
            .upload(dst_at, &bytes[src_at..src_at + len])?);
    }
    let mut tmp = vec![0u8; len];
    src.backing().download(src_at, &mut tmp)?;
    Ok(dst.backing_mut().upload(dst_at, &tmp)?)
}

/// The contents of a new buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holds {
    Weights,
    Other,
}

/// Buffers that outlive one plan.
#[derive(Default)]
pub struct Held {
    /// Every live buffer.
    all: Vec<Buffer>,
    /// Each weight's places in `all`, by `(weight, part)`.
    weights: HashMap<WeightKey, Vec<Place>>,
    /// Each kept tensor's place in `all`, by kept index.
    kept: HashMap<u32, Place>,
    /// Accessible bytes of each kept `(buffer, offset)`.
    kept_spans: HashMap<(usize, usize), usize>,
    /// The page table, when kept rows are paged.
    table: Option<usize>,
    /// Grow-only scratch buffers in `all`, per zone.
    scratch: HashMap<u32, Vec<usize>>,
    /// Where each of the plan's buffers lies in `all`.
    bound: Vec<usize>,
}

impl Held {
    pub fn get(&self, slot: Slot) -> &Buffer {
        &self.all[self.bound[slot.buffer as usize]]
    }

    /// Bytes of every live buffer.
    pub fn total(&self) -> u64 {
        self.all.iter().map(|b| b.bytes() as u64).sum()
    }

    /// Buffers bound for the active plan, in plan order.
    pub(crate) fn bound(&self) -> impl Iterator<Item = &Buffer> + '_ {
        self.bound.iter().map(|&at| &self.all[at])
    }

    /// Accessible bytes of `slot`: kept room, else `slot.bytes()`.
    pub fn span(&self, slot: Slot) -> usize {
        let key = (self.bound[slot.buffer as usize], slot.start());
        self.kept_spans
            .get(&key)
            .copied()
            .unwrap_or_else(|| slot.bytes())
    }

    /// Disjoint byte slices `(buffer, start, len)` for the host.
    pub fn mapped_mut(&mut self, slices: &[(u32, usize, usize)]) -> Vec<&mut [u8]> {
        let mut by_buf: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); self.all.len()];
        for (i, &(buf, start, len)) in slices.iter().enumerate() {
            by_buf[self.bound[buf as usize]].push((i, start, start + len));
        }
        let mut out: Vec<Option<&mut [u8]>> = (0..slices.len()).map(|_| None).collect();
        for (at, (buffer, reqs)) in self.all.iter_mut().zip(&mut by_buf).enumerate() {
            if reqs.is_empty() {
                continue;
            }
            reqs.sort_unstable_by_key(|&(_, start, _)| start);
            let mut rest = buffer.backing_mut().mapped_mut().expect("host memory");
            let mut cursor = 0usize;
            for &(i, start, end) in reqs.iter() {
                assert!(start >= cursor, "overlapping slices in buffer {at}");
                let (_, tail) = rest.split_at_mut(start - cursor);
                let (slice, after) = tail.split_at_mut(end - start);
                out[i] = Some(slice);
                rest = after;
                cursor = end;
            }
        }
        out.into_iter().map(|o| o.expect("host memory")).collect()
    }

    pub fn view(&self, slot: Slot) -> View<'_> {
        self.get(slot)
            .view(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// Raw packed bytes of `slot`, copied onto the host.
    pub fn bytes(&self, slot: Slot) -> Result<Vec<u8>, Error> {
        let (start, len) = (slot.start(), slot.bytes());
        let backing = self.get(slot).backing();
        if let Some(all) = backing.mapped() {
            return Ok(all[start..start + len].to_vec());
        }
        let mut out = vec![0u8; len];
        backing.download(start, &mut out)?;
        Ok(out)
    }

    /// Reads `count` little-endian `u32` values from `slot`'s start.
    pub fn read_u32s(&self, slot: Slot, count: usize) -> Result<Vec<u32>, Error> {
        let len = count * 4;
        assert!(len <= slot.bytes(), "read_u32s past slot");
        let start = slot.start();
        let backing = self.get(slot).backing();
        let raw = if let Some(all) = backing.mapped() {
            all[start..start + len].to_vec()
        } else {
            let mut out = vec![0u8; len];
            backing.download(start, &mut out)?;
            out
        };
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect())
    }

    /// `bytes`, packed in the form, into the slot.
    pub fn upload(&mut self, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        let at = self.bound[slot.buffer as usize];
        Ok(self.all[at].backing_mut().upload(slot.start(), bytes)?)
    }

    /// The slot's bytes in place, when the host addresses them.
    pub fn mapped_slot(&mut self, slot: Slot) -> Option<&mut [u8]> {
        let at = self.bound[slot.buffer as usize];
        let all = self.all[at].backing_mut().mapped_mut()?;
        all.get_mut(slot.start()..slot.start() + slot.bytes())
    }

    pub fn view_mut(&mut self, slot: Slot) -> ViewMut<'_> {
        let at = self.bound[slot.buffer as usize];
        self.all[at].view_mut(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// Remaps subset weight arenas to already-loaded offsets in place.
    pub fn reuse_weights<'a>(
        &self,
        plan: &'a Plan,
        sized: impl Fn(u32, Option<u8>) -> Option<usize>,
    ) -> Cow<'a, Plan> {
        let mut by_buf: HashMap<u32, Vec<&Loaded>> = HashMap::new();
        for l in &plan.weights {
            by_buf.entry(l.slot.buffer).or_default().push(l);
        }
        let mut remap: HashMap<(u32, usize), usize> = HashMap::new();
        let mut sizes: HashMap<u32, usize> = HashMap::new();
        for (&b, entries) in &by_buf {
            let zone = plan.zone(b);
            let first = (entries[0].weight, entries[0].part);
            let Some(places) = self.weights.get(&first) else {
                continue;
            };
            let hit = places.iter().find_map(|p| {
                let buf = &self.all[p.buffer];
                if buf.zone != zone {
                    return None;
                }
                let mut moved = Vec::with_capacity(entries.len());
                let mut shifted = false;
                for l in entries {
                    let bytes = sized(l.weight, l.part).unwrap_or_else(|| l.slot.bytes());
                    let q = self
                        .weights
                        .get(&(l.weight, l.part))?
                        .iter()
                        .find(|q| q.buffer == p.buffer && q.bytes == bytes)?;
                    shifted |= q.offset != l.slot.offset;
                    moved.push((l.slot.offset, q.offset));
                }
                shifted.then_some((buf.bytes(), moved))
            });
            if let Some((buf_bytes, moved)) = hit {
                sizes.insert(b, buf_bytes);
                for (old_off, new_off) in moved {
                    remap.insert((b, old_off), new_off);
                }
            }
        }
        if remap.is_empty() {
            return Cow::Borrowed(plan);
        }
        let mut out = plan.clone();
        for (&b, &bytes) in &sizes {
            out.buffers[b as usize] = bytes;
        }
        for l in &mut out.weights {
            if let Some(&off) = remap.get(&(l.slot.buffer, l.slot.offset)) {
                l.slot.offset = off;
            }
        }
        for step in &mut out.steps {
            for r in &mut step.reads {
                if let Some(&off) = remap.get(&(r.buffer, r.offset)) {
                    r.offset = off;
                }
            }
        }
        Cow::Owned(out)
    }

    /// Binds `plan`'s buffers. Returns true if any buffer dropped.
    pub fn bind(
        &mut self,
        plan: &Plan,
        alloc: impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<bool, Error> {
        self.bind_sized(plan, |_, _| None, alloc)
    }

    /// As `bind`, allowing custom byte spans for paged weight descriptors.
    pub fn bind_sized(
        &mut self,
        plan: &Plan,
        sized: impl Fn(u32, Option<u8>) -> Option<usize>,
        mut alloc: impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<bool, Error> {
        let count = plan.buffers.len();
        let mut weights: Vec<Vec<(WeightKey, usize, usize)>> = vec![Vec::new(); count];
        for l in &plan.weights {
            let bytes = sized(l.weight, l.part).unwrap_or_else(|| l.slot.bytes());
            let entry = ((l.weight, l.part), l.slot.start(), bytes);
            weights[l.slot.buffer as usize].push(entry);
        }
        let mut kept: Vec<Vec<(u32, usize)>> = vec![Vec::new(); count];
        for &(index, slot) in &plan.kept {
            kept[slot.buffer as usize].push((index, slot.start()));
        }
        self.bound.clear();
        let mut seen: HashMap<u32, usize> = HashMap::new();
        let mut replaced = false;
        for (b, &bytes) in plan.buffers.iter().enumerate() {
            let zone = plan.zone(b as u32);
            let is_table = plan.table.is_some_and(|s| s.buffer as usize == b);
            let at = if !weights[b].is_empty() {
                self.bind_weights(zone, bytes, &weights[b], &mut alloc)?
            } else if !kept[b].is_empty() {
                let entries = kept_spans(&mut kept[b], bytes);
                self.bind_kept(zone, bytes, &entries, &mut alloc)?
            } else if is_table {
                self.bind_table(zone, bytes, &mut alloc)?
            } else {
                let slot = seen.entry(zone).or_insert(0);
                let idx = *slot;
                *slot += 1;
                self.bind_scratch(zone, idx, bytes, &mut alloc, &mut replaced)?
            };
            self.bound.push(at);
        }
        let dropped = self.compact();
        self.kept_spans.clear();
        for p in self.kept.values() {
            self.kept_spans.insert((p.buffer, p.offset), p.bytes);
        }
        Ok(replaced || dropped)
    }

    /// Reuses a zone's `slot`-th scratch buffer, growing in place.
    fn bind_scratch(
        &mut self,
        zone: u32,
        slot: usize,
        bytes: usize,
        alloc: &mut impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
        replaced: &mut bool,
    ) -> Result<usize, Error> {
        let slots = self.scratch.entry(zone).or_default();
        if let Some(&at) = slots.get(slot) {
            if self.all[at].bytes() >= bytes {
                return Ok(at);
            }
            // Scratch grows in steps, so a growing context reuses it.
            let want = bytes.max(LEAST).next_power_of_two();
            self.all[at] = alloc(zone, want, Holds::Other)?;
            *replaced = true;
            return Ok(at);
        }
        let want = bytes.max(LEAST).next_power_of_two();
        let at = self.all.len();
        self.all.push(alloc(zone, want, Holds::Other)?);
        slots.push(at);
        Ok(at)
    }

    /// Drops unreferenced buffers in `all`. True if any dropped.
    fn compact(&mut self) -> bool {
        let mut keep = vec![false; self.all.len()];
        for places in self.weights.values() {
            for p in places {
                keep[p.buffer] = true;
            }
        }
        for p in self.kept.values() {
            keep[p.buffer] = true;
        }
        if let Some(at) = self.table {
            keep[at] = true;
        }
        for slots in self.scratch.values() {
            for &at in slots {
                keep[at] = true;
            }
        }
        for &at in &self.bound {
            keep[at] = true;
        }
        if keep.iter().all(|&k| k) {
            return false;
        }
        let old = std::mem::take(&mut self.all);
        let mut remap = vec![0usize; old.len()];
        for (at, buf) in old.into_iter().enumerate() {
            if keep[at] {
                remap[at] = self.all.len();
                self.all.push(buf);
            }
        }
        for places in self.weights.values_mut() {
            for p in places {
                p.buffer = remap[p.buffer];
            }
        }
        for p in self.kept.values_mut() {
            p.buffer = remap[p.buffer];
        }
        if let Some(at) = &mut self.table {
            *at = remap[*at];
        }
        for slots in self.scratch.values_mut() {
            for at in slots {
                *at = remap[*at];
            }
        }
        for at in &mut self.bound {
            *at = remap[*at];
        }
        true
    }

    /// Reuses a matching weight arena, else allocates and copies.
    fn bind_weights(
        &mut self,
        zone: u32,
        plan_bytes: usize,
        entries: &[(WeightKey, usize, usize)],
        alloc: &mut impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<usize, Error> {
        let need = plan_bytes.max(LEAST);
        if let Some(places) = self.weights.get(&entries[0].0) {
            for p in places {
                let buf = &self.all[p.buffer];
                if buf.zone != zone || buf.bytes() < need {
                    continue;
                }
                let matches = entries.iter().all(|&(key, offset, bytes)| {
                    let want = Place {
                        buffer: p.buffer,
                        offset,
                        bytes,
                    };
                    self.weights
                        .get(&key)
                        .is_some_and(|list| list.contains(&want))
                });
                if matches {
                    return Ok(p.buffer);
                }
            }
        }
        let mut buf = alloc(zone, need, Holds::Weights)?;
        for &(key, offset, bytes) in entries {
            let found = self
                .weights
                .get(&key)
                .and_then(|list| list.iter().find(|p| p.bytes == bytes))
                .copied();
            if let Some(src) = found {
                copy_span(&self.all[src.buffer], src.offset, &mut buf, offset, bytes)?;
            }
        }
        let at = self.all.len();
        self.all.push(buf);
        for &(key, offset, bytes) in entries {
            let list = self.weights.entry(key).or_default();
            list.retain(|p| p.bytes == bytes);
            let place = Place {
                buffer: at,
                offset,
                bytes,
            };
            if !list.contains(&place) {
                list.push(place);
            }
        }
        Ok(at)
    }

    /// Reuses a matching kept arena, else allocates a fresh one.
    fn bind_kept(
        &mut self,
        zone: u32,
        plan_bytes: usize,
        entries: &[(u32, usize, usize)],
        alloc: &mut impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<usize, Error> {
        let need = plan_bytes.max(LEAST);
        if let Some(first) = self.kept.get(&entries[0].0).copied() {
            let buf = &self.all[first.buffer];
            if buf.zone == zone && buf.bytes() >= need {
                let matches = entries.iter().all(|&(index, offset, bytes)| {
                    let want = Place {
                        buffer: first.buffer,
                        offset,
                        bytes,
                    };
                    self.kept.get(&index) == Some(&want)
                });
                if matches {
                    return Ok(first.buffer);
                }
            }
        }
        let at = self.all.len();
        self.all.push(alloc(zone, need, Holds::Other)?);
        for &(index, offset, bytes) in entries {
            let place = Place {
                buffer: at,
                offset,
                bytes,
            };
            self.kept.insert(index, place);
        }
        Ok(at)
    }

    /// Reuses the page table buffer when its size matches.
    fn bind_table(
        &mut self,
        zone: u32,
        plan_bytes: usize,
        alloc: &mut impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<usize, Error> {
        let need = plan_bytes.max(LEAST);
        if let Some(at) = self.table
            && self.all[at].zone == zone
            && self.all[at].bytes() == need
        {
            return Ok(at);
        }
        let at = self.all.len();
        self.all.push(alloc(zone, need, Holds::Other)?);
        self.table = Some(at);
        Ok(at)
    }
}

/// Computes each kept entry's byte span within its arena.
fn kept_spans(items: &mut [(u32, usize)], total: usize) -> Vec<(u32, usize, usize)> {
    items.sort_unstable_by_key(|&(_, offset)| offset);
    let mut out = Vec::with_capacity(items.len());
    for i in 0..items.len() {
        let (index, offset) = items[i];
        let end = if i + 1 < items.len() {
            items[i + 1].1
        } else {
            total
        };
        out.push((index, offset, end - offset));
    }
    out
}
