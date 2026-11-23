//! Buffers a runner keeps across plans.

use crate::Error;
use std::collections::HashMap;
use tessera_lower::{Plan, Slot};
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
    /// Where each of the plan's buffers lies in `all`.
    bound: Vec<usize>,
}

impl Held {
    pub fn get(&self, slot: Slot) -> &Buffer {
        &self.all[self.bound[slot.buffer as usize]]
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

    /// Binds `plan`'s buffers. Weights, kept rows and tables stay.
    pub fn bind(
        &mut self,
        plan: &Plan,
        mut alloc: impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<(), Error> {
        let mut spare = self.compact();
        let count = plan.buffers.len();
        let mut weights: Vec<Vec<(WeightKey, usize, usize)>> = vec![Vec::new(); count];
        for l in &plan.weights {
            let entry = ((l.weight, l.part), l.slot.start(), l.slot.bytes());
            weights[l.slot.buffer as usize].push(entry);
        }
        let mut kept: Vec<Vec<(u32, usize)>> = vec![Vec::new(); count];
        for &(index, slot) in &plan.kept {
            kept[slot.buffer as usize].push((index, slot.start()));
        }
        self.bound.clear();
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
                let fits = |buf: &Buffer| buf.zone == zone && buf.bytes() >= bytes;
                match spare.iter().position(fits) {
                    Some(pos) => {
                        self.all.push(spare.swap_remove(pos));
                        self.all.len() - 1
                    }
                    None => {
                        // Scratch grows in steps, so a growing context reuses it.
                        let want = bytes.max(LEAST).next_power_of_two();
                        self.all.push(alloc(zone, want, Holds::Other)?);
                        self.all.len() - 1
                    }
                }
            };
            self.bound.push(at);
        }
        self.kept_spans.clear();
        for p in self.kept.values() {
            self.kept_spans.insert((p.buffer, p.offset), p.bytes);
        }
        Ok(())
    }

    /// Keeps persistent buffers in `all`, returning old scratch buffers.
    fn compact(&mut self) -> Vec<Buffer> {
        let old = std::mem::take(&mut self.all);
        let mut keep = vec![false; old.len()];
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
        let mut remap = vec![0usize; old.len()];
        let mut spare = Vec::new();
        for (at, buf) in old.into_iter().enumerate() {
            if keep[at] {
                remap[at] = self.all.len();
                self.all.push(buf);
            } else {
                spare.push(buf);
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
        spare
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
