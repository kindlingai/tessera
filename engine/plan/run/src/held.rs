//! Buffers a runner keeps across plans.

use crate::Error;
use std::borrow::Cow;
use std::collections::HashMap;
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_lower::form::Load;
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
    Kept,
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
        self.upload_at(slot, 0, bytes)
    }

    /// `bytes` into `slot` starting at byte `offset`.
    pub fn upload_at(&mut self, slot: Slot, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        assert!(offset + bytes.len() <= slot.bytes(), "upload_at past slot");
        let at = self.bound[slot.buffer as usize];
        Ok(self.all[at]
            .backing_mut()
            .upload(slot.start() + offset, bytes)?)
    }

    /// Reads raw bytes from `slot` starting `offset` bytes in.
    pub fn download(&self, slot: Slot, offset: usize, dst: &mut [u8]) -> Result<(), Error> {
        let start = slot.start() + offset;
        let backing = self.get(slot).backing();
        if let Some(all) = backing.mapped() {
            dst.copy_from_slice(&all[start..start + dst.len()]);
            return Ok(());
        }
        Ok(backing.download(start, dst)?)
    }

    /// Paged kept-slot metadata `(index, width, bytes, form)` for `plan`.
    pub fn kept_page_meta(
        plan: &Plan,
        page: usize,
    ) -> Option<Vec<(u32, u32, usize, &'static str)>> {
        plan.table?;
        Some(
            plan.kept
                .iter()
                .zip(&plan.kept_info)
                .filter(|(_, i)| i.context > 1 && i.page_rows as usize == page)
                .map(|(&(id, s), i)| {
                    (
                        id,
                        i.width,
                        s.form.bytes(page * i.width as usize),
                        s.form.name(),
                    )
                })
                .collect(),
        )
    }

    fn page_spans(
        &self,
        plan: &Plan,
        phys: u32,
        page: usize,
    ) -> Option<Vec<(Slot, usize, usize, usize)>> {
        let (mut out, mut at) = (Vec::new(), 0usize);
        for (&(_, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
            if info.context <= 1 || info.page_rows as usize != page {
                continue;
            }
            let elems = page * info.width as usize;
            let (off, len) = (
                slot.form.offset(phys as usize * elems),
                slot.form.bytes(elems),
            );
            if off + len > self.span(slot) {
                return None;
            }
            out.push((slot, off, at, len));
            at += len;
        }
        Some(out)
    }

    /// Downloads one physical KV page across all paged kept tensors.
    pub fn download_page(
        &self,
        plan: &Plan,
        phys: u32,
        page: usize,
        dst: &mut [u8],
    ) -> Result<bool, Error> {
        let Some(spans) = self.page_spans(plan, phys, page) else {
            return Ok(false);
        };
        for (slot, off, at, len) in spans {
            self.download(slot, off, &mut dst[at..at + len])?;
        }
        Ok(true)
    }

    /// Uploads one physical KV page across all paged kept tensors.
    pub fn upload_page(
        &mut self,
        plan: &Plan,
        phys: u32,
        page: usize,
        src: &[u8],
    ) -> Result<bool, Error> {
        let Some(spans) = self.page_spans(plan, phys, page) else {
            return Ok(false);
        };
        for (slot, off, at, len) in spans {
            self.upload_at(slot, off, &src[at..at + len])?;
        }
        Ok(true)
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
        let mut retired = Vec::new();
        self.bind_sized(plan, |_, _| None, &mut retired, alloc)?;
        Ok(!retired.is_empty())
    }

    /// As `bind`, appending retired buffers for caller-deferred drop.
    pub(crate) fn bind_sized(
        &mut self,
        plan: &Plan,
        sized: impl Fn(u32, Option<u8>) -> Option<usize>,
        retired: &mut Vec<Buffer>,
        mut alloc: impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
    ) -> Result<(), Error> {
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
        let mut bound = Vec::with_capacity(count);
        let mut seen: HashMap<u32, usize> = HashMap::new();
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
                self.bind_scratch(zone, idx, bytes, &mut alloc, retired)?
            };
            bound.push(at);
        }
        self.bound = bound;
        self.compact(retired);
        self.kept_spans.clear();
        for p in self.kept.values() {
            self.kept_spans.insert((p.buffer, p.offset), p.bytes);
        }
        Ok(())
    }

    /// Reuses a zone's `slot`-th scratch buffer, growing in place.
    fn bind_scratch(
        &mut self,
        zone: u32,
        slot: usize,
        bytes: usize,
        alloc: &mut impl FnMut(u32, usize, Holds) -> Result<Buffer, Error>,
        retired: &mut Vec<Buffer>,
    ) -> Result<usize, Error> {
        let slots = self.scratch.entry(zone).or_default();
        if let Some(&at) = slots.get(slot) {
            if self.all[at].bytes() >= bytes {
                return Ok(at);
            }
            // Scratch grows in steps, so a growing context reuses it.
            let want = bytes.max(LEAST).next_power_of_two();
            let fresh = alloc(zone, want, Holds::Other)?;
            retired.push(std::mem::replace(&mut self.all[at], fresh));
            return Ok(at);
        }
        let want = bytes.max(LEAST).next_power_of_two();
        let at = self.all.len();
        self.all.push(alloc(zone, want, Holds::Other)?);
        slots.push(at);
        Ok(at)
    }

    /// Moves unreferenced buffers in `all` into `retired`.
    fn compact(&mut self, retired: &mut Vec<Buffer>) {
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
            return;
        }
        let old = std::mem::take(&mut self.all);
        let mut remap = vec![0usize; old.len()];
        for (at, buf) in old.into_iter().enumerate() {
            if keep[at] {
                remap[at] = self.all.len();
                self.all.push(buf);
            } else {
                retired.push(buf);
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
        self.all.push(alloc(zone, need, Holds::Kept)?);
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

/// Default pinned host staging buffer size per rank (64 MiB).
const STAGE_BYTES: usize = 64 << 20;

/// Reusable pinned host buffer for batched KV DMA transfers.
pub(crate) struct KvStage<D: Device> {
    ctx: &'static D,
    host: u64,
    cap: usize,
}

impl<D: Device> KvStage<D> {
    fn open(ctx: &'static D, bytes: usize) -> Option<Self> {
        let (host, _) = ctx.pin(bytes).ok()?;
        Some(Self {
            ctx,
            host,
            cap: bytes,
        })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `host` is a live `cap`-byte pinned allocation.
        unsafe { std::slice::from_raw_parts_mut(self.host as *mut u8, self.cap) }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: `host` is a live `cap`-byte pinned allocation.
        unsafe { std::slice::from_raw_parts(self.host as *const u8, self.cap) }
    }
}

impl<D: Device> Drop for KvStage<D> {
    fn drop(&mut self) {
        self.ctx.unpin(self.host);
    }
}

struct KeptTensor {
    dev_base: u64,
    span: usize,
    form: &'static dyn Load,
    elems: usize,
    page_off: usize,
    len: usize,
    linear: bool,
}

fn kept_tensors<D: Device>(held: &Held, plan: &Plan, page: usize) -> (Vec<KeptTensor>, usize) {
    let (mut out, mut page_off) = (Vec::new(), 0usize);
    for (&(_, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
        if info.context <= 1 || info.page_rows as usize != page {
            continue;
        }
        let elems = page * info.width as usize;
        let len = slot.form.bytes(elems);
        let linear = slot.form.offset(0) == 0 && slot.form.offset(elems) == len;
        let ptr = held
            .get(slot)
            .backing()
            .as_any()
            .downcast_ref::<DeviceBuffer<D>>()
            .expect("device runner holds device buffers")
            .device_ptr();
        out.push(KeptTensor {
            dev_base: ptr + slot.start() as u64,
            span: held.span(slot),
            form: slot.form,
            elems,
            page_off,
            len,
            linear,
        });
        page_off += len;
    }
    (out, page_off)
}

fn fits_all(tensors: &[KeptTensor], phys: &[u32]) -> bool {
    phys.iter().all(|&p| {
        tensors
            .iter()
            .all(|t| t.form.offset(p as usize * t.elems) + t.len <= t.span)
    })
}

fn page_runs(phys: &[u32]) -> Vec<(usize, u32, usize)> {
    let (mut out, mut i) = (Vec::new(), 0usize);
    while i < phys.len() {
        let (p0, mut len) = (phys[i], 1usize);
        while i + len < phys.len() && phys[i + len] == p0 + len as u32 {
            len += 1;
        }
        out.push((i, p0, len));
        i += len;
    }
    out
}

fn ensure_stage<'a, D: Device>(
    ctx: &'static D,
    stage: &'a mut Option<KvStage<D>>,
    page_bytes: usize,
) -> Option<&'a mut KvStage<D>> {
    if ctx.unified() {
        return None;
    }
    if stage.as_ref().is_none_or(|s| s.cap < page_bytes) {
        let want = STAGE_BYTES.max(page_bytes);
        *stage = KvStage::open(ctx, want).or_else(|| KvStage::open(ctx, page_bytes));
    }
    stage.as_mut()
}

impl Held {
    /// Uploads `phys` pages packed in `src` over pinned async DMA.
    pub(crate) fn upload_pages<D: Device>(
        &mut self,
        ctx: &'static D,
        stage: &mut Option<KvStage<D>>,
        plan: &Plan,
        phys: &[u32],
        page: usize,
        src: &[u8],
    ) -> Result<bool, Error> {
        if phys.is_empty() {
            return Ok(true);
        }
        let (tensors, page_bytes) = kept_tensors::<D>(self, plan, page);
        if page_bytes == 0 || src.len() != phys.len() * page_bytes || !fits_all(&tensors, phys) {
            return Ok(false);
        }
        let Some(buf) = ensure_stage(ctx, stage, page_bytes) else {
            for (i, &p) in phys.iter().enumerate() {
                if !self.upload_page(plan, p, page, &src[i * page_bytes..(i + 1) * page_bytes])? {
                    return Ok(false);
                }
            }
            return Ok(true);
        };
        let chunk = (buf.cap / page_bytes).max(1);
        for (ci, p_chunk) in phys.chunks(chunk).enumerate() {
            let base = ci * chunk * page_bytes;
            let src_chunk = &src[base..base + p_chunk.len() * page_bytes];
            let merged = page_runs(p_chunk);
            let one: Vec<_> = p_chunk
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p, 1))
                .collect();
            let (mut spans, mut rem) = (Vec::new(), buf.bytes_mut());
            for t in &tensors {
                for &(i0, p0, rlen) in if t.linear { &merged } else { &one } {
                    let (dst, tail) = rem.split_at_mut(rlen * t.len);
                    rem = tail;
                    for k in 0..rlen {
                        let at = (i0 + k) * page_bytes + t.page_off;
                        dst[k * t.len..(k + 1) * t.len].copy_from_slice(&src_chunk[at..at + t.len]);
                    }
                    let dev = t.dev_base + t.form.offset(p0 as usize * t.elems) as u64;
                    spans.push((dev, &*dst));
                }
            }
            ctx.upload_batch(&spans)?;
        }
        Ok(true)
    }

    /// Downloads `phys` pages packed into `dst` over pinned async DMA.
    pub(crate) fn download_pages<D: Device>(
        &self,
        ctx: &'static D,
        stage: &mut Option<KvStage<D>>,
        plan: &Plan,
        phys: &[u32],
        page: usize,
        dst: &mut [u8],
    ) -> Result<bool, Error> {
        if phys.is_empty() {
            return Ok(true);
        }
        let (tensors, page_bytes) = kept_tensors::<D>(self, plan, page);
        if page_bytes == 0 || dst.len() != phys.len() * page_bytes || !fits_all(&tensors, phys) {
            return Ok(false);
        }
        let Some(buf) = ensure_stage(ctx, stage, page_bytes) else {
            for (i, &p) in phys.iter().enumerate() {
                let s = &mut dst[i * page_bytes..(i + 1) * page_bytes];
                if !self.download_page(plan, p, page, s)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        };
        let chunk = (buf.cap / page_bytes).max(1);
        for (ci, p_chunk) in phys.chunks(chunk).enumerate() {
            let base = ci * chunk * page_bytes;
            let dst_chunk = &mut dst[base..base + p_chunk.len() * page_bytes];
            let merged = page_runs(p_chunk);
            let one: Vec<_> = p_chunk
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p, 1))
                .collect();
            {
                let (mut spans, mut rem) = (Vec::new(), buf.bytes_mut());
                for t in &tensors {
                    for &(_, p0, rlen) in if t.linear { &merged } else { &one } {
                        let (slot, tail) = rem.split_at_mut(rlen * t.len);
                        rem = tail;
                        let dev = t.dev_base + t.form.offset(p0 as usize * t.elems) as u64;
                        spans.push((dev, slot));
                    }
                }
                ctx.download_batch(&mut spans)?;
            }
            let (raw, mut cursor) = (buf.bytes(), 0usize);
            for t in &tensors {
                for &(i0, _, rlen) in if t.linear { &merged } else { &one } {
                    let bytes = rlen * t.len;
                    let src = &raw[cursor..cursor + bytes];
                    cursor += bytes;
                    for k in 0..rlen {
                        let at = (i0 + k) * page_bytes + t.page_off;
                        dst_chunk[at..at + t.len].copy_from_slice(&src[k * t.len..(k + 1) * t.len]);
                    }
                }
            }
        }
        Ok(true)
    }
}
