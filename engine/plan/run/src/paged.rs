//! Hot-cold VRAM cache for stacked trellis expert weights.

use crate::Error;
use crate::held::Held;
use crate::promote::{Decayed, Paged, Promote};
use crate::run::Base;
use std::collections::{BTreeMap, HashMap};
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_lower::{Loaded, Plan, Slot};
use tessera_mem::Backing;

const HEAD: usize = 16;
const TABLE: usize = 24;
const PAGED_FLAG: u32 = 0x8000_0000;
const ALIGN: usize = 256;
const RESERVE: u64 = 512 << 20;

fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// Whether `l` is a pageable stacked trellis weight.
pub(crate) fn is_pageable(l: &Loaded) -> bool {
    let name = l.slot.form.name();
    l.part.is_none()
        && l.slot.dims == 3
        && l.slot.shape[0] > 1
        && name.starts_with("trellis_k")
        && !name.ends_with("_rot")
}

/// Bytes reserved in the descriptor buffer for `experts` entries.
pub(crate) fn desc_bytes(experts: usize) -> usize {
    aligned(TABLE + experts * 8)
}

/// Descriptor layout for one paged weight, independent of backend.
#[derive(Clone, Debug)]
pub(crate) struct Spec {
    pub(crate) experts: usize,
    pub(crate) stride: usize,
    pub(crate) desc_offset: usize,
    pub(crate) desc_size: usize,
}

/// Deterministic layout of paged descriptors and non-paged weight arenas.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub(crate) specs: Vec<Spec>,
    pub(crate) by_weight: HashMap<u32, usize>,
    pub(crate) desc_total: usize,
}

impl Layout {
    /// Collects all pageable weights in `plan` in weight order.
    pub(crate) fn build(plan: &Plan) -> Option<Self> {
        let mut raw: Vec<(u32, usize, usize)> = Vec::new();
        for l in &plan.weights {
            if !is_pageable(l) || raw.iter().any(|(w, _, _)| *w == l.weight) {
                continue;
            }
            let experts = l.slot.shape[0] as usize;
            let total = l.slot.bytes();
            if total <= HEAD || !(total - HEAD).is_multiple_of(experts) {
                continue;
            }
            let stride = (total - HEAD) / experts;
            raw.push((l.weight, experts, stride));
        }
        if raw.is_empty() {
            return None;
        }
        raw.sort_unstable_by_key(|(w, _, _)| *w);
        let mut specs = Vec::with_capacity(raw.len());
        let mut by_weight = HashMap::with_capacity(raw.len());
        let mut cursor = 0usize;
        for (weight, experts, stride) in raw {
            let desc_size = desc_bytes(experts);
            let idx = specs.len();
            specs.push(Spec {
                experts,
                stride,
                desc_offset: cursor,
                desc_size,
            });
            by_weight.insert(weight, idx);
            cursor += desc_size;
        }
        Some(Self {
            specs,
            by_weight,
            desc_total: cursor,
        })
    }

    /// Moves paged weights of `plan` into a descriptor buffer.
    pub(crate) fn rewrite(&self, plan: &Plan) -> Plan {
        let mut buffers = plan.buffers.clone();
        let mut zones = if plan.zones.is_empty() {
            vec![0; buffers.len()]
        } else {
            plan.zones.clone()
        };
        // Keep non-weight buffer indices fixed for caller slots.
        let mut zone_buf: BTreeMap<u32, u32> = BTreeMap::new();
        for l in &plan.weights {
            let z = plan.zone(l.slot.buffer);
            zone_buf.entry(z).or_insert(l.slot.buffer);
            buffers[l.slot.buffer as usize] = 0;
        }
        let desc_buf = buffers.len() as u32;
        buffers.push(self.desc_total);
        zones.push(0);
        // Pack non-paged weights into each zone's existing weight buffer.
        let mut non_paged: Vec<(u32, u32, Option<u8>, Slot)> = Vec::new();
        let mut remap: HashMap<(u32, usize), (u32, usize)> = HashMap::new();
        for l in &plan.weights {
            if let Some(&idx) = self.by_weight.get(&l.weight) {
                let s = &self.specs[idx];
                remap.insert((l.slot.buffer, l.slot.offset), (desc_buf, s.desc_offset));
            } else {
                let zone = plan.zone(l.slot.buffer);
                non_paged.push((zone, l.weight, l.part, l.slot));
            }
        }
        non_paged.sort_unstable_by_key(|&(z, w, p, s)| (z, w, p, s.buffer, s.offset));
        let mut cursors: BTreeMap<u32, usize> = BTreeMap::new();
        for (zone, _, _, slot) in non_paged {
            if remap.contains_key(&(slot.buffer, slot.offset)) {
                continue;
            }
            let buf = zone_buf[&zone];
            let cur = cursors.entry(zone).or_insert(0);
            let off = *cur;
            *cur += aligned(slot.bytes());
            remap.insert((slot.buffer, slot.offset), (buf, off));
        }
        for (zone, size) in cursors {
            buffers[zone_buf[&zone] as usize] = size;
        }
        let move_slot = |mut s: Slot| {
            if let Some(&(b, off)) = remap.get(&(s.buffer, s.offset)) {
                s.buffer = b;
                s.offset = off;
            }
            s
        };
        let mut out = plan.clone();
        out.buffers = buffers;
        out.zones = zones;
        for l in &mut out.weights {
            l.slot = move_slot(l.slot);
        }
        for (_, s) in &mut out.kept {
            *s = move_slot(*s);
        }
        out.table = out.table.map(move_slot);
        out.index_table = out.index_table.map(move_slot);
        for s in &mut out.inputs {
            *s = move_slot(*s);
        }
        for o in &mut out.outputs {
            o.slot = move_slot(o.slot);
        }
        for step in &mut out.steps {
            for r in &mut step.reads {
                *r = move_slot(*r);
            }
            step.write = move_slot(step.write);
            for w in &mut step.also {
                *w = move_slot(*w);
            }
            step.table = step.table.map(move_slot);
            step.index_table = step.index_table.map(move_slot);
        }
        out
    }
}

/// Per-weight backing store and VRAM hot slots.
struct Entry<D: Device> {
    spec: Spec,
    host_buf: DeviceBuffer<D>,
    vram_offset: usize,
    resident: usize,
    hits_offset: usize,
    hits_ptr: u64,
    desc_ptr: u64,
    slot_of: Vec<Option<usize>>,
}

/// Hot VRAM expert cache backed by pinned host memory.
pub(crate) struct Pager<D: Device> {
    layout: Layout,
    vram_pool: Option<DeviceBuffer<D>>,
    vram_bytes: u64,
    hits_buf: DeviceBuffer<D>,
    raw_hits: Vec<u8>,
    entries: Vec<Entry<D>>,
    hits: Vec<u32>,
    policy: Box<dyn Promote>,
}

impl<D: Device> Pager<D> {
    /// Opens a pager when `plan` passes `budget`. Fails unless `streams`.
    pub(crate) fn open(
        ctx: &D,
        plan: &Plan,
        budget: u64,
        streams: bool,
    ) -> Result<Option<Self>, Error> {
        let planned: u64 = plan.buffers.iter().map(|b| *b as u64).sum();
        if planned <= budget {
            return Ok(None);
        }
        let Some(layout) = Layout::build(plan) else {
            return Ok(None);
        };
        let rewritten = layout.rewrite(plan);
        let non_paged: u64 = rewritten.buffers.iter().map(|b| *b as u64).sum();
        let reserve = RESERVE.min(budget / 8);
        if non_paged.saturating_add(reserve) >= budget {
            return Ok(None);
        }
        if !streams {
            return Err(Error::Streaming { planned, budget });
        }
        let avail = (budget - non_paged - reserve) as usize;
        let total_paged: usize = layout.specs.iter().map(|s| s.experts * s.stride).sum();
        let total_experts: usize = layout.specs.iter().map(|s| s.experts).sum();
        let hits_bytes = (total_experts * 4).max(16);
        let hits_buf = DeviceBuffer::alloc(ctx, hits_bytes)?;
        hits_buf.zero()?;
        let hits_base = hits_buf.device_ptr();

        // Give each paged weight a proportional share of slots.
        let mut counts = Vec::with_capacity(layout.specs.len());
        let mut pool_used = 0usize;
        for s in &layout.specs {
            let share = ((avail as u128 * (s.experts * s.stride) as u128)
                / total_paged.max(1) as u128) as usize;
            let h = (share / s.stride).min(s.experts);
            counts.push(h);
            pool_used += h * s.stride;
        }
        // Top up any leftover space with extra slots.
        for (i, s) in layout.specs.iter().enumerate() {
            while counts[i] < s.experts && pool_used + s.stride <= avail {
                counts[i] += 1;
                pool_used += s.stride;
            }
        }
        let vram_pool = if pool_used > 0 {
            Some(DeviceBuffer::alloc(ctx, pool_used)?)
        } else {
            None
        };
        let mut entries = Vec::with_capacity(layout.specs.len());
        let mut vram_cur = 0usize;
        let mut hits_cur = 0usize;
        for (s, resident) in layout.specs.iter().zip(counts) {
            let host_buf = DeviceBuffer::pinned(ctx, s.experts * s.stride)?;
            let mut slot_of = vec![None; s.experts];
            for (m, slot) in slot_of.iter_mut().enumerate().take(resident) {
                *slot = Some(m);
            }
            entries.push(Entry {
                spec: s.clone(),
                host_buf,
                vram_offset: vram_cur,
                resident,
                hits_offset: hits_cur,
                hits_ptr: hits_base + (hits_cur * 4) as u64,
                desc_ptr: 0,
                slot_of,
            });
            vram_cur += resident * s.stride;
            hits_cur += s.experts;
        }
        let pct = (pool_used as f64 / total_paged.max(1) as f64) * 100.0;
        let numa = ctx
            .numa_node()
            .map_or(String::new(), |n| format!(" (NUMA node {n})"));
        eprintln!(
            "tessera: paging {} expert weights: {:.1} GiB VRAM cache ({pct:.0}% resident) + {:.1} GiB pinned host{numa}",
            entries.len(),
            pool_used as f64 / (1u64 << 30) as f64,
            total_paged as f64 / (1u64 << 30) as f64,
        );
        Ok(Some(Self {
            layout,
            vram_pool,
            vram_bytes: (pool_used + hits_bytes) as u64,
            hits_buf,
            raw_hits: vec![0u8; hits_bytes],
            entries,
            hits: Vec::new(),
            policy: Box::new(Decayed::default()),
        }))
    }

    pub(crate) fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub(crate) fn rewrite(&self, plan: &Plan) -> Plan {
        self.layout.rewrite(plan)
    }

    pub(crate) fn slot_bytes(&self, weight: u32, part: Option<u8>) -> Option<usize> {
        if part.is_some() {
            return None;
        }
        let &idx = self.layout.by_weight.get(&weight)?;
        Some(self.layout.specs[idx].desc_size)
    }

    /// Loads a paged weight into host, VRAM and descriptor.
    pub(crate) fn upload(
        &mut self,
        ctx: &D,
        weight: u32,
        slot: Slot,
        desc_ptr: u64,
        bytes: &[u8],
        held: &mut Held,
    ) -> Result<bool, Error> {
        let Some(&idx) = self.layout.by_weight.get(&weight) else {
            return Ok(false);
        };
        let entry = &mut self.entries[idx];
        let payload = &bytes[HEAD..];
        let expect = entry.spec.experts * entry.spec.stride;
        if bytes.len() < HEAD || payload.len() != expect {
            return Err(Error::WeightSize {
                weight,
                form: slot.form.name(),
            });
        }
        // Copy all experts into pinned host backing memory.
        let host = entry
            .host_buf
            .mapped_mut()
            .expect("pinned buffer is host-mapped");
        host.copy_from_slice(payload);
        // Copy initial hot experts into VRAM cache slots.
        if entry.resident > 0
            && let Some(pool) = &self.vram_pool
        {
            let hot_bytes = entry.resident * entry.spec.stride;
            let dst = pool.device_ptr() + entry.vram_offset as u64;
            ctx.upload(dst, &payload[..hot_bytes])?;
        }
        // Build paged descriptor: 16-byte header, hits_ptr, then delta[E].
        let mut desc = vec![0u8; entry.spec.desc_size];
        desc[..HEAD].copy_from_slice(&bytes[..HEAD]);
        let side = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) | PAGED_FLAG;
        desc[8..12].copy_from_slice(&side.to_le_bytes());
        desc[HEAD..TABLE].copy_from_slice(&entry.hits_ptr.to_le_bytes());
        let pool_base = self.vram_pool.as_ref().map_or(0, DeviceBuffer::device_ptr);
        let host_base = entry.host_buf.device_ptr();
        for m in 0..entry.spec.experts {
            let eptr = match entry.slot_of[m] {
                Some(s) => pool_base + (entry.vram_offset + s * entry.spec.stride) as u64,
                None => host_base + (m * entry.spec.stride) as u64,
            };
            let delta = eptr.wrapping_sub(desc_ptr);
            let at = TABLE + m * 8;
            desc[at..at + 8].copy_from_slice(&delta.to_le_bytes());
        }
        entry.desc_ptr = desc_ptr;
        held.upload(slot, &desc)?;
        Ok(true)
    }

    /// Reports hits and makes the policy's swaps.
    pub(crate) fn maintain(&mut self, ctx: &D) -> Result<(), Error> {
        let Some(pool) = &self.vram_pool else {
            return Ok(());
        };
        self.hits_buf.read_bytes(&mut self.raw_hits)?;
        self.hits_buf.zero()?;
        let (words, _) = self.raw_hits.as_chunks::<4>();
        let mut weights = Vec::with_capacity(self.entries.len());
        for (idx, entry) in self.entries.iter().enumerate() {
            let counts = &words[entry.hits_offset..][..entry.spec.experts];
            self.hits.clear();
            self.hits
                .extend(counts.iter().map(|w| u32::from_le_bytes(*w)));
            self.policy.observe(idx, &self.hits);
            weights.push(Paged {
                slots: &entry.slot_of,
                stride: entry.spec.stride,
                ready: entry.resident > 0
                    && entry.resident < entry.spec.experts
                    && entry.desc_ptr != 0,
            });
        }
        let swaps = self.policy.swaps(&weights);
        let pool_base = pool.device_ptr();
        for swap in swaps {
            let (cold, hot) = (swap.cold, swap.hot);
            let entry = &mut self.entries[swap.weight];
            let Some(slot) = entry.slot_of[hot].take() else {
                continue;
            };
            let stride = entry.spec.stride;
            let desc_ptr = entry.desc_ptr;
            let host_base = entry.host_buf.device_ptr();
            // Point evicted expert back to its pinned host copy.
            let cold_delta = (host_base + (hot * stride) as u64).wrapping_sub(desc_ptr);
            ctx.upload(
                desc_ptr + (TABLE + hot * 8) as u64,
                &cold_delta.to_le_bytes(),
            )?;
            // Copy promoted expert from pinned host into VRAM slot.
            let vram_at = pool_base + (entry.vram_offset + slot * stride) as u64;
            let host_mem = entry
                .host_buf
                .mapped()
                .expect("pinned buffer is host-mapped");
            let off = cold * stride;
            ctx.upload(vram_at, &host_mem[off..off + stride])?;
            // Point promoted expert to its new VRAM slot.
            let hot_delta = vram_at.wrapping_sub(desc_ptr);
            ctx.upload(
                desc_ptr + (TABLE + cold * 8) as u64,
                &hot_delta.to_le_bytes(),
            )?;
            entry.slot_of[cold] = Some(slot);
        }
        Ok(())
    }
}

/// Opens `pager` if needed, returning the active plan.
pub(crate) fn prepare<'a, D: Device>(
    pager: &mut Option<Pager<D>>,
    ctx: &D,
    plan: &'a Plan,
    base: &Base,
) -> Result<(std::borrow::Cow<'a, Plan>, u64), Error> {
    if pager.is_none()
        && !ctx.unified()
        && let Some(limit) = base.budget
    {
        *pager = Pager::open(ctx, plan, limit, base.streams)?;
    }
    match pager.as_ref() {
        Some(p) => Ok((std::borrow::Cow::Owned(p.rewrite(plan)), p.vram_bytes())),
        None => Ok((std::borrow::Cow::Borrowed(plan), 0)),
    }
}

/// Pinned device-mapped host KV pool for `TIER_HOST` sparse reads.
pub(crate) struct HostKv<D: Device> {
    ctx: &'static D,
    host: u64,
    dev: u64,
    pages: usize,
    page_rows: usize,
    by_index: HashMap<u32, (usize, usize)>,
    by_slot: HashMap<(u32, usize), u64>,
}

impl<D: Device> Drop for HostKv<D> {
    fn drop(&mut self) {
        self.ctx.unpin(self.host);
    }
}

fn pair_runs(pairs: &[(u32, u32)]) -> Vec<(u32, u32, usize)> {
    let (mut out, mut i) = (Vec::new(), 0usize);
    while i < pairs.len() {
        let ((v0, h0), mut len) = (pairs[i], 1usize);
        while i + len < pairs.len()
            && pairs[i + len].0 == v0 + len as u32
            && pairs[i + len].1 == h0 + len as u32
        {
            len += 1;
        }
        out.push((v0, h0, len));
        i += len;
    }
    out
}

impl<D: Device> HostKv<D> {
    /// Pins a host-RAM pool for sparse paged KV tensors.
    pub(crate) fn open(
        ctx: &'static D,
        plan: &Plan,
        pages: usize,
        page_rows: usize,
    ) -> Option<Self> {
        if pages == 0 || page_rows == 0 || ctx.unified() {
            return None;
        }
        let mut by_index = HashMap::new();
        let mut total = 0usize;
        for (&(index, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
            if !info.sparse || info.context <= 1 || info.page_rows as usize != page_rows {
                continue;
            }
            let elems = page_rows * info.width as usize;
            let pbytes = slot.form.bytes(elems);
            if pbytes == 0 {
                continue;
            }
            by_index.insert(index, (total, pbytes));
            total += aligned(pages * pbytes);
        }
        if by_index.is_empty() || total == 0 {
            return None;
        }
        let (host, dev) = ctx.pin(total).ok()?;
        let mut out = Self {
            ctx,
            host,
            dev,
            pages,
            page_rows,
            by_index,
            by_slot: HashMap::new(),
        };
        out.bind(plan);
        Some(out)
    }

    /// Updates slot-to-device-pointer bindings for `plan`.
    pub(crate) fn bind(&mut self, plan: &Plan) {
        self.by_slot.clear();
        for &(index, slot) in &plan.kept {
            if let Some(&(off, _)) = self.by_index.get(&index) {
                self.by_slot
                    .insert((slot.buffer, slot.offset), self.dev + off as u64);
            }
        }
    }

    /// Pinned device-mapped pointer for `slot`, or zero.
    pub(crate) fn ptr(&self, slot: Slot) -> u64 {
        self.by_slot
            .get(&(slot.buffer, slot.offset))
            .copied()
            .unwrap_or(0)
    }

    fn spans(
        &self,
        held: &Held,
        plan: &Plan,
        pairs: &[(u32, u32)],
    ) -> Option<Vec<(u64, *mut u8, usize)>> {
        let runs = pair_runs(pairs);
        let mut out = Vec::new();
        for (&(index, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
            let Some(&(off, pbytes)) = self.by_index.get(&index) else {
                continue;
            };
            let elems = self.page_rows * info.width as usize;
            let linear = slot.form.offset(0) == 0 && slot.form.offset(elems) == pbytes;
            let dev_base = held
                .get(slot)
                .backing()
                .as_any()
                .downcast_ref::<DeviceBuffer<D>>()
                .expect("device runner holds device buffers")
                .device_ptr()
                + slot.start() as u64;
            let span = held.span(slot);
            let push =
                |v0: u32, h0: u32, len: usize, out: &mut Vec<(u64, *mut u8, usize)>| -> bool {
                    let v_off = slot.form.offset(v0 as usize * elems);
                    if v_off + len * pbytes > span || h0 as usize + len > self.pages {
                        return false;
                    }
                    let ptr = (self.host as usize + off + h0 as usize * pbytes) as *mut u8;
                    out.push((dev_base + v_off as u64, ptr, len * pbytes));
                    true
                };
            if linear {
                for &(v0, h0, len) in &runs {
                    if !push(v0, h0, len, &mut out) {
                        return None;
                    }
                }
            } else {
                for &(v, h) in pairs {
                    if !push(v, h, 1, &mut out) {
                        return None;
                    }
                }
            }
        }
        Some(out)
    }

    /// Copies `(vram_phys, host_phys)` sparse KV pages from VRAM to host.
    pub(crate) fn demote(
        &mut self,
        held: &Held,
        plan: &Plan,
        pairs: &[(u32, u32)],
    ) -> Result<bool, Error> {
        if pairs.is_empty() {
            return Ok(true);
        }
        let Some(raw) = self.spans(held, plan, pairs) else {
            return Ok(false);
        };
        let mut spans: Vec<(u64, &mut [u8])> = raw
            .into_iter()
            // SAFETY: distinct `host_phys` ranges inside live pinned allocation.
            .map(|(dev, ptr, len)| (dev, unsafe { std::slice::from_raw_parts_mut(ptr, len) }))
            .collect();
        self.ctx.download_batch(&mut spans)?;
        Ok(true)
    }

    /// Copies `(vram_phys, host_phys)` sparse KV pages from host to VRAM.
    pub(crate) fn promote(
        &self,
        held: &Held,
        plan: &Plan,
        pairs: &[(u32, u32)],
    ) -> Result<bool, Error> {
        if pairs.is_empty() {
            return Ok(true);
        }
        let Some(raw) = self.spans(held, plan, pairs) else {
            return Ok(false);
        };
        let spans: Vec<(u64, &[u8])> = raw
            .into_iter()
            // SAFETY: valid range inside live pinned allocation.
            .map(|(dev, ptr, len)| (dev, unsafe { std::slice::from_raw_parts(ptr, len) }))
            .collect();
        self.ctx.upload_batch(&spans)?;
        Ok(true)
    }
}

/// Pinned KV transfer state owned by one device runner.
pub(crate) struct KvMem<D: Device> {
    pub(crate) stage: Option<crate::held::KvStage<D>>,
    pub(crate) host: Option<HostKv<D>>,
}

impl<D: Device> Default for KvMem<D> {
    fn default() -> Self {
        Self {
            stage: None,
            host: None,
        }
    }
}

impl<D: Device> KvMem<D> {
    pub(crate) fn bind(&mut self, plan: &Plan) {
        if let Some(h) = &mut self.host {
            h.bind(plan);
        }
    }

    pub(crate) fn open_host(
        &mut self,
        ctx: &'static D,
        plan: &Plan,
        pages: usize,
        page: usize,
    ) -> bool {
        self.host = HostKv::open(ctx, plan, pages, page);
        self.host.is_some()
    }

    /// Host pool device pointer for kept `slot`, or zero.
    pub(crate) fn host_ptr(&self, slot: Slot) -> u64 {
        self.host.as_ref().map_or(0, |h| h.ptr(slot))
    }

    pub(crate) fn move_pages(
        &mut self,
        held: &Held,
        plan: &Plan,
        promote: bool,
        pairs: &[(u32, u32)],
    ) -> Result<bool, Error> {
        match &mut self.host {
            Some(h) if promote => h.promote(held, plan, pairs),
            Some(h) => h.demote(held, plan, pairs),
            None => Ok(false),
        }
    }
}
