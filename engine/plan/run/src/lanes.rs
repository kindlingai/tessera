//! Lanes beside compute: their sync events and staged copies.

use crate::Error;
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_lower::Step;
use tessera_lower::hazard::hazards;
use tessera_lower::kernels::{Batches, Ring};
use tessera_lower::paging::{Forecast, Unknown};
use tessera_mem::Backing;

// The plan's copy lane is the device's copy stream.
const _: () = assert!(tessera_lower::hazard::COPY_LANE == tessera_gpukit::device::COPY_LANE);

/// Maximum distinct execution lanes tracked per plan.
const MAX_LANES: usize = 4;

/// Event records and waits keeping cross-lane buffer hazards ordered.
#[derive(Debug, Default)]
pub(crate) struct LaneSync {
    pub(crate) active: bool,
    /// Initial `(lane, event)` forks for unsynced secondary lanes.
    pub(crate) forks: Vec<(u8, usize)>,
    /// Event slots `step.lane` must wait on before running.
    pub(crate) waits: Vec<Vec<usize>>,
    /// Event slot `step.lane` records after running, if any.
    pub(crate) records: Vec<Option<usize>>,
    /// Event slots lane `0` waits on after the final step.
    pub(crate) joins: Vec<usize>,
}

impl LaneSync {
    /// Derives minimal cross-lane event records and waits for `steps`.
    pub(crate) fn plan<'a>(steps: impl IntoIterator<Item = &'a Step>, turns: bool) -> Self {
        let steps: Vec<&Step> = steps.into_iter().collect();
        if !steps.iter().any(|s| s.lane != 0) {
            return Self::default();
        }
        let n = steps.len();
        let mut waits = vec![Vec::new(); n];
        let mut records: Vec<Option<usize>> = vec![None; n];
        let mut next_event = 0usize;
        let mut forks = Vec::new();
        let mut forked = [false; MAX_LANES];
        forked[0] = true;
        let mut last_on: [Option<usize>; MAX_LANES] = [None; MAX_LANES];
        // Latest step on `src` that `dst` ordered after.
        let mut synced = [[None::<usize>; MAX_LANES]; MAX_LANES];
        let mut synced_at = vec![[None::<usize>; MAX_LANES]; n];

        for i in 0..n {
            let dst = steps[i].lane as usize;
            assert!(dst < MAX_LANES, "lane {dst} exceeds {MAX_LANES}");
            let mut need = [None::<usize>; MAX_LANES];
            for j in (0..i).rev() {
                let src = steps[j].lane as usize;
                if src == dst || need[src].is_some() {
                    continue;
                }
                if synced[dst][src].is_some_and(|s| s >= j) {
                    continue;
                }
                if hazards(steps[j], steps[i], turns) {
                    need[src] = Some(j);
                }
            }
            for src in 0..MAX_LANES {
                let Some(j) = need[src] else {
                    continue;
                };
                if synced[dst][src].is_some_and(|s| s >= j) {
                    continue;
                }
                let ev = *records[j].get_or_insert_with(|| {
                    let id = next_event;
                    next_event += 1;
                    id
                });
                waits[i].push(ev);
                if src == 0 || forked[src] {
                    forked[dst] = true;
                }
                synced[dst][src] = Some(j);
                for k in 0..MAX_LANES {
                    if let Some(t) = synced_at[j][k] {
                        synced[dst][k] = Some(synced[dst][k].map_or(t, |cur| cur.max(t)));
                    }
                }
            }
            if !forked[dst] {
                let ev = next_event;
                next_event += 1;
                forks.push((dst as u8, ev));
                forked[dst] = true;
            }
            synced[dst][dst] = Some(i);
            synced_at[i] = synced[dst];
            last_on[dst] = Some(i);
        }

        let mut joins = Vec::new();
        for lane in 1..MAX_LANES {
            let Some(last) = last_on[lane] else {
                continue;
            };
            if synced[0][lane].is_none_or(|s| s < last) {
                let ev = *records[last].get_or_insert_with(|| {
                    let id = next_event;
                    next_event += 1;
                    id
                });
                joins.push(ev);
            }
        }
        Self {
            active: true,
            forks,
            waits,
            records,
            joins,
        }
    }

    pub(crate) fn begin<D: Device>(&self, ctx: &D) -> Result<(), tessera_gpukit::Error> {
        if !self.active {
            return Ok(());
        }
        for &(lane, ev) in &self.forks {
            ctx.record_lane(0, ev)?;
            ctx.wait_lane(lane, ev)?;
        }
        Ok(())
    }

    pub(crate) fn before<D: Device>(
        &self,
        ctx: &D,
        rel: usize,
        lane: u8,
    ) -> Result<(), tessera_gpukit::Error> {
        if !self.active {
            return Ok(());
        }
        for &ev in &self.waits[rel] {
            ctx.wait_lane(lane, ev)?;
        }
        ctx.set_lane(lane);
        Ok(())
    }

    pub(crate) fn after<D: Device>(
        &self,
        ctx: &D,
        rel: usize,
        lane: u8,
    ) -> Result<(), tessera_gpukit::Error> {
        if self.active
            && let Some(ev) = self.records[rel]
        {
            ctx.record_lane(lane, ev)?;
        }
        Ok(())
    }

    pub(crate) fn finish<D: Device>(&self, ctx: &D) -> Result<(), tessera_gpukit::Error> {
        if !self.active {
            return Ok(());
        }
        for &ev in &self.joins {
            ctx.wait_lane(0, ev)?;
        }
        // Later launches, such as the end stamp, join lane 0.
        ctx.set_lane(0);
        Ok(())
    }
}

/// Where one paged weight lies now.
#[derive(Clone, Copy)]
pub(crate) struct Source<'a> {
    /// Its descriptor's device address. Zero before it loads.
    pub desc: u64,
    /// Byte offset of the delta table in the descriptor.
    pub table: usize,
    /// Its pinned host copy, as host and device addresses.
    pub host: &'a [u8],
    pub mapped: u64,
    pub stride: usize,
    /// Each expert's VRAM slot, and slot 0's address.
    pub slots: &'a [Option<usize>],
    pub pool: u64,
}

impl Source<'_> {
    /// Expert `m`'s staged and unstaged deltas.
    fn deltas(&self, m: usize, ring: u64) -> (u64, u64) {
        match self.slots[m] {
            // A hot expert keeps its VRAM slot either way.
            Some(slot) => {
                let hot = (self.pool + (slot * self.stride) as u64).wrapping_sub(self.desc);
                (hot, hot)
            }
            None => {
                let cold = self.mapped + (m * self.stride) as u64;
                (ring.wrapping_sub(self.desc), cold.wrapping_sub(self.desc))
            }
        }
    }
}

/// A batch's descriptor deltas: ring pointers, then host pointers.
struct Deltas<D: Device> {
    pinned: DeviceBuffer<D>,
    /// The descriptor address the deltas were made for.
    desc: u64,
}

/// The staging ring, its batches and its forecast.
pub(crate) struct Staging<D: Device> {
    pub(crate) forecast: Box<dyn Forecast>,
    pub(crate) ring: Ring,
    part_bytes: usize,
    pool: Option<DeviceBuffer<D>>,
    pub(crate) batches: Batches,
    deltas: Vec<Option<Deltas<D>>>,
    /// Set while a plan stopped with experts still staged.
    open: bool,
}

impl<D: Device> Staging<D> {
    /// Allocates `ring` for experts of up to `stride` bytes.
    pub(crate) fn open(
        ctx: &D,
        forecast: Box<dyn Forecast>,
        ring: Ring,
        stride: usize,
    ) -> Result<Self, Error> {
        let part_bytes = ring.slots * stride;
        let bytes = ring.parts * part_bytes;
        let pool = match bytes > 0 {
            true => Some(DeviceBuffer::alloc(ctx, bytes)?),
            false => None,
        };
        Ok(Self {
            forecast,
            ring,
            part_bytes,
            pool,
            batches: Batches::default(),
            deltas: Vec::new(),
            open: false,
        })
    }

    /// A staging with no ring. It copies nothing.
    pub(crate) fn idle() -> Self {
        Self {
            forecast: Box::new(Unknown),
            ring: Ring::default(),
            part_bytes: 0,
            pool: None,
            batches: Batches::default(),
            deltas: Vec::new(),
            open: false,
        }
    }

    /// Makes the next run rewrite every batch's deltas.
    pub(crate) fn reaim(&mut self) {
        for deltas in self.deltas.iter_mut().flatten() {
            deltas.desc = 0;
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.pool.as_ref().map_or(0, DeviceBuffer::size)
    }

    /// Readies the batches before a run of steps from `start`.
    pub(crate) fn enter<'a>(
        &mut self,
        ctx: &D,
        start: usize,
        ends: bool,
        source: impl Fn(usize) -> Source<'a>,
    ) -> Result<(), Error> {
        if self.batches.list().is_empty() {
            return Ok(());
        }
        if start == 0 && self.open {
            self.settle(ctx, &source)?;
        }
        self.aim(ctx, &source)?;
        self.open = !ends;
        Ok(())
    }

    /// Writes each batch's deltas for the current addresses.
    fn aim<'a>(&mut self, ctx: &D, source: &impl Fn(usize) -> Source<'a>) -> Result<(), Error> {
        let Some(pool) = &self.pool else {
            return Ok(());
        };
        self.deltas.resize_with(self.batches.list().len(), || None);
        for (batch, held) in self.batches.list().iter().zip(&mut self.deltas) {
            let s = source(batch.weight);
            let count = batch.experts.len();
            if count == 0 || s.desc == 0 || held.as_ref().is_some_and(|d| d.desc == s.desc) {
                continue;
            }
            // Recorded copies keep reading this buffer, so reuse it.
            let mut pinned = match held.take() {
                Some(deltas) => deltas.pinned,
                None => DeviceBuffer::pinned(ctx, count * 16)?,
            };
            let words = pinned.mapped_mut().expect("pinned buffer is host-mapped");
            let ring = pool.device_ptr() + (batch.part * self.part_bytes) as u64;
            for (i, &m) in batch.experts.iter().enumerate() {
                let (staged, unstaged) = s.deltas(m, ring + (i * s.stride) as u64);
                words[i * 8..][..8].copy_from_slice(&staged.to_le_bytes());
                words[(count + i) * 8..][..8].copy_from_slice(&unstaged.to_le_bytes());
            }
            *held = Some(Deltas {
                pinned,
                desc: s.desc,
            });
        }
        Ok(())
    }

    /// Rewrites `weight`'s deltas for `expert` after a swap.
    pub(crate) fn retarget(&mut self, weight: usize, expert: usize, s: Source<'_>) {
        let Some(pool) = &self.pool else {
            return;
        };
        for (batch, held) in self.batches.list().iter().zip(&mut self.deltas) {
            let at = batch.experts.iter().position(|&m| m == expert);
            let (Some(i), Some(deltas), true) = (at, held, batch.weight == weight) else {
                continue;
            };
            let count = batch.experts.len();
            let ring = pool.device_ptr() + (batch.part * self.part_bytes + i * s.stride) as u64;
            let (staged, unstaged) = s.deltas(expert, ring);
            let words = deltas
                .pinned
                .mapped_mut()
                .expect("pinned buffer is host-mapped");
            words[i * 8..][..8].copy_from_slice(&staged.to_le_bytes());
            words[(count + i) * 8..][..8].copy_from_slice(&unstaged.to_le_bytes());
        }
    }

    /// Points every staged expert back at host memory, waiting.
    fn settle<'a>(&self, ctx: &D, source: &impl Fn(usize) -> Source<'a>) -> Result<(), Error> {
        ctx.finish()?;
        for (batch, held) in self.batches.list().iter().zip(&self.deltas) {
            let Some(deltas) = held else {
                continue;
            };
            let (s, count) = (source(batch.weight), batch.experts.len());
            let words = deltas
                .pinned
                .mapped()
                .expect("pinned buffer is host-mapped");
            for (i, &m) in batch.experts.iter().enumerate() {
                let at = deltas.desc + (s.table + m * 8) as u64;
                ctx.upload(at, &words[(count + i) * 8..][..8])?;
            }
        }
        Ok(())
    }

    /// Queues one batch's descriptor deltas on the active lane.
    fn flip(&self, ctx: &D, id: usize, cold: bool, s: Source<'_>) -> Result<(), Error> {
        let Some(Some(deltas)) = self.deltas.get(id) else {
            return Ok(());
        };
        let experts = &self.batches.list()[id].experts;
        let words = deltas
            .pinned
            .mapped()
            .expect("pinned buffer is host-mapped");
        let first = if cold { experts.len() } else { 0 };
        for (i, &m) in experts.iter().enumerate() {
            let at = deltas.desc + (s.table + m * 8) as u64;
            ctx.queue_upload(at, &words[(first + i) * 8..][..8])?;
        }
        Ok(())
    }

    /// Queues batch `id`'s copies on the active lane.
    pub(crate) fn copy<'a>(
        &self,
        ctx: &D,
        id: usize,
        plan: u64,
        source: impl Fn(usize) -> Source<'a>,
    ) -> Result<(), Error> {
        let (Some(batch), Some(pool)) = (self.batches.list().get(id), &self.pool) else {
            return Ok(());
        };
        if let Some(evicts) = batch.evicts {
            let weight = self.batches.list()[evicts].weight;
            self.flip(ctx, evicts, true, source(weight))?;
        }
        // Unaimed and unstaged experts stay cold, with the same bytes.
        if !matches!(self.deltas.get(id), Some(Some(_))) || !self.forecast.stages(plan) {
            return Ok(());
        }
        let s = source(batch.weight);
        let ring = pool.device_ptr() + (batch.part * self.part_bytes) as u64;
        for (i, &m) in batch.experts.iter().enumerate() {
            let at = ring + (i * s.stride) as u64;
            ctx.queue_upload(at, &s.host[m * s.stride..][..s.stride])?;
        }
        self.flip(ctx, id, false, s)
    }
}
