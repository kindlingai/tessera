//! Times between `TS_MARK()` lines and graphed step boundaries.

use crate::Error;
use crate::profile::{Marks, Profile, Region};
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_gpukit::launch::{KernelArgs, Launch};
use tessera_lower::Plan;

/// Put before kernel text to record its marks.
pub const ON: &str = include_str!("marks.cuh");
/// Put before kernel text that runs unmarked.
pub const OFF: &str = "#ifndef TS_MARK\n#define TS_MARK()\n#endif\n#line 1\n";

const STAMP_ENTRY: &str = "tessera_stamp";
const STAMP_KERNEL: &str = concat!(
    include_str!("marks.cuh"),
    "extern \"C\" __global__ void tessera_stamp(\
     unsigned long long* starts, unsigned long long* accum, unsigned slot, unsigned mode) {\
     tessera_stamp_step(starts, accum, slot, mode); }\n",
);

pub(crate) const BEGIN: u32 = 0;
pub(crate) const STEP: u32 = 1;
pub(crate) const END: u32 = 2;
pub(crate) const LANE_BEGIN: u32 = 3;
pub(crate) const LANE_END: u32 = 4;
pub(crate) const LAST_STEP: u32 = 5;

/// In-graph device clock accumulators for profiled graph replays.
pub(crate) struct Stamps<D: Device> {
    kernel: D::Kernel,
    buf: DeviceBuffer<D>,
    cap: usize,
    last: Vec<u64>,
    active: Option<(u64, usize, usize, u64)>,
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Stamps<D> {
    pub(crate) fn open(ctx: &D, steps: usize) -> Result<Self, Error> {
        let cap = steps.max(64);
        let kernel = ctx.compile(STAMP_KERNEL, STAMP_ENTRY)?;
        let buf = DeviceBuffer::alloc(ctx, (2 * cap + 2) * 8)?;
        buf.zero()?;
        Ok(Self {
            kernel,
            buf,
            cap,
            last: vec![0; cap + 1],
            active: None,
        })
    }

    pub(crate) fn needs_grow(&self, steps: usize) -> bool {
        steps > self.cap
    }

    /// Grows the stamp buffer for `steps`. Caller must pause grids.
    pub(crate) fn grow(&mut self, ctx: &D, steps: usize) -> Result<(), Error> {
        if steps <= self.cap {
            return Ok(());
        }
        let cap = steps.max(self.cap * 2);
        let buf = DeviceBuffer::alloc(ctx, (2 * cap + 2) * 8)?;
        buf.zero()?;
        self.buf = buf;
        self.cap = cap;
        self.last = vec![0; cap + 1];
        self.active = None;
        Ok(())
    }

    /// Enqueues one 1-thread clock stamp node into the active stream.
    pub(crate) fn emit(&self, ctx: &D, slot: usize, mode: u32) -> Result<(), Error> {
        let starts = self.buf.device_ptr();
        let accum = starts + ((self.cap + 1) * 8) as u64;
        let mut args = KernelArgs::new();
        args.device_ptr(starts)
            .device_ptr(accum)
            .u32(slot as u32)
            .u32(mode);
        ctx.programmatic(false);
        Ok(ctx.launch(&self.kernel, (1, 1, 1), (1, 1, 1), 0, &mut args)?)
    }

    pub(crate) fn differs(&self, plan_id: u64, start: usize, end: usize) -> bool {
        self.active
            .is_some_and(|(id, s, e, _)| (id, s, e) != (plan_id, start, end))
    }

    pub(crate) fn record(&mut self, plan_id: u64, start: usize, end: usize) {
        match &mut self.active {
            Some((id, s, e, n)) if (*id, *s, *e) == (plan_id, start, end) => *n += 1,
            slot => *slot = Some((plan_id, start, end, 1)),
        }
    }

    pub(crate) fn discard(&mut self) {
        self.active = None;
    }

    /// Reads accumulated GPU ticks into `profile` without zeroing device memory.
    pub(crate) fn flush(
        &mut self,
        ctx: &D,
        plan: &Plan,
        profile: Option<&mut Profile>,
    ) -> Result<(), Error> {
        let Some((_, start, end, replays)) = self.active.take() else {
            return Ok(());
        };
        let n = end.saturating_sub(start);
        if n == 0 || replays == 0 {
            return Ok(());
        }
        ctx.finish()?;
        let mut raw = vec![0u8; (n + 1) * 8];
        let accum = self.buf.device_ptr() + ((self.cap + 1) * 8) as u64;
        ctx.download(accum, &mut raw)?;
        let ticks = ctx.ticks();
        let words = raw.as_chunks::<8>().0;
        let Some(profile) = profile else {
            for (k, bytes) in words.iter().enumerate() {
                self.last[k] = u64::from_ne_bytes(*bytes);
            }
            return Ok(());
        };
        for (k, bytes) in words[..n].iter().enumerate() {
            let now = u64::from_ne_bytes(*bytes);
            let secs = now.wrapping_sub(self.last[k]) as f64 / ticks;
            self.last[k] = now;
            profile.step_many(&plan.steps[start + k], secs, replays);
        }
        let span_now = u64::from_ne_bytes(words[n]);
        let span = span_now.wrapping_sub(self.last[n]) as f64 / ticks;
        self.last[n] = span_now;
        profile.add_many("gpu: graph", span, replays);
        Ok(())
    }
}

/// Merges mark regions from multiple launches into one summary.
pub(crate) fn merge(taken: impl IntoIterator<Item = Marks>) -> Marks {
    let mut merged = Marks::default();
    for found in taken {
        merged.lost += found.lost;
        for region in found.regions {
            match merged
                .regions
                .iter_mut()
                .find(|r| (r.from, r.to) == (region.from, region.to))
            {
                Some(existing) => existing.seconds.extend(region.seconds),
                None => merged.regions.push(region),
            }
        }
    }
    merged
}

/// Runs `step`'s marked kernel and gathers `TS_MARK()` samples.
pub(crate) fn step_marks<D: Device>(
    ctx: &D,
    marked: &mut std::collections::HashMap<String, D::Kernel>,
    ledger: &mut Option<DeviceBuffer<D>>,
    (entry, text): (&str, &str),
    mut run: impl FnMut(&D::Kernel) -> Result<(), Error>,
) -> Result<Option<Marks>, Error> {
    if !text.contains("TS_MARK") {
        return Ok(Some(Marks::default()));
    }
    let src = format!("{ON}{text}");
    let kernel = match marked.get(&src) {
        Some(k) => k.clone(),
        None => {
            let k = ctx.compile(&src, entry)?;
            marked.insert(src, k.clone());
            k
        }
    };
    let Some(at) = ctx.global(&kernel, "tessera_marks")? else {
        return Ok(None);
    };
    if ledger.is_none() {
        *ledger = Some(DeviceBuffer::alloc(ctx, COUNTS + records(BLOCKS))?);
    }
    let ptr = ledger.as_ref().map_or(0, DeviceBuffer::device_ptr);
    ctx.finish()?;
    ctx.upload(at, &ptr.to_ne_bytes())?;
    ctx.zero(ptr, COUNTS)?;
    run(&kernel)?;
    ctx.finish()?;
    let mut found = Marks::default();
    collect(ctx, ptr, &mut found)?;
    Ok(Some(found))
}

/// Blocks that record, and records each one keeps.
pub const BLOCKS: usize = 4096;
pub const ROOM: usize = 256;
/// Bytes of counts before the records.
pub const COUNTS: usize = BLOCKS * 4;

/// Bytes of records for the first `blocks` blocks.
pub fn records(blocks: usize) -> usize {
    blocks * ROOM * 16
}

/// Adds the gaps between each block's marks to `marks`.
pub fn read(counts: &[u32], records: &[u64], ticks: f64, marks: &mut Marks) {
    for (block, &count) in counts.iter().enumerate() {
        let kept = (count as usize).min(ROOM);
        marks.lost += count as u64 - kept as u64;
        let at = |i: usize| {
            (
                records[2 * (block * ROOM + i)],
                records[2 * (block * ROOM + i) + 1],
            )
        };
        for i in 1..kept {
            let ((from, start), (to, end)) = (at(i - 1), at(i));
            let seconds = end.wrapping_sub(start) as f64 / ticks;
            let (from, to) = (from as u32, to as u32);
            match marks
                .regions
                .iter_mut()
                .find(|r| (r.from, r.to) == (from, to))
            {
                Some(region) => region.seconds.push(seconds),
                None => marks.regions.push(Region {
                    from,
                    to,
                    seconds: vec![seconds],
                }),
            }
        }
    }
}

/// Downloads counts and records from `ledger` into `found`.
pub fn collect<D: Device>(
    ctx: &D,
    ledger: u64,
    found: &mut Marks,
) -> Result<(), tessera_gpukit::Error> {
    let mut bytes = vec![0u8; COUNTS];
    ctx.download(ledger, &mut bytes)?;
    let counts: Vec<u32> = bytes
        .as_chunks()
        .0
        .iter()
        .map(|b| u32::from_ne_bytes(*b))
        .collect();
    let blocks = counts.iter().rposition(|c| *c > 0).map_or(0, |at| at + 1);
    if blocks == 0 {
        return Ok(());
    }
    let mut bytes = vec![0u8; records(blocks)];
    ctx.download(ledger + COUNTS as u64, &mut bytes)?;
    let records: Vec<u64> = bytes
        .as_chunks()
        .0
        .iter()
        .map(|b| u64::from_ne_bytes(*b))
        .collect();
    read(&counts[..blocks], &records, ctx.ticks(), found);
    Ok(())
}
