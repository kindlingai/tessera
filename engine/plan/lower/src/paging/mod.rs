//! Which paged experts hold a VRAM slot.

use crate::plan::{Loaded, Plan, Slot};
use std::collections::{BTreeMap, HashMap};

#[cfg(test)]
mod paging_test;

/// The policy's view of one paged weight.
pub struct Paged<'a> {
    /// Each expert's VRAM slot. `None`: host memory.
    pub slots: &'a [Option<usize>],
    /// Bytes per expert.
    pub stride: usize,
    /// Whether its experts may move now.
    pub ready: bool,
}

/// Gives `hot`'s VRAM slot to `cold`.
#[derive(Debug, PartialEq, Eq)]
pub struct Swap {
    pub weight: usize,
    pub cold: usize,
    pub hot: usize,
}

/// A promoted expert whose bytes still copy to its slot.
#[derive(Debug, PartialEq, Eq)]
pub struct Moving {
    pub weight: usize,
    pub cold: usize,
    pub slot: usize,
}

/// Two-phase swaps: free a slot, then hand it over.
#[derive(Default)]
pub struct Moves(Vec<Moving>);

impl Moves {
    /// Frees `swap.hot`'s slot in its weight's `slots`.
    pub fn begin(&mut self, slots: &mut [Option<usize>], swap: &Swap) -> Option<usize> {
        let slot = slots[swap.hot].take()?;
        self.0.push(Moving {
            weight: swap.weight,
            cold: swap.cold,
            slot,
        });
        Some(slot)
    }

    /// The moves whose copies are done, once the device waited.
    pub fn land(&mut self) -> Vec<Moving> {
        std::mem::take(&mut self.0)
    }
}

/// Chooses the experts that trade places after a step.
pub trait Promote: Send {
    /// Adds a step's hit count per expert of `weight`.
    fn observe(&mut self, weight: usize, hits: &[u32]);
    /// The swaps to make now, best first.
    fn swaps(&mut self, weights: &[Paged]) -> Vec<Swap>;
    /// Cold `(weight, expert)` pairs worth a free slot, best first.
    fn fills(&self, weights: &[Paged]) -> Vec<(usize, usize)>;
}

/// Scores each expert by decayed hits.
pub struct Decayed {
    /// Share of a score kept per step.
    pub decay: f32,
    /// Steps between swaps.
    pub every: u64,
    /// A cold score must beat a hot one by this.
    pub ratio: f32,
    /// The least score that earns a slot.
    pub floor: f32,
    /// Most bytes moved per call.
    pub bytes: usize,
    step: u64,
    scores: Vec<Vec<f32>>,
}

impl Default for Decayed {
    fn default() -> Self {
        Self {
            decay: 0.96,
            every: 2,
            ratio: 1.2,
            floor: 0.5,
            bytes: 64 << 20,
            step: 0,
            scores: Vec::new(),
        }
    }
}

impl Decayed {
    /// The best cold and worst hot expert of one weight.
    fn pair(&self, scores: &[f32], slots: &[Option<usize>]) -> Option<(f32, usize, usize)> {
        let mut cold: Option<(usize, f32)> = None;
        let mut hot: Option<(usize, f32)> = None;
        for (m, (&score, slot)) in scores.iter().zip(slots).enumerate() {
            match slot {
                None if cold.is_none_or(|(_, s)| score > s) => cold = Some((m, score)),
                Some(_) if hot.is_none_or(|(_, s)| score < s) => hot = Some((m, score)),
                _ => {}
            }
        }
        let ((cold, cs), (hot, hs)) = (cold?, hot?);
        (cs > self.floor && cs > hs * self.ratio).then_some((cs - hs, cold, hot))
    }
}

impl Promote for Decayed {
    fn observe(&mut self, weight: usize, hits: &[u32]) {
        if self.scores.len() <= weight {
            self.scores.resize(weight + 1, Vec::new());
        }
        let scores = &mut self.scores[weight];
        scores.resize(hits.len(), 0.0);
        for (score, &h) in scores.iter_mut().zip(hits) {
            *score = *score * self.decay + h as f32;
        }
    }

    fn swaps(&mut self, weights: &[Paged]) -> Vec<Swap> {
        self.step += 1;
        if self.step > 1 && !self.step.is_multiple_of(self.every) {
            return Vec::new();
        }
        let mut ranked = Vec::new();
        for (weight, (paged, scores)) in weights.iter().zip(&self.scores).enumerate() {
            if !paged.ready {
                continue;
            }
            if let Some((gain, cold, hot)) = self.pair(scores, paged.slots) {
                ranked.push((gain, Swap { weight, cold, hot }));
            }
        }
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut moved = 0;
        let mut out = Vec::new();
        for (_, swap) in ranked {
            let stride = weights[swap.weight].stride;
            if moved + stride > self.bytes {
                break;
            }
            moved += stride;
            out.push(swap);
        }
        out
    }

    fn fills(&self, weights: &[Paged]) -> Vec<(usize, usize)> {
        let mut ranked = Vec::new();
        for (weight, (paged, scores)) in weights.iter().zip(&self.scores).enumerate() {
            if !paged.ready {
                continue;
            }
            for (m, (&score, slot)) in scores.iter().zip(paged.slots).enumerate() {
                if slot.is_none() && score > self.floor {
                    ranked.push((score, weight, m));
                }
            }
        }
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let mut moved = 0;
        let mut out = Vec::new();
        for (_, weight, m) in ranked {
            moved += weights[weight].stride;
            if moved > self.bytes {
                break;
            }
            out.push((weight, m));
        }
        out
    }
}

/// VRAM slots per weight within `avail`, and their bytes.
pub fn shares(sizes: &[(usize, usize)], avail: usize) -> (Vec<usize>, usize) {
    // Each size is a weight's experts and bytes per expert.
    let total: usize = sizes.iter().map(|(experts, stride)| experts * stride).sum();
    let mut counts = Vec::with_capacity(sizes.len());
    let mut used = 0usize;
    // Give each paged weight a proportional share of slots.
    for &(experts, stride) in sizes {
        let share = ((avail as u128 * (experts * stride) as u128) / total.max(1) as u128) as usize;
        let h = (share / stride).min(experts);
        counts.push(h);
        used += h * stride;
    }
    // Top up any leftover space with extra slots.
    for (i, &(experts, stride)) in sizes.iter().enumerate() {
        while counts[i] < experts && used + stride <= avail {
            counts[i] += 1;
            used += stride;
        }
    }
    (counts, used)
}

/// One expert in a slot lent from a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    pub weight: usize,
    pub expert: usize,
    /// Its slot's byte offset and length in the buffer.
    pub at: usize,
    pub len: usize,
    /// Set once its bytes are in the slot.
    pub ready: bool,
}

/// Expert slots taken top-down from a buffer's idle end.
#[derive(Default)]
pub struct Lent {
    bytes: usize,
    /// The lowest offset a slot uses.
    low: usize,
    loans: Vec<Loan>,
    /// Slots of ended loans: offset and length.
    free: Vec<(usize, usize)>,
}

impl Lent {
    /// The buffer's length.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Starts over with `bytes` of buffer. Returns the old loans.
    pub fn rebase(&mut self, bytes: usize) -> Vec<Loan> {
        (self.bytes, self.low) = (bytes, bytes);
        self.free.clear();
        std::mem::take(&mut self.loans)
    }

    /// The loans a plan that writes `need` bytes pushes out.
    pub fn evict(&mut self, need: usize) -> Vec<Loan> {
        let mut out = Vec::new();
        while let Some(loan) = self.loans.pop_if(|loan| loan.at < need) {
            out.push(loan);
        }
        self.free.retain(|(at, _)| *at >= need);
        let lows = self.loans.iter().map(|loan| loan.at);
        let lows = lows.chain(self.free.iter().map(|(at, _)| *at));
        self.low = lows.min().unwrap_or(self.bytes);
        out
    }

    /// A slot of `len` bytes above `need`, if room.
    pub fn take(&mut self, weight: usize, expert: usize, len: usize, need: usize) -> Option<usize> {
        // A freed slot of this size keeps the stack dense.
        let fits = |slot: &(usize, usize)| slot.1 == len && slot.0 >= need;
        let at = match self.free.iter().position(fits) {
            Some(i) => self.free.swap_remove(i).0,
            None => self.low.checked_sub(len).filter(|at| *at >= need)?,
        };
        self.low = self.low.min(at);
        let loan = Loan {
            weight,
            expert,
            at,
            len,
            ready: false,
        };
        // Loans stay sorted by offset, highest first.
        let after = self.loans.iter().position(|other| other.at < at);
        self.loans.insert(after.unwrap_or(self.loans.len()), loan);
        Some(at)
    }

    /// Marks the queued loans ready. Returns them.
    pub fn land(&mut self) -> Vec<Loan> {
        let mut out = Vec::new();
        for loan in self.loans.iter_mut().filter(|loan| !loan.ready) {
            loan.ready = true;
            out.push(*loan);
        }
        out
    }

    /// Ends the loans that `gone` names and frees their slots.
    pub fn forget(&mut self, gone: impl Fn(&Loan) -> bool) {
        for loan in self.loans.iter().filter(|loan| gone(loan)) {
            self.free.push((loan.at, loan.len));
        }
        self.loans.retain(|loan| !gone(loan));
    }

    /// Every loan, highest slot first.
    pub fn loans(&self) -> &[Loan] {
        &self.loans
    }
}

/// Decides which plans stage their cold experts.
pub trait Forecast: Send {
    /// Ring parts it needs. None stages nothing.
    fn parts(&self) -> usize;
    /// One launch of `plan` read `share` of its cold experts.
    fn observe(&mut self, plan: u64, share: f64);
    /// Whether `plan`'s launches copy cold experts ahead.
    fn stages(&self, plan: u64) -> bool;
    /// Whether copies wait for cross-rank steps on a shared bus.
    fn turns(&self) -> bool;
}

/// Predicts nothing, so every cold read streams from host.
pub struct Unknown;

impl Forecast for Unknown {
    fn parts(&self) -> usize {
        0
    }

    fn observe(&mut self, _: u64, _: f64) {}

    fn stages(&self, _: u64) -> bool {
        false
    }

    fn turns(&self) -> bool {
        false
    }
}

/// Stages a plan whose launches read most cold experts.
pub struct Dense {
    /// The read share that starts staging a plan.
    pub sure: f64,
    /// A staged plan stops below this read share.
    pub idle: f64,
    /// Share of the old estimate kept per launch.
    pub decay: f64,
    /// Weights staged ahead of the one in use.
    pub parts: usize,
    /// Copies and cross-rank steps take turns on the bus.
    pub turns: bool,
    /// Per plan: its read share, and whether it stages.
    seen: HashMap<u64, (f64, bool)>,
}

impl Default for Dense {
    fn default() -> Self {
        Self {
            sure: 0.9,
            idle: 0.5,
            decay: 0.5,
            parts: 4,
            turns: true,
            seen: HashMap::new(),
        }
    }
}

impl Forecast for Dense {
    fn parts(&self) -> usize {
        self.parts
    }

    fn observe(&mut self, plan: u64, share: f64) {
        let (read, stages) = self.seen.entry(plan).or_insert((share, false));
        *read = *read * self.decay + share * (1.0 - self.decay);
        // Two bounds keep a plan from flipping each launch.
        *stages = match *stages {
            true => *read > self.idle,
            false => *read >= self.sure,
        };
    }

    fn stages(&self, plan: u64) -> bool {
        self.seen.get(&plan).is_some_and(|(_, stages)| *stages)
    }

    fn turns(&self) -> bool {
        self.turns
    }
}

/// Bytes of a stacked weight's head, before its experts.
pub const HEAD: usize = 16;
/// Byte offset of a descriptor's delta table.
pub const TABLE: usize = 24;
const ALIGN: usize = 256;

/// `bytes` rounded up to the arena alignment.
pub fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// Whether `l` is a pageable stacked trellis weight.
fn is_pageable(l: &Loaded) -> bool {
    let name = l.slot.form.name();
    l.part.is_none()
        && l.slot.dims == 3
        && l.slot.shape[0] > 1
        && name.starts_with("trellis_k")
        && !name.ends_with("_rot")
}

/// Bytes reserved in the descriptor buffer for `experts` entries.
fn desc_bytes(experts: usize) -> usize {
    aligned(TABLE + experts * 8)
}

/// Descriptor layout for one paged weight, independent of backend.
#[derive(Clone, Debug)]
pub struct Spec {
    pub experts: usize,
    pub stride: usize,
    pub desc_offset: usize,
    pub desc_size: usize,
}

/// Deterministic layout of paged descriptors and non-paged weight arenas.
#[derive(Clone, Debug)]
pub struct Layout {
    pub specs: Vec<Spec>,
    pub by_weight: HashMap<u32, usize>,
    pub desc_total: usize,
}

impl Layout {
    /// Collects all pageable weights in `plan` in weight order.
    pub fn build(plan: &Plan) -> Option<Self> {
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
    pub fn rewrite(&self, plan: &Plan) -> Plan {
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
