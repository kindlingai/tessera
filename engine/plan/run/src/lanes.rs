//! Cross-lane hazard analysis and event synchronization for multi-stream plans.

use tessera_gpukit::device::Device;
use tessera_lower::{Slot, Step};

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

fn overlaps(a: Slot, b: Slot) -> bool {
    if a.buffer != b.buffer {
        return false;
    }
    let a_end = a.offset + a.bytes().max(1);
    let b_end = b.offset + b.bytes().max(1);
    a.offset < b_end && b.offset < a_end
}

fn writes(step: &Step) -> impl Iterator<Item = Slot> + '_ {
    std::iter::once(step.write).chain(step.also.iter().copied())
}

fn hazards(earlier: &Step, later: &Step) -> bool {
    let raw = writes(earlier).any(|w| later.reads.iter().any(|&r| overlaps(w, r)));
    let war = earlier
        .reads
        .iter()
        .any(|&r| writes(later).any(|w| overlaps(r, w)));
    let waw = writes(earlier).any(|w1| writes(later).any(|w2| overlaps(w1, w2)));
    raw || war || waw
}

impl LaneSync {
    /// Derives minimal cross-lane event records and waits for `steps`.
    pub(crate) fn plan(steps: &[Step]) -> Self {
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
                if hazards(&steps[j], &steps[i]) {
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
        Ok(())
    }
}
