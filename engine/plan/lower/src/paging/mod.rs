//! Which paged experts hold a VRAM slot.

use std::collections::HashMap;

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
