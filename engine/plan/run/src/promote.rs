//! Which paged experts hold a VRAM slot.

use tessera_lower::Plan;

/// The policy's view of one paged weight.
pub(crate) struct Paged<'a> {
    /// Each expert's VRAM slot. `None`: host memory.
    pub slots: &'a [Option<usize>],
    /// Bytes per expert.
    pub stride: usize,
    /// Whether its experts may move now.
    pub ready: bool,
}

/// Gives `hot`'s VRAM slot to `cold`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Swap {
    pub weight: usize,
    pub cold: usize,
    pub hot: usize,
}

/// Chooses the experts that trade places after a step.
pub(crate) trait Promote: Send {
    /// Adds a step's hit count per expert of `weight`.
    fn observe(&mut self, weight: usize, hits: &[u32]);
    /// The swaps to make now, best first.
    fn swaps(&mut self, weights: &[Paged]) -> Vec<Swap>;
}

/// Scores each expert by decayed hits.
pub(crate) struct Decayed {
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

/// Names the cold experts a plan will read.
pub(crate) trait Forecast: Send {
    /// Ring parts it needs. None stages nothing.
    fn parts(&self) -> usize;
    /// Cold experts of `weight` that `plan` reads, likeliest first.
    fn cold(&mut self, plan: &Plan, weight: u32, slots: &[Option<usize>]) -> Vec<usize>;
}

/// Predicts nothing, so every cold read streams from host.
pub(crate) struct Unknown;

impl Forecast for Unknown {
    fn parts(&self) -> usize {
        0
    }

    fn cold(&mut self, _: &Plan, _: u32, _: &[Option<usize>]) -> Vec<usize> {
        Vec::new()
    }
}

/// Stages every cold expert a launch almost surely reads.
pub(crate) struct Dense {
    /// The least chance of a read that stages an expert.
    pub sure: f64,
    /// Weights staged ahead of the one in use.
    pub parts: usize,
}

impl Default for Dense {
    fn default() -> Self {
        Self {
            sure: 0.99,
            parts: 4,
        }
    }
}

impl Dense {
    /// Expert rows the plan's steps pick from `weight`.
    fn picks(plan: &Plan, weight: u32) -> usize {
        let loaded = |l: &&tessera_lower::Loaded| l.weight == weight && l.part.is_none();
        let Some(slot) = plan.weights.iter().find(loaded).map(|l| l.slot) else {
            return 0;
        };
        // A reader writes one output row per pick.
        let out = slot.shape[1] as usize;
        let mut picks = 0;
        for step in &plan.steps[plan.forward()] {
            let reads =
                |r: &tessera_lower::Slot| r.buffer == slot.buffer && r.offset == slot.offset;
            if out > 0 && step.reads.iter().any(reads) && step.write.len.is_multiple_of(out) {
                picks = picks.max(step.write.len / out);
            }
        }
        picks
    }
}

impl Forecast for Dense {
    fn parts(&self) -> usize {
        self.parts
    }

    fn cold(&mut self, plan: &Plan, weight: u32, slots: &[Option<usize>]) -> Vec<usize> {
        // Picks spread evenly until hit counts say otherwise.
        let miss = 1.0 - 1.0 / slots.len().max(1) as f64;
        let read = 1.0 - miss.powf(Self::picks(plan, weight) as f64);
        if read < self.sure {
            return Vec::new();
        }
        let cold = |(m, slot): (usize, &Option<usize>)| slot.is_none().then_some(m);
        slots.iter().enumerate().filter_map(cold).collect()
    }
}
