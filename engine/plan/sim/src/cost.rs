//! Timed launches and the estimates they give.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The guessed cost of every launch before any work.
pub const FIXED: f64 = 1e-4;

/// How many samples one estimate blends.
const NEAR: usize = 3;

/// A plan's use of each resource.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub seconds: f64,
    /// Bytes its launches read and write.
    pub bytes: u64,
    pub launches: usize,
}

/// One kernel's timed launch.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Sample {
    /// The kernel as it prints: its name and every size.
    pub kernel: String,
    /// A hash of its source when it was timed.
    pub source: u64,
    pub sizes: Vec<usize>,
    pub work: f64,
    pub seconds: f64,
    /// The middle half of the tries, over the kept time.
    #[serde(default)]
    pub spread: f64,
    /// Seconds when its weights come from memory, if timed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cold: Option<f64>,
    /// Logarithms of the sizes and the work.
    #[serde(skip)]
    at: Vec<f64>,
}

/// Where a launch lies among samples: nearby ones cost alike.
pub fn place(sizes: &[usize], work: f64) -> Vec<f64> {
    let sizes = sizes.iter().map(|s| (*s).max(1) as f64);
    sizes.chain([work.max(1.0)]).map(f64::ln).collect()
}

impl Sample {
    pub fn new(kernel: String, source: u64, sizes: Vec<usize>, work: f64, seconds: f64) -> Self {
        Self {
            kernel,
            source,
            at: place(&sizes, work),
            sizes,
            work,
            seconds,
            spread: 0.0,
            cold: None,
        }
    }

    /// The same kernel and sizes, whatever its source or time.
    fn same(&self, other: &Sample) -> bool {
        self.kernel == other.kernel && self.work == other.work
    }
}

/// Timed launches, by kernel name.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(transparent)]
pub struct Timings(BTreeMap<String, Vec<Sample>>);

impl Timings {
    pub fn len(&self) -> usize {
        self.0.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether `now` was timed with its current source.
    pub fn known(&self, name: &str, now: &Sample) -> bool {
        self.fresh(name, now, false)
    }

    /// Whether `now` has a cold time with its current source.
    pub fn known_cold(&self, name: &str, now: &Sample) -> bool {
        self.fresh(name, now, true)
    }

    fn fresh(&self, name: &str, now: &Sample, cold: bool) -> bool {
        let fresh =
            |s: &Sample| s.same(now) && s.source == now.source && (!cold || s.cold.is_some());
        self.0.get(name).is_some_and(|t| t.iter().any(fresh))
    }

    /// Seconds at `work`, cold ones scaled by the nearest ratio.
    pub fn estimate(&self, name: &str, sizes: &[usize], work: f64, cold: bool) -> Option<f64> {
        let (timed, at) = (self.0.get(name)?, place(sizes, work));
        let scaled = |s: &Sample| match s.work > 0.0 {
            true => Some(s.seconds * work / s.work),
            false => Some(s.seconds),
        };
        let warm = estimate(timed, &at, scaled)?;
        let ratio = |s: &Sample| Some(s.cold? / s.seconds);
        Some(match cold {
            true => warm * estimate(timed, &at, ratio).unwrap_or(1.0),
            false => warm,
        })
    }

    /// Adds timed launches. A launch timed before is replaced.
    pub fn record(&mut self, samples: impl IntoIterator<Item = (&'static str, Sample)>) {
        let mut seen: Vec<(&'static str, Sample, Vec<f64>)> = Vec::new();
        for (name, sample) in samples {
            let seconds = sample.seconds;
            match seen.iter_mut().find(|(_, s, _)| s.same(&sample)) {
                Some((_, _, all)) => all.push(seconds),
                None => seen.push((name, sample, vec![seconds])),
            }
        }
        for (name, mut sample, mut all) in seen {
            // Noise only adds time, so a low quartile is kept.
            all.sort_by(f64::total_cmp);
            let (low, high) = (all[(all.len() - 1) / 4], all[3 * (all.len() - 1) / 4]);
            sample.seconds = low;
            sample.spread = (high - low) / low.max(f64::MIN_POSITIVE);
            let timed = self.0.entry(name.to_string()).or_default();
            // A warm sample timed again keeps its cold time.
            sample.cold = timed
                .iter()
                .find(|old| old.same(&sample))
                .and_then(|old| old.cold);
            timed.retain(|old| !old.same(&sample));
            timed.push(sample);
        }
    }

    /// Adds cold times to the samples they match.
    pub fn record_cold(&mut self, samples: impl IntoIterator<Item = (&'static str, Sample)>) {
        for (name, sample) in samples {
            let timed = self.0.entry(name.to_string()).or_default();
            let cold = Some(sample.seconds);
            match timed.iter_mut().find(|old| old.same(&sample)) {
                Some(old) => old.cold = cold,
                None => timed.push(Sample { cold, ..sample }),
            }
        }
    }

    /// Per kernel: cold samples and the median cold over warm.
    pub fn colder(&self) -> Vec<(&str, usize, f64)> {
        let mut out = Vec::new();
        for (name, timed) in &self.0 {
            let mut ratios: Vec<f64> = timed
                .iter()
                .filter_map(|s| Some(s.cold? / s.seconds))
                .collect();
            if ratios.is_empty() {
                continue;
            }
            ratios.sort_by(f64::total_cmp);
            out.push((name.as_str(), ratios.len(), ratios[ratios.len() / 2]));
        }
        out
    }

    /// The share of sampled seconds whose spread passes `above`.
    pub fn noisy(&self, above: f64) -> f64 {
        let (mut all, mut noisy) = (0.0, 0.0);
        for s in self.0.values().flatten() {
            all += s.seconds;
            if s.spread > above {
                noisy += s.seconds;
            }
        }
        if all > 0.0 { noisy / all } else { 0.0 }
    }

    /// Rebuilds what loading skipped.
    pub fn restore(&mut self) {
        for s in self.0.values_mut().flatten() {
            s.at = place(&s.sizes, s.work);
        }
    }
}

/// `value` at `at`, blended from the nearest samples.
fn estimate(timed: &[Sample], at: &[f64], value: impl Fn(&Sample) -> Option<f64>) -> Option<f64> {
    let mut near = [(f64::INFINITY, 0.0); NEAR];
    for s in timed.iter().filter(|s| s.at.len() == at.len()) {
        let Some(v) = value(s) else { continue };
        let apart = |(a, b): (&f64, &f64)| (a - b) * (a - b);
        let far: f64 = s.at.iter().zip(at).map(apart).sum();
        // Keeps `near` sorted, nearest first.
        let mut held = (far, v);
        for slot in &mut near {
            if held.0 < slot.0 {
                std::mem::swap(slot, &mut held);
            }
        }
    }
    // Far samples fade fast. An exact match outweighs the rest.
    let weigh = |(far, _): &(f64, f64)| 1.0 / (far * far + 1e-12);
    let found = near.iter().filter(|(far, _)| far.is_finite());
    let total: f64 = found.clone().map(weigh).sum();
    let sum: f64 = found.map(|n| weigh(n) * n.1).sum();
    (total > 0.0).then(|| sum / total)
}
