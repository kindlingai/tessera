//! What launches cost on one device, fitted to measurements.

use crate::kernel::Kernel;
use crate::plan::{Plan, Step};
use std::collections::BTreeMap;

/// What a plan uses of each resource.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub seconds: f64,
    /// Bytes its launches read and write.
    pub bytes: u64,
    pub launches: usize,
}

/// One device's launch times, by kernel name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Device {
    /// Work and seconds, sorted by work.
    curves: BTreeMap<&'static str, Vec<(f64, f64)>>,
    /// A rule to pick wherever it applies, whatever it costs.
    favor: Option<&'static str>,
}

/// What every launch is guessed to cost before any work.
const FIXED: f64 = 1e-4;

impl Device {
    /// No measurements: each kernel's own guess decides.
    pub fn guess() -> Self {
        Self::default()
    }

    /// The same device, never choosing `kernel`.
    pub fn without(mut self, kernel: &'static str) -> Self {
        self.curves.insert(kernel, vec![(0.0, f64::INFINITY)]);
        self
    }

    /// The same device, always choosing `rule` where it applies.
    pub fn favoring(mut self, rule: &'static str) -> Self {
        self.favor = Some(rule);
        self
    }

    pub fn favors(&self, rule: &str) -> bool {
        self.favor == Some(rule)
    }

    /// Estimated seconds for one launch writing `len` elements.
    pub fn seconds(&self, k: &dyn Kernel, len: usize) -> f64 {
        let work = k.work(len);
        match self.curves.get(k.name()) {
            Some(curve) => along(curve, work),
            None => FIXED + k.guess() * work,
        }
    }

    /// The plan's estimated use of each resource.
    pub fn cost(&self, plan: &Plan) -> Cost {
        let mut total = Cost::default();
        for step in &plan.steps {
            let moved: usize = step.reads.iter().map(|r| r.len).sum();
            total.seconds += self.seconds(&*step.kernel, step.write.len);
            total.bytes += 4 * (moved + step.write.len) as u64;
            total.launches += step.launches().len();
        }
        total
    }

    /// Replaces the curve of every kernel that `samples` measured.
    pub fn fit<'a>(&mut self, samples: impl IntoIterator<Item = (&'a Step, f64)>) {
        let mut by: BTreeMap<&'static str, BTreeMap<u64, Vec<f64>>> = BTreeMap::new();
        for (step, seconds) in samples {
            let units = step.kernel.work(step.write.len) as u64;
            let sizes = by.entry(step.kernel.name()).or_default();
            sizes.entry(units).or_default().push(seconds);
        }
        for (name, sizes) in by {
            let point = |(units, mut seen): (u64, Vec<f64>)| {
                // The median shrugs off one slow launch.
                seen.sort_by(f64::total_cmp);
                (units as f64, seen[seen.len() / 2])
            };
            self.curves
                .insert(name, sizes.into_iter().map(point).collect());
        }
    }
}

/// Seconds at `x`: between measured points, or scaled past them.
fn along(curve: &[(f64, f64)], x: f64) -> f64 {
    let (first, last) = (curve[0], curve[curve.len() - 1]);
    if x <= first.0 {
        return first.1;
    }
    if x >= last.0 {
        let scale = if last.0 > 0.0 { x / last.0 } else { 1.0 };
        return last.1 * scale;
    }
    let above = curve.iter().position(|p| p.0 >= x).expect("x is inside");
    let (a, b) = (curve[above - 1], curve[above]);
    a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0)
}
