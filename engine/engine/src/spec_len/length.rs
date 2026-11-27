//! How an estimator's rates become a draft length.

use super::{DEFAULT_LEVELS, SpecLen};

/// Chooses draft lengths. Ranks name a choice by its index.
pub(super) trait Length: Sync {
    /// The lengths the choice may give, up to `cap`.
    fn levels(&self, cap: usize) -> Vec<usize>;
    /// The length to verify and its tokens per ms.
    fn pick(
        &self,
        spec: &SpecLen,
        lp: Option<&[f32]>,
        limit: usize,
        drafted: usize,
    ) -> (usize, f64);
    /// Whether one more MTP draft past `n` pays.
    fn more(&self, spec: &SpecLen, lp: Option<&[f32]>, n: usize) -> bool;
    /// MTP drafts to chain from `avail`.
    fn chain(&self, spec: &SpecLen, avail: usize) -> usize;
}

/// Every choice, in wire order.
pub(super) static LENGTHS: [&dyn Length; 2] = [&Fixed, &Adaptive];
pub(super) const FIXED: usize = 0;
pub(super) const ADAPTIVE: usize = 1;

/// Always the longest length allowed.
struct Fixed;

impl Length for Fixed {
    fn levels(&self, cap: usize) -> Vec<usize> {
        vec![cap]
    }

    fn pick(
        &self,
        spec: &SpecLen,
        lp: Option<&[f32]>,
        limit: usize,
        drafted: usize,
    ) -> (usize, f64) {
        (limit, spec.ratio(limit, lp, drafted.max(limit)).1)
    }

    fn more(&self, spec: &SpecLen, _lp: Option<&[f32]>, n: usize) -> bool {
        n < spec.max_k
    }

    fn chain(&self, _spec: &SpecLen, avail: usize) -> usize {
        avail
    }
}

/// The level with the most expected tokens per ms.
struct Adaptive;

impl Length for Adaptive {
    fn levels(&self, cap: usize) -> Vec<usize> {
        let mut levels: Vec<usize> = DEFAULT_LEVELS
            .iter()
            .copied()
            .filter(|&k| k <= cap)
            .collect();
        if levels.last().copied() != Some(cap) {
            levels.push(cap);
        }
        levels
    }

    fn pick(
        &self,
        spec: &SpecLen,
        lp: Option<&[f32]>,
        limit: usize,
        drafted: usize,
    ) -> (usize, f64) {
        let mut best_k = 0usize;
        let mut best_r = -1.0f64;
        for &k in &spec.levels {
            if k > limit {
                break;
            }
            let r = spec.ratio(k, lp, drafted).1;
            if r > best_r * (1.0 + 1e-9) {
                best_r = r;
                best_k = k;
            }
        }
        match best_k {
            0 => (limit, spec.ratio(limit, lp, drafted).1),
            _ => (best_k, best_r),
        }
    }

    fn more(&self, spec: &SpecLen, lp: Option<&[f32]>, n: usize) -> bool {
        let Some(&next) = spec.levels.iter().find(|&&k| k > n) else {
            return false;
        };
        let now = match spec.levels.iter().any(|&k| k <= n) {
            true => spec.best_rate(lp, n, n),
            false => 0.0,
        };
        let known = lp.map(|arr| &arr[..arr.len().min(n)]);
        spec.ratio(next, known, next).1 > now
    }

    fn chain(&self, spec: &SpecLen, avail: usize) -> usize {
        let mut n = 1;
        while n < avail && self.more(spec, None, n) {
            n += 1;
        }
        n
    }
}
