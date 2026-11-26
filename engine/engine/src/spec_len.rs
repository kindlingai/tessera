//! Deterministic adaptive draft length controller for speculative decoding.

use tessera_engine_comm::{Reader, Writer};

/// Speculative decoding telemetry for one generation call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Target verification launches run.
    pub steps: usize,
    /// Draft tokens proposed across verify steps.
    pub drafted: usize,
    /// Draft tokens accepted by the target model.
    pub accepted: usize,
    /// Total tokens emitted to the caller.
    pub emitted: usize,
    /// Verify steps proposed by the DFlash drafter.
    pub dflash_steps: usize,
    /// Verify steps proposed by the MTP drafter.
    pub mtp_steps: usize,
    /// Online drafter switches during this request.
    pub switches: usize,
}

impl SpecStats {
    /// Average tokens produced per speculative verify step.
    pub fn mean_step(&self) -> f32 {
        if self.steps == 0 {
            return 0.0;
        }
        (self.accepted + self.steps) as f32 / self.steps as f32
    }
}

/// Maximum draft tokens tracked per speculative round.
pub const MAX_K: usize = 15;

/// Number of confidence buckets for optional draft log-probabilities.
pub const BUCKETS: usize = 8;

const EDGES: [f64; BUCKETS - 1] = [0.2, 0.4, 0.6, 0.75, 0.88, 0.95, 0.99];

const DEFAULT_LEVELS: [usize; 7] = [1, 2, 3, 5, 7, 11, 15];

const DEFAULT_SEEDS: [f64; MAX_K + 1] = [
    0.0, 0.75, 0.72, 0.70, 0.70, 0.70, 0.68, 0.68, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
];

const MTP_SEEDS: [f64; MAX_K + 1] = [
    0.0, 0.75, 0.70, 0.60, 0.55, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50,
];

/// Fixed deterministic step-latency model in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub base_ms: f64,
    pub per_row_ms: f64,
    pub draft_round_ms: f64,
    pub draft_each_ms: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self::dflash()
    }
}

impl CostModel {
    /// Default GB10 cost model for a DFlash drafter.
    pub const fn dflash() -> Self {
        Self {
            base_ms: 44.0,
            per_row_ms: 8.0,
            draft_round_ms: 6.0,
            draft_each_ms: 0.0,
        }
    }

    /// Default GB10 cost model for an MTP drafter.
    pub const fn mtp() -> Self {
        Self {
            base_ms: 44.0,
            per_row_ms: 8.0,
            draft_round_ms: 2.5,
            draft_each_ms: 2.0,
        }
    }

    /// Default GB10 DFlash cost inside dual-drafter mode.
    pub const fn dual_dflash() -> Self {
        Self {
            base_ms: 44.0,
            per_row_ms: 8.0,
            draft_round_ms: 6.7,
            draft_each_ms: 0.0,
        }
    }

    /// Default GB10 MTP cost inside dual-drafter mode.
    pub const fn dual_mtp() -> Self {
        Self {
            base_ms: 44.0,
            per_row_ms: 8.0,
            draft_round_ms: 4.0,
            draft_each_ms: 2.0,
        }
    }

    /// Deterministic verify latency for `rows` target rows.
    pub fn verify_ms(&self, rows: usize) -> f64 {
        let extra = rows.saturating_sub(1) as f64;
        self.base_ms + self.per_row_ms * extra
    }

    /// Deterministic drafting latency for `drafted` proposed tokens.
    pub fn draft_ms(&self, drafted: usize) -> f64 {
        let extra = drafted.saturating_sub(1) as f64;
        self.draft_round_ms + self.draft_each_ms * extra
    }
}

/// Deterministic draft-length policy and acceptance rate estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecLen {
    adaptive: bool,
    max_k: usize,
    levels: Vec<usize>,
    pub cost: CostModel,
    pub seed: [f64; MAX_K + 1],
    pn: [f64; MAX_K + 1],
    pk: [f64; MAX_K + 1],
    bn: [[f64; BUCKETS]; MAX_K + 1],
    bk: [[f64; BUCKETS]; MAX_K + 1],
    pub pos_decay: f64,
    pub bucket_decay: f64,
    pub prior: f64,
}

impl SpecLen {
    /// Fixed draft length policy capped at `max_k`.
    pub fn fixed(max_k: usize) -> Self {
        let cap = max_k.clamp(1, MAX_K);
        Self {
            adaptive: false,
            max_k: cap,
            levels: vec![cap],
            cost: CostModel::default(),
            seed: DEFAULT_SEEDS,
            pn: [0.0; MAX_K + 1],
            pk: [0.0; MAX_K + 1],
            bn: [[0.0; BUCKETS]; MAX_K + 1],
            bk: [[0.0; BUCKETS]; MAX_K + 1],
            pos_decay: 0.9,
            bucket_decay: 0.998,
            prior: 4.0,
        }
    }

    /// Adaptive draft length policy over compact levels up to `max_k`.
    pub fn adaptive(max_k: usize, cost: CostModel) -> Self {
        let cap = max_k.clamp(1, MAX_K);
        let mut s = Self::fixed(cap);
        let mut levels = Vec::new();
        for &k in &DEFAULT_LEVELS {
            if k <= cap {
                levels.push(k);
            }
        }
        if levels.last().copied() != Some(cap) {
            levels.push(cap);
        }
        s.adaptive = true;
        s.levels = levels;
        s.cost = cost;
        s
    }

    /// Sets MTP-tuned position acceptance priors.
    pub fn with_mtp_seeds(mut self) -> Self {
        self.seed = MTP_SEEDS;
        self
    }

    /// Replaces the deterministic cost model.
    pub fn with_cost(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self
    }

    /// Returns a fresh estimator keeping policy and costs.
    pub fn fresh(&self) -> Self {
        Self {
            pn: [0.0; MAX_K + 1],
            pk: [0.0; MAX_K + 1],
            bn: [[0.0; BUCKETS]; MAX_K + 1],
            bk: [[0.0; BUCKETS]; MAX_K + 1],
            ..self.clone()
        }
    }

    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    pub fn max_k(&self) -> usize {
        self.max_k
    }

    pub fn levels(&self) -> &[usize] {
        &self.levels
    }

    /// Maps a draft log-probability to its confidence bucket.
    pub fn bucket(lp: f32) -> usize {
        if !lp.is_finite() {
            return 0;
        }
        let p = (lp as f64).exp();
        let mut b = 0;
        while b < BUCKETS - 1 && p >= EDGES[b] {
            b += 1;
        }
        b
    }

    /// Smoothed acceptance rate for 1-based draft position `j`.
    pub fn rate(&self, j: usize) -> f64 {
        let j = j.clamp(1, MAX_K);
        (self.pk[j] + self.prior * self.seed[j]) / (self.pn[j] + self.prior)
    }

    /// Conditional acceptance probability for position `j` and optional `lp`.
    pub fn cond(&self, j: usize, lp: Option<f32>) -> f64 {
        let j = j.clamp(1, MAX_K);
        let r = self.rate(j);
        let Some(lp) = lp else {
            return r;
        };
        if lp.is_nan() {
            return r;
        }
        let b = Self::bucket(lp);
        let (mut n, mut k) = (0.0, 0.0);
        for i in 1..=MAX_K {
            n += self.bn[i][b];
            k += self.bk[i][b];
        }
        let pool = (k + self.prior * r) / (n + self.prior);
        (self.bk[j][b] + self.prior * pool) / (self.bn[j][b] + self.prior)
    }

    /// Expected committed tokens `E(k)` and tokens per ms.
    pub fn ratio(&self, k: usize, lp: Option<&[f32]>, drafted: usize) -> (f64, f64) {
        let mut e = 1.0;
        let mut s = 1.0;
        for j in 1..=k.min(MAX_K) {
            let l = lp.and_then(|arr| arr.get(j - 1).copied());
            s *= self.cond(j, l);
            e += s;
        }
        let ms = self.cost.verify_ms(k + 1) + self.cost.draft_ms(drafted.max(k));
        (e, e / ms)
    }

    /// Best expected tokens per ms across allowed levels.
    pub fn best_rate(&self, lp: Option<&[f32]>, n: usize, drafted: usize) -> f64 {
        self.pick(lp, n, drafted).1
    }

    /// Picks draft count `k` to verify from `n` drafts.
    pub fn choose(&self, lp: Option<&[f32]>, n: usize, drafted: usize) -> usize {
        self.pick(lp, n, drafted).0
    }

    /// Best `(k, tokens_per_ms)` across allowed levels up to `n`.
    fn pick(&self, lp: Option<&[f32]>, n: usize, drafted: usize) -> (usize, f64) {
        let limit = n.min(self.max_k);
        if limit == 0 {
            return (0, 0.0);
        }
        if !self.adaptive {
            return (limit, self.ratio(limit, lp, drafted.max(limit)).1);
        }
        let mut best_k = 0usize;
        let mut best_r = -1.0f64;
        for &k in &self.levels {
            if k > limit {
                break;
            }
            let r = self.ratio(k, lp, drafted).1;
            if r > best_r * (1.0 + 1e-9) {
                best_r = r;
                best_k = k;
            }
        }
        if best_k == 0 {
            (limit, self.ratio(limit, lp, drafted).1)
        } else {
            (best_k, best_r)
        }
    }

    /// Whether chaining another MTP draft past `n` improves rate.
    pub fn more(&self, lp: Option<&[f32]>, n: usize) -> bool {
        if !self.adaptive {
            return n < self.max_k;
        }
        let Some(&next) = self.levels.iter().find(|&&k| k > n) else {
            return false;
        };
        let now = if self.levels.iter().any(|&k| k <= n) {
            self.best_rate(lp, n, n)
        } else {
            0.0
        };
        let known = lp.map(|arr| &arr[..arr.len().min(n)]);
        self.ratio(next, known, next).1 > now
    }

    /// Updates estimates after verifying `k` drafts with `accepted` matches.
    pub fn observe(&mut self, k: usize, accepted: usize, lp: Option<&[f32]>) {
        let k = k.min(MAX_K);
        if k == 0 {
            return;
        }
        for j in 1..=MAX_K {
            self.pn[j] *= self.pos_decay;
            self.pk[j] *= self.pos_decay;
            for b in 0..BUCKETS {
                self.bn[j][b] *= self.bucket_decay;
                self.bk[j][b] *= self.bucket_decay;
            }
        }
        let stop = k.min(accepted + 1);
        for j in 1..=stop {
            let acc = if j <= accepted { 1.0 } else { 0.0 };
            self.pn[j] += 1.0;
            self.pk[j] += acc;
            if let Some(lps) = lp
                && let Some(&v) = lps.get(j - 1)
                && !v.is_nan()
            {
                let b = Self::bucket(v);
                self.bn[j][b] += 1.0;
                self.bk[j][b] += acc;
            }
        }
    }

    /// Encodes static policy settings for rank broadcast.
    pub(crate) fn encode(&self, w: &mut Writer) {
        let levels: Vec<u32> = self.levels.iter().map(|&k| k as u32).collect();
        w.u32(u32::from(self.adaptive))
            .u32(self.max_k as u32)
            .u32s(&levels)
            .f64(self.cost.base_ms)
            .f64(self.cost.per_row_ms)
            .f64(self.cost.draft_round_ms)
            .f64(self.cost.draft_each_ms)
            .f64s(&self.seed)
            .f64(self.pos_decay)
            .f64(self.bucket_decay)
            .f64(self.prior);
    }

    /// Decodes static policy settings sent by rank 0.
    pub(crate) fn decode(r: &mut Reader) -> Option<Self> {
        let adaptive = r.u32()? != 0;
        let max_k = r.u32()? as usize;
        let levels: Vec<usize> = r.u32s()?.into_iter().map(|k| k as usize).collect();
        let cost = CostModel {
            base_ms: r.f64()?,
            per_row_ms: r.f64()?,
            draft_round_ms: r.f64()?,
            draft_each_ms: r.f64()?,
        };
        let seed: [f64; MAX_K + 1] = r.f64s()?.try_into().ok()?;
        let pos_decay = r.f64()?;
        let bucket_decay = r.f64()?;
        let prior = r.f64()?;
        let mut s = Self::fixed(max_k);
        s.adaptive = adaptive;
        s.levels = levels;
        s.cost = cost;
        s.seed = seed;
        s.pos_decay = pos_decay;
        s.bucket_decay = bucket_decay;
        s.prior = prior;
        Some(s)
    }
}
