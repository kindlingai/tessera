//! Token choice: sampling from logits, draft acceptance and answer limits.

use crate::guide::Rules;
use crate::guide::{Edit, edit};
use crate::{Allowed, Reason, Request};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};
use tessera_engine_comm::{Reader, Writer};

/// Sampling settings for one answer. Zero temperature is greedy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
        }
    }
}

impl Sampling {
    /// Whether this request uses greedy argmax decoding.
    pub fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Resolves an omitted seed once so all ranks sample alike.
    pub(crate) fn seeded(self) -> Self {
        match self.greedy() || self.seed.is_some() {
            true => self,
            false => Self {
                seed: Some(fresh_seed()),
                ..self
            },
        }
    }

    /// Serializes sampling settings for cross-rank broadcast.
    pub(crate) fn encode(&self, w: &mut Writer) {
        let seed = self.seed.unwrap_or(0);
        w.u32(self.temperature.to_bits())
            .u32(self.top_p.to_bits())
            .u32(self.top_k as u32)
            .u32(self.min_p.to_bits())
            .u32(u32::from(self.seed.is_some()))
            .u32(seed as u32)
            .u32((seed >> 32) as u32);
    }

    /// Deserializes sampling settings from a rank broadcast.
    pub(crate) fn decode(r: &mut Reader) -> Option<Self> {
        let temperature = f32::from_bits(r.u32()?);
        let top_p = f32::from_bits(r.u32()?);
        let top_k = r.u32()? as usize;
        let min_p = f32::from_bits(r.u32()?);
        let has_seed = r.u32()? != 0;
        let lo = r.u32()? as u64;
        let hi = r.u32()? as u64;
        Some(Self {
            temperature,
            top_p,
            top_k,
            min_p,
            seed: has_seed.then_some(lo | (hi << 32)),
        })
    }
}

/// Per-sequence deterministic PRNG and token sampler.
pub(crate) struct Sampler {
    cfg: Sampling,
    state: u64,
}

impl Sampler {
    pub(crate) fn new(cfg: Sampling) -> Self {
        Self {
            cfg,
            state: cfg.seed.unwrap_or_else(fresh_seed),
        }
    }

    /// The candidate weights of one row, after `edits` and filters.
    pub(crate) fn weights(
        &self,
        logits: &mut [f32],
        allowed: Option<&Allowed>,
        edits: &[Edit],
    ) -> Weights {
        edit(logits, edits);
        if let Some(Allowed::One(id)) = allowed {
            return Weights::one(*id);
        }
        if self.cfg.greedy() {
            return Weights::one(argmax(logits, allowed));
        }
        let mut max = f32::NEG_INFINITY;
        each_allowed(logits, allowed, |_, v| {
            if v > max {
                max = v;
            }
        });
        if !max.is_finite() {
            return Weights::one(argmax(logits, allowed));
        }
        let temp = self.cfg.temperature;
        let mut floor = max - 20.0 * temp;
        if self.cfg.min_p > 0.0 {
            floor = floor.max(max + temp * self.cfg.min_p.ln());
        }
        let mut cands = Vec::new();
        each_allowed(logits, allowed, |id, v| {
            if v >= floor {
                cands.push((id, v));
            }
        });
        if cands.is_empty() {
            return Weights::one(argmax(logits, allowed));
        }
        filter_top(&mut cands, self.cfg.top_k, self.cfg.top_p, max, temp);
        for c in &mut cands {
            c.1 = ((c.1 - max) / temp).exp();
        }
        Weights::of(cands)
    }

    /// Whether this sampler uses greedy argmax decoding.
    pub(crate) fn greedy(&self) -> bool {
        self.cfg.greedy()
    }

    /// Draws one token from `logits` after optional guide rules.
    pub(crate) fn draw_row(
        &mut self,
        logits: &mut [f32],
        allowed: Option<&Allowed>,
        edits: &[Edit],
    ) -> u32 {
        let u = self.unit();
        self.weights(logits, allowed, edits).draw(u)
    }

    /// Verifies `drafts` against sampled target `logits`.
    pub(crate) fn verify(
        &mut self,
        drafts: &[u32],
        logits: &mut [f32],
        vocab: usize,
        rules: &[Rules],
        fixed: usize,
    ) -> (usize, u32) {
        let n = drafts.len();
        let mut rows = Vec::with_capacity(n + 1);
        for i in 0..=n {
            if i < fixed {
                rows.push(Weights::one(drafts[i]));
            } else {
                let row = &mut logits[i * vocab..(i + 1) * vocab];
                let (allowed, edits) = match rules.get(i) {
                    Some((a, e)) => (Some(a), e.as_slice()),
                    None => (None, &[][..]),
                };
                rows.push(self.weights(row, allowed, edits));
            }
        }
        let units: Vec<f32> = (0..=n).map(|_| self.unit()).collect();
        accept(drafts, Target::Drawn(&rows, &units))
    }

    /// A uniform in `[0, 1)`. Each row draws one.
    pub(crate) fn unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }
}

/// One row's candidate ids and their unnormalized weights.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Weights {
    cands: Vec<(u32, f32)>,
    total: f32,
}

impl Weights {
    /// All the weight on `id`.
    pub fn one(id: u32) -> Self {
        Self::of(vec![(id, 1.0)])
    }

    /// Weights from `(id, weight)` pairs.
    pub fn of(cands: Vec<(u32, f32)>) -> Self {
        let total = cands.iter().map(|c| c.1).sum();
        Self { cands, total }
    }

    /// The share of the weight on `id`.
    pub fn share(&self, id: u32) -> f32 {
        let on = self.cands.iter().filter(|c| c.0 == id);
        on.map(|c| c.1).sum::<f32>() / self.total
    }

    /// The id at `unit` of the weight.
    pub fn draw(&self, unit: f32) -> u32 {
        let target = unit * self.total;
        let mut acc = 0.0f32;
        for &(id, w) in &self.cands {
            acc += w;
            if target < acc {
                return id;
            }
        }
        self.cands.last().map_or(0, |c| c.0)
    }

    /// As `draw`, over the ids other than `id`.
    fn redraw(&self, id: u32, unit: f32) -> u32 {
        let rest = self.cands.iter().filter(|c| c.0 != id);
        let rest = Self::of(rest.copied().collect());
        match rest.total > 0.0 {
            true => rest.draw(unit),
            false => self.draw(unit),
        }
    }
}

/// The target's answer after each verify row.
#[derive(Clone, Copy, Debug)]
pub enum Target<'a> {
    /// The likeliest id after each row.
    Ids(&'a [u32]),
    /// Each row's weights, and one uniform for each row.
    Drawn(&'a [Weights], &'a [f32]),
}

/// The count of held drafts, and the next token.
pub fn accept(drafts: &[u32], target: Target) -> (usize, u32) {
    // Zero drafts is a plain pick.
    let n = drafts.len();
    match target {
        Target::Ids(ids) => {
            assert_eq!(ids.len(), n + 1);
            let held = drafts.iter().zip(ids).take_while(|(d, i)| d == i).count();
            (held, ids[held])
        }
        Target::Drawn(rows, units) => {
            assert_eq!((rows.len(), units.len()), (n + 1, n + 1));
            for (i, &draft) in drafts.iter().enumerate() {
                if units[i] >= rows[i].share(draft) {
                    return (i, rows[i].redraw(draft, units[n]));
                }
            }
            (n, rows[n].draw(units[n]))
        }
    }
}

/// Counts `id` as an answer token, unless it stops.
pub(crate) fn take(
    request: &Request,
    made: &mut usize,
    id: u32,
    send: impl FnOnce() -> bool,
) -> Option<Reason> {
    if *made >= request.min_tokens && request.stop.contains(&id) {
        return Some(Reason::Stop);
    }
    *made += 1;
    if !send() {
        return Some(Reason::Stop);
    }
    (*made >= request.max_tokens).then_some(Reason::Length)
}

/// Calls `f` for every allowed finite `(id, logit)`.
fn each_allowed(logits: &[f32], allowed: Option<&Allowed>, mut f: impl FnMut(u32, f32)) {
    match allowed {
        None | Some(Allowed::Any) => {
            for (i, &v) in logits.iter().enumerate() {
                if v.is_finite() {
                    f(i as u32, v);
                }
            }
        }
        Some(Allowed::One(id)) => {
            if let Some(&v) = logits.get(*id as usize)
                && v.is_finite()
            {
                f(*id, v);
            }
        }
        Some(Allowed::Few(ids)) => {
            for &id in ids {
                if let Some(&v) = logits.get(id as usize)
                    && v.is_finite()
                {
                    f(id, v);
                }
            }
        }
        Some(Allowed::Except(banned)) => {
            for (i, &v) in logits.iter().enumerate() {
                if v.is_finite() && !banned.contains(&(i as u32)) {
                    f(i as u32, v);
                }
            }
        }
        Some(Allowed::Bits(bits)) => {
            for (w, &word) in bits.iter().enumerate() {
                let mut left = word;
                while left != 0 {
                    let b = left.trailing_zeros() as usize;
                    let idx = w * 32 + b;
                    if let Some(&v) = logits.get(idx)
                        && v.is_finite()
                    {
                        f(idx as u32, v);
                    }
                    left &= left - 1;
                }
            }
        }
    }
}

/// Lowest-index argmax over allowed tokens.
pub(crate) fn argmax(logits: &[f32], allowed: Option<&Allowed>) -> u32 {
    let mut best_id = 0u32;
    let mut best = f32::NEG_INFINITY;
    let mut found = false;
    each_allowed(logits, allowed, |id, v| {
        if !found || v > best || (v == best && id < best_id) {
            best = v;
            best_id = id;
            found = true;
        }
    });
    if !found && allowed.is_some() {
        return argmax(logits, None);
    }
    best_id
}

/// Orders candidates by descending logit, then ascending id.
fn by_logit(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

/// Applies top-k and top-p truncation in place.
fn filter_top(cands: &mut Vec<(u32, f32)>, top_k: usize, top_p: f32, max: f32, temp: f32) {
    if top_k > 0 && cands.len() > top_k {
        cands.select_nth_unstable_by(top_k - 1, by_logit);
        cands.truncate(top_k);
    }
    if top_p < 1.0 && cands.len() > 1 {
        cands.sort_unstable_by(by_logit);
        let total: f32 = cands.iter().map(|&(_, v)| ((v - max) / temp).exp()).sum();
        let limit = top_p * total;
        let mut acc = 0.0f32;
        let mut keep = cands.len();
        for (i, &(_, v)) in cands.iter().enumerate() {
            acc += ((v - max) / temp).exp();
            if acc >= limit {
                keep = i + 1;
                break;
            }
        }
        cands.truncate(keep.max(1));
    }
}

/// Fresh entropy seed from time and a monotonic counter.
fn fresh_seed() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos
        ^ NEXT
            .fetch_add(1, AtomicOrdering::Relaxed)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
}
