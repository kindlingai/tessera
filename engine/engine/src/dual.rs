//! Online dual-drafter selection combining DFlash and native MTP.

use crate::mtp_draft::{MtpDrafter, draft_mtp, flush_mtp};
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{SpecReq, TapsBuf, TapsIn, run_speculate, spec_entry};
use crate::speculate::{Drafter, SpecStats};
use crate::{Engine, Error, Event};
use switch::{DWELL, SWITCHES};
use tessera_engine_comm::{Reader, Writer};

mod switch;

#[derive(Clone, Debug, PartialEq)]
struct PendingProbe {
    drafts: Vec<u32>,
    lp: Option<Vec<f32>>,
    matched: usize,
}

/// Deterministic per-round selector between DFlash and MTP drafters.
#[derive(Clone, Debug, PartialEq)]
pub struct DualSelector {
    pub dflash: SpecLen,
    pub mtp: SpecLen,
    pub dwell: usize,
    pub margin: f64,
    pub probe_every: usize,
    /// The drafter choice, by wire index.
    switch: usize,
    use_mtp: bool,
    since: usize,
    mtp_streak: usize,
    dflash_streak: usize,
    probe: Option<PendingProbe>,
}

impl DualSelector {
    /// Creates a selector with default dwell, margin and probe cadence.
    pub fn new(dflash: SpecLen, mtp: SpecLen) -> Self {
        Self {
            dflash,
            mtp,
            dwell: 8,
            margin: 0.15,
            probe_every: 4,
            switch: DWELL,
            use_mtp: false,
            since: 0,
            mtp_streak: 0,
            dflash_streak: 0,
            probe: None,
        }
    }

    /// Sets minimum rounds on one drafter before switching.
    pub fn with_dwell(mut self, dwell: usize) -> Self {
        self.dwell = dwell;
        self
    }

    /// Sets relative throughput gain required to switch drafters.
    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Sets rounds between cross-drafter probe launches.
    pub fn with_probe(mut self, probe_every: usize) -> Self {
        self.probe_every = probe_every;
        self
    }

    /// Returns a fresh selector for a new request.
    pub fn fresh(&self) -> Self {
        let mut fresh = Self::new(self.dflash.fresh(), self.mtp.fresh())
            .with_dwell(self.dwell)
            .with_margin(self.margin)
            .with_probe(self.probe_every);
        fresh.switch = self.switch;
        fresh
    }

    /// Whether the selector currently favors MTP over DFlash.
    pub fn use_mtp(&self) -> bool {
        self.use_mtp
    }

    /// Expected tokens per ms for `(mtp, dflash)`.
    pub fn rates(&self) -> (f64, f64) {
        let rm = self.mtp.best_rate(None, self.mtp.max_k(), self.mtp.max_k());
        let rd = self
            .dflash
            .best_rate(None, self.dflash.max_k(), self.dflash.max_k());
        (rm, rd)
    }

    /// Picks `(use_mtp, switched)` for the next speculative round.
    pub fn decide_next(&mut self, can_dflash: bool) -> (bool, bool) {
        if !can_dflash {
            let switched = !self.use_mtp;
            if switched {
                self.use_mtp = true;
                self.since = 0;
                self.probe = None;
            }
            self.since += 1;
            return (true, switched);
        }
        let want_mtp = SWITCHES[self.switch].want_mtp(self);
        let switched = want_mtp != self.use_mtp;
        if switched {
            self.use_mtp = want_mtp;
            self.since = 0;
            if !want_mtp {
                self.probe = None;
            }
        }
        self.since += 1;
        (self.use_mtp, switched)
    }

    /// Whether this MTP round should run a DFlash probe.
    pub fn should_probe(&mut self, can_dflash: bool) -> bool {
        if !can_dflash || self.probe_every == 0 || self.probe.is_some() {
            return false;
        }
        self.mtp_streak += 1;
        self.mtp_streak.is_multiple_of(self.probe_every)
    }

    /// Whether this DFlash round should catch up and score MTP.
    pub fn should_probe_mtp(&mut self) -> bool {
        if self.probe_every == 0 {
            return false;
        }
        self.dflash_streak += 1;
        self.dflash_streak.is_multiple_of(self.probe_every)
    }

    /// Records DFlash probe drafts and log-probs for upcoming commits.
    pub fn start_probe(&mut self, drafts: Vec<u32>, lp: Option<Vec<f32>>) {
        self.probe = Some(PendingProbe {
            drafts,
            lp,
            matched: 0,
        });
    }

    /// Updates estimates after a DFlash round with optional log-probs.
    pub fn observe_dflash(
        &mut self,
        drafted: usize,
        accepted: usize,
        lp: Option<&[f32]>,
        mtp_d1: Option<u32>,
        first_post: u32,
    ) {
        self.dflash.observe(drafted, accepted, lp);
        if let Some(d1) = mtp_d1 {
            self.mtp.observe(1, usize::from(d1 == first_post), None);
        }
    }

    /// Updates MTP estimates and scores any active DFlash probe.
    pub fn observe_mtp(&mut self, drafted: usize, accepted: usize, committed: &[u32]) {
        self.mtp.observe(drafted, accepted, None);
        self.feed_probe(committed);
    }

    /// Encodes static selector settings for rank broadcast.
    pub(crate) fn encode(&self, w: &mut Writer) {
        self.dflash.encode(w);
        self.mtp.encode(w);
        w.u32(self.dwell as u32)
            .f64(self.margin)
            .u32(self.probe_every as u32)
            .u32(self.switch as u32);
    }

    /// Decodes static selector settings sent by rank 0.
    pub(crate) fn decode(r: &mut Reader) -> Option<Self> {
        let dflash = SpecLen::decode(r)?;
        let mtp = SpecLen::decode(r)?;
        let dwell = r.u32()? as usize;
        let margin = r.f64()?;
        let probe_every = r.u32()? as usize;
        let switch = r.u32()? as usize;
        SWITCHES.get(switch)?;
        let mut selector = Self::new(dflash, mtp)
            .with_dwell(dwell)
            .with_margin(margin)
            .with_probe(probe_every);
        selector.switch = switch;
        Some(selector)
    }

    fn feed_probe(&mut self, committed: &[u32]) {
        let Some(mut p) = self.probe.take() else {
            return;
        };
        for &tok in committed {
            if p.matched < p.drafts.len() && p.drafts[p.matched] == tok {
                p.matched += 1;
            } else {
                let k_obs = (p.matched + 1).min(p.drafts.len());
                self.dflash.observe(k_obs, p.matched, p.lp.as_deref());
                return;
            }
        }
        if p.matched >= p.drafts.len() {
            self.dflash
                .observe(p.drafts.len(), p.matched, p.lp.as_deref());
        } else {
            self.probe = Some(p);
        }
    }
}

pub(crate) fn split_taps(
    taps: &[f32],
    rows: usize,
    d_width: usize,
    m_width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = d_width + m_width;
    let mut d_out = Vec::with_capacity(rows * d_width);
    let mut m_out = Vec::with_capacity(rows * m_width);
    for r in 0..rows {
        let base = r * total;
        d_out.extend_from_slice(&taps[base..base + d_width]);
        m_out.extend_from_slice(&taps[base + d_width..base + total]);
    }
    (d_out, m_out)
}

pub(crate) fn run_dflash(
    dflash: &mut Drafter,
    anchor: u32,
    start: usize,
    ctx_pos: usize,
    pending: &[f32],
) -> Result<(Vec<u32>, Option<Vec<f32>>), Error> {
    let wants = dflash.draft_wants();
    let outputs = dflash.run_taps(0, anchor, start, ctx_pos, TapsIn::Host(pending), wants)?;
    dflash.read_draft(&outputs)
}

/// Step limits and position for one dual-drafter round.
pub(crate) struct DualStep {
    pub slot: u32,
    pub start: usize,
    pub anchor: u32,
    pub room: usize,
    pub depth: usize,
    pub max_ctx: usize,
    pub chunk: usize,
}

/// Shared per-sequence state for dual DFlash and MTP decoding.
pub(crate) struct DualState {
    pub(crate) selector: DualSelector,
    /// Target position mapped to DFlash KV position zero.
    pub(crate) d_base: usize,
    pub(crate) d_ctx_pos: usize,
    pub(crate) d_pending: Vec<f32>,
    /// Next target position not yet written into MTP KV.
    pub(crate) m_start: u32,
    pub(crate) m_tokens: Vec<u32>,
    pub(crate) m_taps: Vec<f32>,
    pub(crate) round_mtp: bool,
    pub(crate) last_lp: Option<Vec<f32>>,
    pub(crate) last_mtp_d1: Option<u32>,
    pub(crate) switched: bool,
}

impl DualState {
    pub(crate) fn new(
        selector: DualSelector,
        (d_base, d_ctx_pos, d_pending): (usize, usize, Vec<f32>),
        (m_start, m_tokens, m_taps): (u32, Vec<u32>, Vec<f32>),
    ) -> Self {
        Self {
            selector,
            d_base,
            d_ctx_pos,
            d_pending,
            m_start,
            m_tokens,
            m_taps,
            round_mtp: false,
            last_lp: None,
            last_mtp_d1: None,
            switched: false,
        }
    }

    fn take_mtp(&mut self, start: usize) -> (u32, Vec<u32>, Vec<f32>) {
        let ms = self.m_start;
        self.m_start = start as u32;
        (
            ms,
            std::mem::take(&mut self.m_tokens),
            std::mem::take(&mut self.m_taps),
        )
    }

    fn step_dflash(
        &mut self,
        dflash: &mut Drafter,
        anchor: u32,
        d_start: usize,
    ) -> Result<(Vec<u32>, Option<Vec<f32>>), Error> {
        let out = run_dflash(dflash, anchor, d_start, self.d_ctx_pos, &self.d_pending)?;
        self.d_ctx_pos = d_start;
        self.d_pending.clear();
        Ok(out)
    }

    /// Proposes next drafts and returns any updated MTP cache watermark.
    pub(crate) fn draft(
        &mut self,
        target: &mut Engine,
        dflash: &mut Drafter,
        mtp: &mut MtpDrafter,
        step: DualStep,
    ) -> Result<(Vec<u32>, Option<usize>), Error> {
        let block = dflash.block();
        let d_start = step.start - self.d_base;
        let can_dflash = step.start + block <= step.max_ctx && d_start + block <= dflash.context();
        let (use_mtp, switched) = self.selector.decide_next(can_dflash);
        self.round_mtp = use_mtp;
        self.switched = switched;
        self.last_mtp_d1 = None;
        if !use_mtp {
            let mut mtp_valid = None;
            if self.selector.should_probe_mtp() {
                let (ms, toks, taps) = self.take_mtp(step.start);
                self.last_mtp_d1 = flush_mtp(mtp, step.slot, ms, &toks, &taps, step.chunk)?;
                mtp_valid = Some(ms as usize + toks.len());
            }
            let (mut drafts, lp) = self.step_dflash(dflash, step.anchor, d_start)?;
            let k = self.selector.dflash.choose(
                lp.as_deref(),
                drafts.len().min(step.room),
                drafts.len(),
            );
            self.last_lp = lp;
            drafts.truncate(k);
            return Ok((drafts, mtp_valid));
        }
        if self.selector.should_probe(can_dflash) {
            let (probe, lp) = self.step_dflash(dflash, step.anchor, d_start)?;
            self.selector.start_probe(probe, lp);
        }
        let (ms, toks, taps) = self.take_mtp(step.start);
        let hidden = mtp.hidden();
        let split = toks.len().saturating_sub(step.chunk);
        if split > 0 {
            flush_mtp(
                mtp,
                step.slot,
                ms,
                &toks[..split],
                &taps[..split * hidden],
                step.chunk,
            )?;
        }
        let avail = step.depth.min(self.selector.mtp.max_k()).min(step.room);
        let tail_taps = TapsBuf::Host(taps[split * hidden..].to_vec());
        let drafts = draft_mtp(
            target,
            mtp,
            &self.selector.mtp,
            step.slot,
            ms + split as u32,
            &toks[split..],
            &tail_taps,
            step.start,
            avail,
        )?;
        Ok((drafts, Some(step.start)))
    }

    /// Records verified tokens and queues committed taps for next round.
    pub(crate) fn observe(
        &mut self,
        drafted: usize,
        accepted: usize,
        committed: &[u32],
        d_taps: &[f32],
        m_taps: &[f32],
        stats: Option<&mut SpecStats>,
    ) {
        if let Some(stats) = stats {
            if self.switched {
                stats.switches += 1;
            }
            if self.round_mtp {
                stats.mtp_steps += 1;
            } else {
                stats.dflash_steps += 1;
            }
        }
        if drafted > 0 {
            if self.round_mtp {
                self.selector.observe_mtp(drafted, accepted, committed);
            } else {
                self.selector.observe_dflash(
                    drafted,
                    accepted,
                    self.last_lp.as_deref(),
                    self.last_mtp_d1,
                    committed[0],
                );
            }
        }
        self.d_pending.extend_from_slice(d_taps);
        self.m_tokens.extend_from_slice(committed);
        self.m_taps.extend_from_slice(m_taps);
    }
}

/// Dual-drafter speculative decoding combining DFlash and native MTP.
pub struct DualSpeculate<'a> {
    target: &'a mut Engine,
    dflash: &'a mut Drafter,
    mtp: &'a mut MtpDrafter,
    depth: usize,
    selector: DualSelector,
    prefill_chunk: Option<usize>,
}

impl<'a> DualSpeculate<'a> {
    /// Pairs a dual-tapped target engine with DFlash and MTP drafters.
    pub fn new(
        target: &'a mut Engine,
        dflash: &'a mut Drafter,
        mtp: &'a mut MtpDrafter,
        depth: usize,
    ) -> Result<Self, Error> {
        if depth == 0 {
            return Err(Error::Rows);
        }
        let width = target.draft_taps()?;
        if width != dflash.tap_width() + mtp.hidden() {
            return Err(Error::Rows);
        }
        let d_len = SpecLen::fixed(dflash.block() - 1).with_cost(CostModel::dual_dflash());
        let m_len = SpecLen::fixed(depth)
            .with_cost(CostModel::dual_mtp())
            .with_mtp_seeds();
        Ok(Self {
            target,
            dflash,
            mtp,
            depth,
            selector: DualSelector::new(d_len, m_len),
            prefill_chunk: None,
        })
    }

    /// Replaces the online drafter selector configuration.
    pub fn with_selector(mut self, selector: DualSelector) -> Self {
        self.selector = selector;
        self
    }
}

spec_entry!(
    DualSpeculate<'a>,
    DualSelector,
    selector,
    |s, target, req, sel, emit| run_dual_req(target, s.dflash, s.mtp, s.depth, req, sel, emit)
);

fn run_dual_req(
    target: &mut Engine,
    dflash: &mut Drafter,
    mtp: &mut MtpDrafter,
    depth: usize,
    req: SpecReq,
    selector: DualSelector,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Result<SpecStats, Error> {
    let block = dflash.block();
    let (d_width, m_width) = (dflash.tap_width(), mtp.hidden());
    let (total, max_ctx, d_ctx) = (
        d_width + m_width,
        target.context().min(mtp.context()),
        dflash.context(),
    );
    let win = dflash.sliding_window();
    let drop_before = win.map_or(0, |w| req.request.tokens.len().saturating_sub(w));
    let mut pair = (dflash, mtp);
    run_speculate(
        target,
        &mut pair,
        req,
        emit,
        |target, (dflash, mtp), pos, prompt, len, taps| {
            let raw = taps.into_host(&*target.runner, total)?;
            let (d_taps, m_taps) = split_taps(&raw, len, d_width, m_width);
            let end = pos + len;
            if end > drop_before && (prompt.len() - drop_before) + block <= d_ctx {
                let skip = drop_before.saturating_sub(pos);
                let (d_end, d_pos) = (end - drop_before, pos + skip - drop_before);
                let (in_t, w) = (
                    TapsIn::Host(&d_taps[skip * d_width..]),
                    dflash.draft_wants(),
                );
                dflash.run_taps(0, prompt[end], d_end, d_pos, in_t, w)?;
            }
            if prompt.len() <= max_ctx {
                let shifted = &prompt[pos + 1..end + 1];
                let wants = mtp.wants();
                mtp.run_taps(0, pos as u32, shifted, TapsIn::Host(&m_taps), wants, true)?;
            }
            Ok(())
        },
        |target, _, pos, prompt, anchor, taps| {
            let len = prompt.len() - pos;
            let raw = taps.into_host(&*target.runner, total)?;
            let (d_all, m_taps) = split_taps(&raw, len, d_width, m_width);
            let skip = drop_before.saturating_sub(pos);
            let (d_pending, d_ctx_pos) =
                (d_all[skip * d_width..].to_vec(), pos + skip - drop_before);
            let m_tokens: Vec<u32> = prompt[pos + 1..].iter().copied().chain([anchor]).collect();
            Ok(DualState::new(
                selector,
                (drop_before, d_ctx_pos, d_pending),
                (pos as u32, m_tokens, m_taps),
            ))
        },
        |target, (dflash, mtp), st, start, anchor, remaining| {
            let room = remaining
                .saturating_sub(1)
                .min(max_ctx.saturating_sub(start + 1));
            if room == 0 {
                return Ok(None);
            }
            let step = DualStep {
                slot: 0,
                start,
                anchor,
                room,
                depth,
                max_ctx,
                chunk: usize::MAX,
            };
            let (drafts, _) = st.draft(target, dflash, mtp, step)?;
            Ok(Some(drafts))
        },
        |target, _, st, _start, drafted, accepted, posterior, block_taps, stats| {
            let commit = 1 + accepted;
            let raw = block_taps.into_host(&*target.runner, total)?;
            let (d_taps, m_taps) = split_taps(&raw, commit, d_width, m_width);
            st.observe(
                drafted,
                accepted,
                &posterior[..commit],
                &d_taps,
                &m_taps,
                Some(stats),
            );
            Ok(())
        },
    )
}
