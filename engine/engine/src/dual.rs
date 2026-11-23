//! Online dual-drafter selection combining DFlash and native MTP.

use crate::mtp_draft::{MtpDrafter, draft_mtp, run_mtp};
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{TapsBuf, TapsIn, run_speculate, spec_entry};
use crate::speculate::{Drafter, SpecStats};
use crate::{Engine, Error, Event, Request};
use tessera_engine_comm::{Reader, Writer};

#[derive(Clone, Debug, PartialEq)]
struct PendingProbe {
    drafts: Vec<u32>,
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
        Self {
            dflash: self.dflash.fresh(),
            mtp: self.mtp.fresh(),
            dwell: self.dwell,
            margin: self.margin,
            probe_every: self.probe_every,
            use_mtp: false,
            since: 0,
            mtp_streak: 0,
            dflash_streak: 0,
            probe: None,
        }
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
        let want_mtp = if self.since < self.dwell {
            self.use_mtp
        } else {
            let (rm, rd) = self.rates();
            if self.use_mtp {
                rm * (1.0 + self.margin) >= rd
            } else {
                rm > rd * (1.0 + self.margin)
            }
        };
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

    /// Records DFlash probe drafts to score against upcoming commits.
    pub fn start_probe(&mut self, drafts: Vec<u32>) {
        self.probe = Some(PendingProbe { drafts, matched: 0 });
    }

    /// Updates estimates after a DFlash round and optional MTP check.
    pub fn observe_dflash(
        &mut self,
        drafted: usize,
        accepted: usize,
        pred: f64,
        mtp_d1: Option<u32>,
        first_post: u32,
    ) {
        self.dflash.observe(drafted, accepted, None, pred);
        if let Some(d1) = mtp_d1 {
            self.mtp
                .observe(1, usize::from(d1 == first_post), None, 0.0);
        }
    }

    /// Updates MTP estimates and scores any active DFlash probe.
    pub fn observe_mtp(&mut self, drafted: usize, accepted: usize, pred: f64, committed: &[u32]) {
        self.mtp.observe(drafted, accepted, None, pred);
        self.feed_probe(committed);
    }

    /// Encodes static selector settings for rank broadcast.
    pub(crate) fn encode(&self, w: &mut Writer) {
        self.dflash.encode(w);
        self.mtp.encode(w);
        w.u32(self.dwell as u32)
            .f64(self.margin)
            .u32(self.probe_every as u32);
    }

    /// Decodes static selector settings sent by rank 0.
    pub(crate) fn decode(r: &mut Reader) -> Option<Self> {
        let dflash = SpecLen::decode(r)?;
        let mtp = SpecLen::decode(r)?;
        let dwell = r.u32()? as usize;
        let margin = r.f64()?;
        let probe_every = r.u32()? as usize;
        Some(
            Self::new(dflash, mtp)
                .with_dwell(dwell)
                .with_margin(margin)
                .with_probe(probe_every),
        )
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
                self.dflash.observe(k_obs, p.matched, None, 0.0);
                return;
            }
        }
        if p.matched >= p.drafts.len() {
            self.dflash.observe(p.drafts.len(), p.matched, None, 0.0);
        } else {
            self.probe = Some(p);
        }
    }
}

fn split_taps(taps: &[f32], rows: usize, d_width: usize, m_width: usize) -> (Vec<f32>, Vec<f32>) {
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

fn run_dflash(
    dflash: &mut Drafter,
    anchor: u32,
    start: usize,
    ctx_pos: usize,
    pending: &[f32],
) -> Result<Vec<u32>, Error> {
    let wants = dflash.draft_wants();
    let outputs = dflash.run_taps(0, anchor, start, ctx_pos, TapsIn::Host(pending), wants)?;
    dflash.read_draft(&outputs)
}

struct DualState {
    selector: DualSelector,
    d_ctx_pos: usize,
    d_pending: Vec<f32>,
    m_start: u32,
    m_tokens: Vec<u32>,
    m_taps: Vec<f32>,
    round_mtp: bool,
    last_pred: f64,
    last_mtp_d1: Option<u32>,
    switched: bool,
}

/// Dual-drafter speculative decoding combining DFlash and native MTP.
pub struct DualSpeculate<'a> {
    target: &'a mut Engine,
    dflash: &'a mut Drafter,
    mtp: &'a mut MtpDrafter,
    depth: usize,
    selector: DualSelector,
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
        let Some(width) = target.tap_width() else {
            return Err(Error::Taps);
        };
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
    request: Request,
    selector: DualSelector,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Result<SpecStats, Error> {
    let block = dflash.block();
    let d_width = dflash.tap_width();
    let m_width = mtp.hidden();
    let total = d_width + m_width;
    let max_ctx = target.context().min(dflash.context()).min(mtp.context());
    run_speculate(
        target,
        request,
        emit,
        |target, prompt, anchor, taps| {
            let raw = taps.into_host(&*target.runner, total)?;
            let (d_pending, m_taps) = split_taps(&raw, prompt.len(), d_width, m_width);
            let m_tokens: Vec<u32> = prompt[1..].iter().copied().chain([anchor]).collect();
            Ok(DualState {
                selector,
                d_ctx_pos: 0,
                d_pending,
                m_start: 0,
                m_tokens,
                m_taps,
                round_mtp: false,
                last_pred: 1.0,
                last_mtp_d1: None,
                switched: false,
            })
        },
        |target, st, start, anchor, remaining| {
            let rem_cap = remaining.saturating_sub(1);
            let ctx_cap = max_ctx.saturating_sub(start + 1);
            if rem_cap == 0 || ctx_cap == 0 {
                return Ok(None);
            }
            let can_dflash = start + block <= max_ctx;
            let (use_mtp, switched) = st.selector.decide_next(can_dflash);
            st.round_mtp = use_mtp;
            st.switched = switched;
            st.last_mtp_d1 = None;
            if !use_mtp {
                if st.selector.should_probe_mtp() {
                    let (mtp_d1, _) = run_mtp(
                        mtp,
                        st.m_start,
                        &st.m_tokens,
                        TapsIn::Host(&st.m_taps),
                        false,
                    )?;
                    st.m_start = start as u32;
                    st.m_tokens.clear();
                    st.m_taps.clear();
                    st.last_mtp_d1 = Some(mtp_d1);
                }
                let mut drafts = run_dflash(dflash, anchor, start, st.d_ctx_pos, &st.d_pending)?;
                st.d_ctx_pos = start;
                st.d_pending.clear();
                let avail = drafts.len().min(rem_cap).min(ctx_cap);
                let (k, pred) = st.selector.dflash.choose(None, avail, drafts.len());
                st.last_pred = pred;
                drafts.truncate(k);
                return Ok(Some(drafts));
            }
            if st.selector.should_probe(can_dflash) {
                let probe = run_dflash(dflash, anchor, start, st.d_ctx_pos, &st.d_pending)?;
                st.d_ctx_pos = start;
                st.d_pending.clear();
                st.selector.start_probe(probe);
            }
            let avail = depth.min(st.selector.mtp.max_k()).min(rem_cap).min(ctx_cap);
            let m_taps = TapsBuf::Host(std::mem::take(&mut st.m_taps));
            let (drafts, pred) = draft_mtp(
                target,
                mtp,
                &st.selector.mtp,
                st.m_start,
                &st.m_tokens,
                &m_taps,
                start,
                avail,
            )?;
            st.m_start = start as u32;
            st.m_tokens.clear();
            st.last_pred = pred;
            Ok(Some(drafts))
        },
        |target, st, _start, drafted, accepted, posterior, block_taps, stats| {
            let commit = 1 + accepted;
            if st.switched {
                stats.switches += 1;
            }
            if st.round_mtp {
                stats.mtp_steps += 1;
                st.selector
                    .observe_mtp(drafted, accepted, st.last_pred, &posterior[..commit]);
            } else {
                stats.dflash_steps += 1;
                st.selector.observe_dflash(
                    drafted,
                    accepted,
                    st.last_pred,
                    st.last_mtp_d1,
                    posterior[0],
                );
            }
            let raw = block_taps.into_host(&*target.runner, total)?;
            let (d_taps, m_taps) = split_taps(&raw, commit, d_width, m_width);
            st.d_pending.extend_from_slice(&d_taps);
            st.m_tokens.extend_from_slice(&posterior[..commit]);
            st.m_taps.extend_from_slice(&m_taps);
            Ok(())
        },
    )
}
