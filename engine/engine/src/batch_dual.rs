//! Multi-slot DFlash decoding and solo DFlash/MTP promotion.

use crate::batch_spec::{
    ActiveSeq, Admitted, BatchMtp, DEFAULT_CHUNK, DraftJob, RoundHost, drive_follow, drive_step,
    queue_request, retire_from,
};
use crate::dual::{DualSelector, run_dflash, split_taps};
use crate::kept::Kept;
use crate::mtp_draft::MtpDrafter;
use crate::schedule::Sink;
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{TapsBuf, TapsIn, fnv_u32, span};
use crate::speculate::Drafter;
use crate::{Engine, Error, Request, Row};
use std::collections::VecDeque;

/// Context rows retained per slot for DFlash tail promotion.
pub(crate) const DFLASH_TAIL_WIN: usize = 128;

/// Minimum remaining tokens to promote a sole survivor to DFlash.
pub(crate) const PROMOTE_MIN_REM: usize = 8;

/// Optional DFlash peer for solo and tail acceleration.
pub(crate) struct DualPeer<'a> {
    pub(crate) dflash: &'a mut Drafter,
    pub(crate) selector: DualSelector,
    pub(crate) d_width: usize,
    pub(crate) m_width: usize,
    pub(crate) block: usize,
}

/// Per-sequence state while decoding solo with DFlash and MTP.
pub(crate) struct SoloState {
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
    pub(crate) last_pred: f64,
    pub(crate) last_lp: Option<Vec<f32>>,
    pub(crate) last_mtp_d1: Option<u32>,
}

/// Appends `src` rows to `buf`, keeping at most `DFLASH_TAIL_WIN` rows.
pub(crate) fn push_tail(buf: &mut Vec<f32>, src: &[f32], width: usize) {
    if src.is_empty() || width == 0 {
        return;
    }
    let cap = DFLASH_TAIL_WIN * width;
    if src.len() >= cap {
        buf.clear();
        buf.extend_from_slice(&src[src.len() - cap..]);
        return;
    }
    buf.extend_from_slice(src);
    if buf.len() > cap {
        let drop_n = buf.len() - cap;
        buf.drain(..drop_n);
    }
}

impl<'a> BatchMtp<'a> {
    /// Pairs a dual-tapped target with DFlash and MTP drafters.
    pub fn new_dual(
        target: &'a mut Engine,
        dflash: &'a mut Drafter,
        drafter: &'a mut MtpDrafter,
        depth: usize,
        selector: DualSelector,
    ) -> Result<Self, Error> {
        if depth == 0 {
            return Err(Error::Rows);
        }
        let width = target.draft_taps()?;
        let d_width = dflash.tap_width();
        let m_width = drafter.hidden();
        if width != d_width + m_width {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        let max_ctx = target
            .context()
            .min(drafter.context())
            .min(dflash.context());
        let block = dflash.block();
        let free = (0..slots as u32).rev().collect();
        Ok(Self {
            target,
            drafter,
            dual: Some(DualPeer {
                dflash,
                selector,
                d_width,
                m_width,
                block,
            }),
            depth,
            prefill_chunk: crate::batch_spec::DEFAULT_CHUNK,
            max_ctx,
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            active: Vec::new(),
            free,
            hash: 2166136261,
        })
    }

    /// Flushes deferred MTP rows and switches to shared MTP.
    pub(crate) fn demote_solo(&mut self) -> Result<(), Error> {
        let Some(solo) = self.active[0].solo.take() else {
            return Ok(());
        };
        let slot = self.active[0].slot;
        self.flush_mtp(slot, solo.m_start, &solo.m_tokens, &solo.m_taps)?;
        self.active[0].drafts.truncate(self.depth);
        if let Some(peer) = &self.dual {
            push_tail(&mut self.active[0].d_tail, &solo.d_pending, peer.d_width);
        }
        Ok(())
    }

    /// Flushes deferred MTP prefill rows when another request joins.
    pub(crate) fn demote_prefill(&mut self) -> Result<(), Error> {
        let Some(p) = self.prefilling.front_mut() else {
            return Ok(());
        };
        if !p.solo_mode {
            return Ok(());
        }
        p.solo_mode = false;
        let slot = p.slot;
        let m_toks = std::mem::take(&mut p.m_tok_buf);
        let m_taps = std::mem::take(&mut p.m_tap_buf);
        if let Some(peer) = &self.dual {
            let raw = std::mem::take(&mut p.d_buf);
            push_tail(&mut p.d_buf, &raw, peer.d_width);
        }
        self.flush_mtp(slot, 0, &m_toks, &m_taps)?;
        Ok(())
    }

    /// Feeds deferred `(tokens, taps)` chunks into `MtpDrafter` for `slot`.
    pub(crate) fn flush_mtp(
        &mut self,
        slot: u32,
        mut pos0: u32,
        tokens: &[u32],
        taps: &[f32],
    ) -> Result<Option<u32>, Error> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let m_width = self.drafter.hidden();
        let wants = self.drafter.wants();
        let mut off = 0usize;
        let mut last_tok = None;
        while off < tokens.len() {
            let len = (tokens.len() - off).min(self.prefill_chunk);
            let h = &taps[off * m_width..(off + len) * m_width];
            let outs = self.drafter.run_taps(
                slot,
                pos0,
                &tokens[off..off + len],
                TapsIn::Host(h),
                wants,
                true,
            )?;
            if off + len == tokens.len() {
                let ids = self.drafter.ids(outs[0])?;
                last_tok = Some(ids[len - 1]);
            }
            pos0 += len as u32;
            off += len;
        }
        Ok(last_tok)
    }

    /// Splits target taps into `(d_taps, m_taps)` for dual mode.
    pub(crate) fn split_dual_taps(
        &self,
        taps: TapsBuf,
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let peer = self.dual.as_ref().expect("dual peer");
        let total = peer.d_width + peer.m_width;
        let raw = taps.into_host(&*self.target.runner, total)?;
        Ok(split_taps(&raw, rows, peer.d_width, peer.m_width))
    }

    /// Reads DFlash taps for `rows` without consuming device `taps`.
    pub(crate) fn read_dual_d_taps(&self, taps: &TapsBuf, rows: usize) -> Result<Vec<f32>, Error> {
        let peer = self.dual.as_ref().expect("dual peer");
        let total = peer.d_width + peer.m_width;
        let d_width = peer.d_width;
        let mut out = Vec::with_capacity(rows * d_width);
        match taps {
            TapsBuf::Host(v) => {
                for r in 0..rows {
                    let base = r * total;
                    out.extend_from_slice(&v[base..base + d_width]);
                }
            }
            TapsBuf::Device { slot, .. } => {
                let raw = self.target.runner.read(*slot)?;
                let floats = raw.floats().map_err(tessera_run::Error::from)?;
                for r in 0..rows {
                    let base = r * total;
                    out.extend_from_slice(&floats[base..base + d_width]);
                }
            }
        }
        Ok(out)
    }

    /// Promotes a lone surviving stream into solo DFlash mode.
    pub(crate) fn promote_solo(&mut self, job: DraftJob, m_all: &[f32]) -> Result<(), Error> {
        let m_width = self.drafter.hidden();
        let (d_w, selector) = {
            let peer = self.dual.as_ref().expect("dual peer");
            (peer.d_width, peer.selector.fresh())
        };
        let commit = job.committed.len();
        let m_lo = job.tap_row_off * m_width;
        let m_taps = m_all[m_lo..m_lo + commit * m_width].to_vec();
        let seq = &mut self.active[job.active_idx];
        let d_pending = std::mem::take(&mut seq.d_tail);
        let w_rows = d_pending.len() / d_w;
        seq.solo = Some(SoloState {
            selector,
            d_base: seq.start - w_rows,
            d_ctx_pos: 0,
            d_pending,
            m_start: job.old_start,
            m_tokens: job.committed,
            m_taps,
            round_mtp: false,
            last_pred: 1.0,
            last_lp: None,
            last_mtp_d1: None,
        });
        self.solo_draft(job.active_idx)
    }

    /// Verifies and re-drafts one solo sequence using DFlash or MTP.
    pub(crate) fn verify_solo(&mut self) -> Result<(), Error> {
        let seq = &self.active[0];
        let k = seq.drafts.len().min(seq.room(self.max_ctx));
        let mut rows = Vec::with_capacity(1 + k);
        seq.verify(&mut rows, k);
        let (ids, taps) = self.target.step_taps_local(&rows)?;
        let (d_all, m_all) = self.split_dual_taps(taps, rows.len())?;
        let d_width = self.dual.as_ref().expect("dual peer").d_width;
        let m_width = self.drafter.hidden();
        let ranked = self.target.control.is_some();
        let (accepted, committed, done) =
            self.active[0].apply_verify(k, &ids, self.max_ctx, &mut self.hash);
        let commit = committed.len();
        if done.is_some() || (!ranked && self.active[0].cancelled) {
            let seq = self.active.remove(0);
            self.free.push(seq.slot);
            return Ok(());
        }
        let seq = &mut self.active[0];
        let solo = seq.solo.as_mut().expect("solo state");
        if k > 0 {
            if solo.round_mtp {
                solo.selector
                    .observe_mtp(k, accepted, solo.last_pred, &committed);
            } else {
                solo.selector.observe_dflash_lp(
                    k,
                    accepted,
                    solo.last_pred,
                    solo.last_lp.as_deref(),
                    solo.last_mtp_d1,
                    ids[0],
                );
            }
        }
        solo.d_pending.extend_from_slice(&d_all[..commit * d_width]);
        solo.m_tokens.extend_from_slice(&committed);
        solo.m_taps.extend_from_slice(&m_all[..commit * m_width]);
        self.solo_draft(0)
    }

    /// Proposes next drafts for solo sequence `idx` via `DualSelector`.
    pub(crate) fn solo_draft(&mut self, idx: usize) -> Result<(), Error> {
        let block = self.dual.as_ref().expect("dual peer").block;
        let room = self.active[idx].room(self.max_ctx);
        if room == 0 {
            self.active[idx].drafts.clear();
            return Ok(());
        }
        let slot = self.active[idx].slot;
        let start = self.active[idx].start;
        let anchor = self.active[idx].anchor;
        let d_start = start - self.active[idx].solo.as_ref().expect("solo").d_base;
        let can_dflash = start + block <= self.max_ctx && d_start + block <= self.max_ctx;
        let (use_mtp, probe_mtp, probe_dflash) = {
            let solo = self.active[idx].solo.as_mut().expect("solo");
            let (use_mtp, _) = solo.selector.decide_next(can_dflash);
            solo.round_mtp = use_mtp;
            solo.last_mtp_d1 = None;
            if !use_mtp {
                (false, solo.selector.should_probe_mtp(), false)
            } else {
                (true, false, solo.selector.should_probe(can_dflash))
            }
        };
        if !use_mtp {
            if probe_mtp {
                let (m_start, m_toks, m_taps) = {
                    let solo = self.active[idx].solo.as_mut().expect("solo");
                    let ms = solo.m_start;
                    solo.m_start = start as u32;
                    (
                        ms,
                        std::mem::take(&mut solo.m_tokens),
                        std::mem::take(&mut solo.m_taps),
                    )
                };
                let d1 = self.flush_mtp(slot, m_start, &m_toks, &m_taps)?;
                self.active[idx].solo.as_mut().expect("solo").last_mtp_d1 = d1;
            }
            let (d_ctx_pos, d_pending) = {
                let solo = self.active[idx].solo.as_mut().expect("solo");
                let cp = solo.d_ctx_pos;
                solo.d_ctx_pos = d_start;
                (cp, std::mem::take(&mut solo.d_pending))
            };
            let dflash = &mut self.dual.as_mut().expect("dual peer").dflash;
            let (mut drafts, lp) = run_dflash(dflash, anchor, d_start, d_ctx_pos, &d_pending)?;
            let avail = drafts.len().min(room);
            let (k, pred) = self.active[idx]
                .solo
                .as_ref()
                .expect("solo")
                .selector
                .dflash
                .choose(lp.as_deref(), avail, drafts.len());
            let solo = self.active[idx].solo.as_mut().expect("solo");
            solo.last_pred = pred;
            solo.last_lp = lp;
            drafts.truncate(k);
            self.active[idx].drafts = drafts;
            return Ok(());
        }
        if probe_dflash {
            let (d_ctx_pos, d_pending) = {
                let solo = self.active[idx].solo.as_mut().expect("solo");
                let cp = solo.d_ctx_pos;
                solo.d_ctx_pos = d_start;
                (cp, std::mem::take(&mut solo.d_pending))
            };
            let dflash = &mut self.dual.as_mut().expect("dual peer").dflash;
            let (probe, lp) = run_dflash(dflash, anchor, d_start, d_ctx_pos, &d_pending)?;
            self.active[idx]
                .solo
                .as_mut()
                .expect("solo")
                .selector
                .start_probe_lp(probe, lp);
        }
        let (m_start, m_toks, m_taps, avail) = {
            let solo = self.active[idx].solo.as_mut().expect("solo");
            let ms = solo.m_start;
            solo.m_start = start as u32;
            let avail = self.depth.min(solo.selector.mtp.max_k()).min(room);
            (
                ms,
                std::mem::take(&mut solo.m_tokens),
                std::mem::take(&mut solo.m_taps),
                avail,
            )
        };
        let m_width = self.drafter.hidden();
        let split = m_toks.len().saturating_sub(self.prefill_chunk);
        if split > 0 {
            self.flush_mtp(slot, m_start, &m_toks[..split], &m_taps[..split * m_width])?;
        }
        let job = DraftJob {
            active_idx: idx,
            slot,
            old_start: m_start + split as u32,
            new_start: start,
            committed: m_toks[split..].to_vec(),
            tap_row_off: 0,
            avail,
        };
        let tail_taps = TapsBuf::Host(m_taps[split * m_width..].to_vec());
        let mut drafts = self
            .draft_batch(&[job], &tail_taps, false)?
            .pop()
            .unwrap_or_default();
        let (k, pred) = self.active[idx]
            .solo
            .as_ref()
            .expect("solo")
            .selector
            .mtp
            .choose(None, drafts.len(), drafts.len());
        self.active[idx].solo.as_mut().expect("solo").last_pred = pred;
        drafts.truncate(k);
        self.active[idx].drafts = drafts;
        Ok(())
    }
}

/// Headroom rows reserved before admitting another paged request.
const PAGE_HEADROOM: usize = 2048;

struct DFlashPrefill {
    slot: u32,
    request: Request,
    sink: Option<Sink>,
    pos: usize,
}

struct DFlashSeq {
    seq: ActiveSeq,
    spec: SpecLen,
    last_pred: f64,
    last_lp: Option<Vec<f32>>,
}

struct DFlashJob {
    active_idx: usize,
    slot: u32,
    old_start: usize,
    new_start: usize,
    anchor: u32,
    tap_row_off: usize,
    commit: usize,
    avail: usize,
}

/// Multi-slot continuous-batching DFlash decoder.
pub struct BatchDFlash<'a> {
    target: &'a mut Engine,
    drafter: &'a mut Drafter,
    spec_len: SpecLen,
    prefill_chunk: usize,
    max_ctx: usize,
    block: usize,
    tap_width: usize,
    kept: Option<Kept>,
    waiting: VecDeque<(Request, Sink)>,
    prefilling: VecDeque<DFlashPrefill>,
    active: Vec<DFlashSeq>,
    free: Vec<u32>,
    hash: u32,
}

impl<'a> BatchDFlash<'a> {
    /// Pairs a multi-slot tapped target engine with its DFlash drafter.
    pub fn new(target: &'a mut Engine, drafter: &'a mut Drafter) -> Result<Self, Error> {
        let width = target.draft_taps()?;
        if width != drafter.tap_width() {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        let max_ctx = target.context().min(drafter.context());
        if let Some(lanes) = &mut target.lanes {
            lanes.reset();
        }
        let kept = target
            .lanes
            .as_ref()
            .map(|l| Kept::paged(slots, max_ctx, l.pages(), l.page));
        let free = if kept.is_some() {
            Vec::new()
        } else {
            (0..slots as u32).rev().collect()
        };
        let block = drafter.block();
        let spec_len = SpecLen::fixed(block - 1).with_cost(CostModel::dflash());
        Ok(Self {
            target,
            drafter,
            spec_len,
            prefill_chunk: DEFAULT_CHUNK,
            max_ctx,
            block,
            tap_width: width,
            kept,
            waiting: VecDeque::new(),
            prefilling: VecDeque::new(),
            active: Vec::new(),
            free,
            hash: 2166136261,
        })
    }

    /// Sets the maximum prompt rows per prefill step.
    pub fn with_prefill_chunk(mut self, rows: usize) -> Self {
        self.prefill_chunk = rows.max(1);
        self
    }

    /// Runs one round of admit, prefill, verify, and draft.
    pub fn step(&mut self) -> Result<bool, Error> {
        drive_step(self)
    }

    /// Follower rank loop executing rank 0's batched rounds.
    pub fn follow(&mut self) -> Result<(), Error> {
        drive_follow(self)
    }

    /// Configures adaptive or truncated draft length selection.
    pub fn with_spec_len(mut self, spec_len: SpecLen) -> Self {
        self.spec_len = spec_len;
        self
    }

    /// Queues an answer. Unfit requests finish at once.
    pub fn submit(&mut self, request: Request, sink: Sink) {
        queue_request(&mut self.waiting, self.max_ctx, request, sink);
    }

    /// Whether any answer is waiting, prefilling, or decoding.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.prefilling.is_empty() || !self.active.is_empty()
    }

    fn free_slot(&mut self, slot: u32) {
        if let Some(kept) = &mut self.kept {
            kept.finish(slot, Vec::new());
        } else {
            self.free.push(slot);
        }
    }

    fn ensure_pages(&mut self, rows: &[Row]) -> Result<(), Error> {
        let (Some(kept), Some(lanes)) = (&mut self.kept, &mut self.target.lanes) else {
            return Ok(());
        };
        for r in rows {
            if let Some(pages) = kept.grow(r.slot, r.position as usize + 1)? {
                lanes.set(r.slot, pages);
            }
        }
        Ok(())
    }

    fn exec_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        for item in admitted {
            if follower && let Some(kept) = &mut self.kept {
                let Some((slot, _)) = kept.admit(&item.request.tokens, 0) else {
                    return Err(Error::RowPlace);
                };
                debug_assert_eq!(slot, item.slot);
            }
            self.prefilling.push_back(DFlashPrefill {
                slot: item.slot,
                request: item.request,
                sink: item.sink,
                pos: 0,
            });
        }
        if !self.prefilling.is_empty() {
            self.prefill_step()?;
        }
        self.active.sort_by_key(|s| s.seq.slot);
        if self.active.is_empty() {
            return Ok(());
        }
        self.verify_and_draft()
    }

    fn prefill_step(&mut self) -> Result<(), Error> {
        let mut seq = self.prefilling.pop_front().expect("prefill seq");
        let n = seq.request.tokens.len();
        let pos0 = seq.pos;
        let len = (n - pos0).min(self.prefill_chunk);
        let end = pos0 + len;
        let mut rows = Vec::with_capacity(len);
        span(
            &mut rows,
            seq.slot,
            pos0,
            seq.request.tokens[pos0..end].iter().copied(),
        );
        self.ensure_pages(&rows)?;
        let (ids, taps) = self.target.step_taps_local(&rows)?;
        let wants = self.drafter.draft_wants();
        if end < n {
            if n + self.block <= self.max_ctx {
                self.drafter.run_taps(
                    seq.slot,
                    seq.request.tokens[end],
                    end,
                    pos0,
                    taps.as_in(),
                    wants,
                )?;
            }
            seq.pos = end;
            self.prefilling.push_front(seq);
            return Ok(());
        }
        let anchor = *ids.last().expect("non-empty chunk");
        fnv_u32(&mut self.hash, anchor);
        let mut act = ActiveSeq {
            slot: seq.slot,
            request: seq.request,
            sink: seq.sink,
            start: n,
            anchor,
            drafts: Vec::new(),
            emitted: 0,
            cancelled: false,
            solo: None,
            d_tail: Vec::new(),
        };
        let done = act.commit(&[anchor], self.max_ctx);
        if done.is_some() || (self.target.control.is_none() && act.cancelled) {
            self.free_slot(act.slot);
            return Ok(());
        }
        let spec = self.spec_len.fresh();
        let mut last_pred = 1.0;
        let mut last_lp = None;
        let room = act.room(self.max_ctx);
        if n + self.block <= self.max_ctx && room > 0 {
            let outs = self
                .drafter
                .run_taps(act.slot, anchor, n, pos0, taps.as_in(), wants)?;
            let (mut drafts, lp) = self.drafter.read_draft(&outs)?;
            let avail = drafts.len().min(room);
            let (k, pred) = spec.choose(lp.as_deref(), avail, drafts.len());
            drafts.truncate(k);
            act.drafts = drafts;
            last_pred = pred;
            last_lp = lp;
        }
        self.active.push(DFlashSeq {
            seq: act,
            spec,
            last_pred,
            last_lp,
        });
        Ok(())
    }

    fn verify_and_draft(&mut self) -> Result<(), Error> {
        let s_count = self.active.len();
        let mut rows = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(s_count);
        for item in &self.active {
            let k = item.seq.drafts.len().min(item.seq.room(self.max_ctx));
            spans.push((rows.len(), k));
            item.seq.verify(&mut rows, k);
        }
        self.ensure_pages(&rows)?;
        let (ids, raw_taps) = self.target.step_taps_local(&rows)?;
        let ranked = self.target.control.is_some();
        let mut jobs: Vec<DFlashJob> = Vec::with_capacity(s_count);
        let mut finished: Vec<bool> = vec![false; s_count];
        for (i, &(off, k)) in spans.iter().enumerate() {
            let old_start = self.active[i].seq.start;
            let (accepted, _, done) = self.active[i].seq.apply_verify(
                k,
                &ids[off..off + 1 + k],
                self.max_ctx,
                &mut self.hash,
            );
            let item = &mut self.active[i];
            if k > 0 {
                item.spec
                    .observe(k, accepted, item.last_lp.as_deref(), item.last_pred);
            }
            if done.is_some() || (!ranked && item.seq.cancelled) {
                finished[i] = true;
                continue;
            }
            let avail = item.seq.room(self.max_ctx);
            if item.seq.start + self.block <= self.max_ctx && avail > 0 {
                jobs.push(DFlashJob {
                    active_idx: i,
                    slot: item.seq.slot,
                    old_start,
                    new_start: item.seq.start,
                    anchor: item.seq.anchor,
                    tap_row_off: off,
                    commit: 1 + accepted,
                    avail,
                });
            }
        }
        let wants = self.drafter.draft_wants();
        for job in jobs {
            let taps_in = match &raw_taps {
                TapsBuf::Device { addr, .. } => TapsIn::Device {
                    addr: *addr + (job.tap_row_off * self.tap_width * 4) as u64,
                    rows: job.commit,
                    stride: 0,
                },
                TapsBuf::Host(v) => {
                    let lo = job.tap_row_off * self.tap_width;
                    let hi = lo + job.commit * self.tap_width;
                    TapsIn::Host(&v[lo..hi])
                }
            };
            let outs = self.drafter.run_taps(
                job.slot,
                job.anchor,
                job.new_start,
                job.old_start,
                taps_in,
                wants,
            )?;
            let (mut drafts, lp) = self.drafter.read_draft(&outs)?;
            let avail = drafts.len().min(job.avail);
            let item = &mut self.active[job.active_idx];
            let (k, pred) = item.spec.choose(lp.as_deref(), avail, drafts.len());
            drafts.truncate(k);
            item.seq.drafts = drafts;
            item.last_pred = pred;
            item.last_lp = lp;
        }
        let mut idx = 0usize;
        let mut orig = 0usize;
        while idx < self.active.len() {
            if finished[orig] {
                let item = self.active.remove(idx);
                self.free_slot(item.seq.slot);
            } else {
                idx += 1;
            }
            orig += 1;
        }
        Ok(())
    }
}

impl RoundHost for BatchDFlash<'_> {
    fn target(&mut self) -> &mut Engine {
        self.target
    }
    fn hash(&self) -> u32 {
        self.hash
    }
    fn is_busy(&self) -> bool {
        self.busy()
    }
    fn collect_cancelled(&self) -> Vec<u32> {
        self.active
            .iter()
            .filter(|s| s.seq.cancelled)
            .map(|s| s.seq.slot)
            .collect()
    }
    fn admit_local(&mut self) -> Vec<Admitted> {
        let mut out = Vec::new();
        while let Some((req, _)) = self.waiting.front() {
            let slot = if let Some(kept) = &mut self.kept {
                let busy =
                    !self.active.is_empty() || !self.prefilling.is_empty() || !out.is_empty();
                let pool = kept.totals().pool;
                let free_rows = pool.size.saturating_sub(pool.active);
                if busy && free_rows < req.tokens.len() + PAGE_HEADROOM {
                    break;
                }
                let Some((slot, _)) = kept.admit(&req.tokens, 0) else {
                    break;
                };
                slot
            } else {
                let Some(slot) = self.free.pop() else {
                    break;
                };
                slot
            };
            let (request, sink) = self.waiting.pop_front().expect("waiting request");
            out.push(Admitted {
                slot,
                request,
                sink: Some(sink),
            });
        }
        out
    }
    fn retire_slots(&mut self, slots: &[u32]) {
        let (kept, free) = (&mut self.kept, &mut self.free);
        retire_from(
            &mut self.active,
            slots,
            |s| s.seq.slot,
            |s| match kept {
                Some(k) => k.finish(s, Vec::new()),
                None => free.push(s),
            },
        );
    }
    fn run_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        self.exec_round(admitted, follower)
    }
}
