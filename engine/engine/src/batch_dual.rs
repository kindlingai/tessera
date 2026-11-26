//! Multi-slot DFlash decoding and solo DFlash/MTP promotion.

use crate::batch_spec::{
    ActiveSeq, Admitted, BatchCore, BatchMtp, DraftJob, RoundHost, batch_methods, drive_follow,
    drive_step, notify_kept, prefill_len, queue_request, retire_finished, retire_from, verify_rows,
};
use crate::dual::{DualSelector, DualState, DualStep, split_taps};
use crate::mtp_draft::{MtpDrafter, flush_mtp, take_cols};
use crate::schedule::Sink;
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{TapsBuf, TapsIn, span};
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
        let max_ctx = target
            .context()
            .min(drafter.context())
            .min(dflash.context());
        let block = dflash.block();
        let peer = DualPeer {
            dflash,
            selector,
            d_width,
            m_width,
            block,
        };
        Ok(Self::init(target, drafter, Some(peer), depth, max_ctx))
    }

    /// Flushes deferred MTP rows and switches to shared MTP.
    pub(crate) fn demote_solo(&mut self) -> Result<(), Error> {
        let Some(solo) = self.active[0].solo.take() else {
            return Ok(());
        };
        let slot = self.active[0].slot;
        flush_mtp(
            self.drafter,
            slot,
            solo.m_start,
            &solo.m_tokens,
            &solo.m_taps,
            self.core.prefill_chunk,
        )?;
        self.active[0].mtp_valid = solo.m_start as usize + solo.m_tokens.len();
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
        let skip = p.skip;
        let m_toks = std::mem::take(&mut p.m_tok_buf);
        let m_taps = std::mem::take(&mut p.m_tap_buf);
        if let Some(peer) = &self.dual {
            let raw = std::mem::take(&mut p.d_buf);
            push_tail(&mut p.d_buf, &raw, peer.d_width);
        }
        flush_mtp(
            self.drafter,
            slot,
            skip as u32,
            &m_toks,
            &m_taps,
            self.core.prefill_chunk,
        )?;
        if let Some(p) = self.prefilling.front_mut() {
            p.mtp_valid = skip + m_toks.len();
        }
        Ok(())
    }

    /// Splits target taps into `(d_taps, m_taps)` for dual mode.
    pub(crate) fn split_dual_taps(
        &self,
        taps: TapsBuf,
        rows: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let peer = self.dual.as_ref().expect("dual peer");
        let total = peer.d_width + peer.m_width;
        let raw = taps.into_host(&*self.core.target.runner, total)?;
        Ok(split_taps(&raw, rows, peer.d_width, peer.m_width))
    }

    /// Reads DFlash taps for `rows` without consuming device `taps`.
    pub(crate) fn read_dual_d_taps(&self, taps: &TapsBuf, rows: usize) -> Result<Vec<f32>, Error> {
        let peer = self.dual.as_ref().expect("dual peer");
        let total = peer.d_width + peer.m_width;
        match taps {
            TapsBuf::Host(v) => Ok(take_cols(v, rows, total, 0, peer.d_width)),
            TapsBuf::Device { slot, .. } => {
                let raw = self.core.target.runner.read(*slot)?;
                let floats = raw.floats().map_err(tessera_run::Error::from)?;
                Ok(take_cols(floats, rows, total, 0, peer.d_width))
            }
        }
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
        seq.solo = Some(DualState::new(
            selector,
            (seq.start - w_rows, 0, d_pending),
            (job.old_start, job.committed, m_taps),
        ));
        self.solo_draft(job.active_idx)
    }

    /// Verifies and re-drafts one solo sequence using DFlash or MTP.
    pub(crate) fn verify_solo(&mut self) -> Result<(), Error> {
        let seq = &self.active[0];
        let k = seq.drafts.len().min(seq.room(self.core.max_ctx));
        let mut rows = Vec::with_capacity(1 + k);
        seq.verify(&mut rows, k);
        let sampling = !self.active[0].sampler.greedy();
        let (mut out, taps) = self.core.target.step_verify_local(&rows, sampling)?;
        let (d_all, m_all) = self.split_dual_taps(taps, rows.len())?;
        let d_width = self.dual.as_ref().expect("dual peer").d_width;
        let m_width = self.drafter.hidden();
        let ranked = self.core.target.control.is_some();
        let (accepted, committed, done) =
            self.active[0].apply_verify(k, 0, &mut out, self.core.max_ctx, &mut self.core.hash);
        self.active[0].history.extend_from_slice(&committed);
        let commit = committed.len();
        if done.is_some() || (!ranked && self.active[0].cancelled) {
            let mut seq = self.active.remove(0);
            if self.core.kept.is_some()
                && let Some(solo) = seq.solo.take()
            {
                flush_mtp(
                    self.drafter,
                    seq.slot,
                    solo.m_start,
                    &solo.m_tokens,
                    &solo.m_taps,
                    self.core.prefill_chunk,
                )?;
                seq.mtp_valid = solo.m_start as usize + solo.m_tokens.len();
            }
            self.retire_slot(seq.slot, seq.history, seq.mtp_valid);
            return Ok(());
        }
        let seq = &mut self.active[0];
        let solo = seq.solo.as_mut().expect("solo state");
        solo.observe(
            k,
            accepted,
            &committed,
            &d_all[..commit * d_width],
            &m_all[..commit * m_width],
            None,
        );
        self.solo_draft(0)
    }

    /// Proposes next drafts for solo sequence `idx` via `DualSelector`.
    pub(crate) fn solo_draft(&mut self, idx: usize) -> Result<(), Error> {
        let room = self.active[idx].room(self.core.max_ctx);
        if room == 0 {
            self.active[idx].drafts.clear();
            return Ok(());
        }
        let seq = &mut self.active[idx];
        let solo = seq.solo.as_mut().expect("solo");
        let peer = self.dual.as_mut().expect("dual peer");
        let step = DualStep {
            slot: seq.slot,
            start: seq.start,
            anchor: seq.anchor,
            room,
            depth: self.depth,
            max_ctx: self.core.max_ctx,
            chunk: self.core.prefill_chunk,
        };
        let (drafts, mtp_valid) = solo.draft(self.core.target, peer.dflash, self.drafter, step)?;
        seq.drafts = drafts;
        if let Some(valid) = mtp_valid {
            seq.mtp_valid = valid;
        }
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
    d_base: usize,
}

struct DFlashSeq {
    seq: ActiveSeq,
    spec: SpecLen,
    last_lp: Option<Vec<f32>>,
    d_base: usize,
}

struct DFlashJob {
    active_idx: usize,
    slot: u32,
    d_old: usize,
    d_new: usize,
    anchor: u32,
    tap_row_off: usize,
    commit: usize,
    avail: usize,
}

/// Multi-slot continuous-batching DFlash decoder.
pub struct BatchDFlash<'a> {
    core: BatchCore<'a>,
    drafter: &'a mut Drafter,
    spec_len: SpecLen,
    draft_ctx: usize,
    block: usize,
    tap_width: usize,
    reuse: bool,
    admitted_skips: VecDeque<usize>,
    prefilling: VecDeque<DFlashPrefill>,
    active: Vec<DFlashSeq>,
}

impl<'a> BatchDFlash<'a> {
    /// Pairs a multi-slot tapped target engine with its DFlash drafter.
    pub fn new(target: &'a mut Engine, drafter: &'a mut Drafter) -> Result<Self, Error> {
        let width = target.draft_taps()?;
        if width != drafter.tap_width() {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        let max_ctx = target.context();
        let draft_ctx = drafter.context();
        let block = drafter.block();
        let spec_len = SpecLen::fixed(block - 1).with_cost(CostModel::dflash());
        Ok(Self {
            core: BatchCore::new(target, slots, max_ctx, true),
            drafter,
            spec_len,
            draft_ctx,
            block,
            tap_width: width,
            reuse: false,
            admitted_skips: VecDeque::new(),
            prefilling: VecDeque::new(),
            active: Vec::new(),
        })
    }

    /// Configures adaptive or truncated draft length selection.
    pub fn with_spec_len(mut self, spec_len: SpecLen) -> Self {
        self.spec_len = spec_len;
        self
    }

    fn admit_limit(&self, len: usize) -> usize {
        if self.reuse { len.saturating_sub(1) } else { 0 }
    }

    fn initial_d_base(&self, prompt_len: usize) -> usize {
        if prompt_len + self.block <= self.draft_ctx {
            return 0;
        }
        let chunk = self.core.prefill_chunk;
        let keep = self
            .draft_ctx
            .saturating_sub(self.block + chunk)
            .min(self.draft_ctx / 2);
        (prompt_len.saturating_sub(keep) / chunk) * chunk
    }

    fn free_slot(&mut self, slot: u32, history: Vec<u32>) {
        self.core
            .retire(slot, if self.reuse { history } else { Vec::new() });
    }

    fn ensure_pages(&mut self, rows: &[Row]) -> Result<(), Error> {
        let (Some(kept), Some(lanes)) = (&mut self.core.kept, &mut self.core.target.lanes) else {
            return Ok(());
        };
        kept.grow_rows(rows, lanes)
    }

    fn exec_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        for mut item in admitted {
            let limit = self.admit_limit(item.request.tokens.len());
            let skip = if follower {
                if let Some(kept) = &mut self.core.kept {
                    let Some((slot, skip)) = kept.admit(&item.request.tokens, limit) else {
                        return Err(Error::RowPlace);
                    };
                    debug_assert_eq!(slot, item.slot);
                    skip
                } else {
                    0
                }
            } else {
                self.admitted_skips.pop_front().unwrap_or(0)
            };
            notify_kept(&mut item.sink, skip);
            let d_base = skip + self.initial_d_base(item.request.tokens.len() - skip);
            self.prefilling.push_back(DFlashPrefill {
                slot: item.slot,
                request: item.request,
                sink: item.sink,
                pos: skip,
                d_base,
            });
        }
        if !self.prefilling.is_empty() {
            self.prefill_step()?;
        }
        self.active.sort_by_key(|s| s.seq.slot);
        if !self.active.is_empty() {
            self.verify_and_draft()?;
        }
        if let Some(kept) = &self.core.kept {
            kept.publish();
        }
        Ok(())
    }

    fn prefill_step(&mut self) -> Result<(), Error> {
        let mut seq = self.prefilling.pop_front().expect("prefill seq");
        let n = seq.request.tokens.len();
        let pos0 = seq.pos;
        let len = prefill_len(
            n,
            pos0,
            self.core.prefill_chunk,
            seq.request.sampling.greedy(),
        );
        let end = pos0 + len;
        let mut rows = Vec::with_capacity(len);
        span(
            &mut rows,
            seq.slot,
            pos0,
            seq.request.tokens[pos0..end].iter().copied(),
        );
        self.ensure_pages(&rows)?;
        let wants = self.drafter.draft_wants();
        if end < n {
            let (_, taps) = self.core.target.step_taps_local(&rows)?;
            if pos0 >= seq.d_base && end - seq.d_base + self.block <= self.draft_ctx {
                self.drafter.run_taps(
                    seq.slot,
                    seq.request.tokens[end],
                    end - seq.d_base,
                    pos0 - seq.d_base,
                    taps.as_in(),
                    wants,
                )?;
            }
            seq.pos = end;
            self.prefilling.push_front(seq);
            return Ok(());
        }
        let (mut act, taps, ended) =
            self.core
                .finish_prefill(seq.slot, seq.request, seq.sink, &rows, 0, false)?;
        if ended {
            self.free_slot(act.slot, act.history);
            return Ok(());
        }
        let spec = self.spec_len.fresh();
        let mut last_lp = None;
        let room = act.room(self.core.max_ctx);
        let mut d_base = seq.d_base.min(pos0);
        if n - d_base + self.block > self.draft_ctx {
            d_base = pos0;
        }
        if n - d_base + self.block <= self.draft_ctx && room > 0 {
            let outs = self.drafter.run_taps(
                act.slot,
                act.anchor,
                n - d_base,
                pos0 - d_base,
                taps.as_in(),
                wants,
            )?;
            let (mut drafts, lp) = self.drafter.read_draft(&outs)?;
            let avail = drafts.len().min(room);
            let k = spec.choose(lp.as_deref(), avail, drafts.len());
            drafts.truncate(k);
            act.drafts = drafts;
            last_lp = lp;
        }
        self.active.push(DFlashSeq {
            seq: act,
            spec,
            last_lp,
            d_base,
        });
        Ok(())
    }

    fn verify_and_draft(&mut self) -> Result<(), Error> {
        let s_count = self.active.len();
        let (rows, spans, sampling) = verify_rows(&self.active, self.core.max_ctx, |s| &s.seq);
        self.ensure_pages(&rows)?;
        let (mut out, raw_taps) = self.core.target.step_verify_local(&rows, sampling)?;
        let ranked = self.core.target.control.is_some();
        let mut jobs: Vec<DFlashJob> = Vec::with_capacity(s_count);
        let mut finished: Vec<bool> = vec![false; s_count];
        for (i, &(off, k)) in spans.iter().enumerate() {
            let old_start = self.active[i].seq.start;
            let old_anchor = self.active[i].seq.anchor;
            let (accepted, committed, done) = self.active[i].seq.apply_verify(
                k,
                off,
                &mut out,
                self.core.max_ctx,
                &mut self.core.hash,
            );
            let item = &mut self.active[i];
            item.seq.history.push(old_anchor);
            item.seq.history.extend_from_slice(&committed[..accepted]);
            if k > 0 {
                item.spec.observe(k, accepted, item.last_lp.as_deref());
            }
            if done.is_some() || (!ranked && item.seq.cancelled) {
                finished[i] = true;
                continue;
            }
            let avail = item.seq.room(self.core.max_ctx);
            if item.seq.start - item.d_base + self.block > self.draft_ctx {
                item.d_base = old_start;
            }
            let d_old = old_start - item.d_base;
            let d_new = item.seq.start - item.d_base;
            if d_new + self.block <= self.draft_ctx && avail > 0 {
                jobs.push(DFlashJob {
                    active_idx: i,
                    slot: item.seq.slot,
                    d_old,
                    d_new,
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
            let outs = self
                .drafter
                .run_taps(job.slot, job.anchor, job.d_new, job.d_old, taps_in, wants)?;
            let (mut drafts, lp) = self.drafter.read_draft(&outs)?;
            let avail = drafts.len().min(job.avail);
            let item = &mut self.active[job.active_idx];
            let k = item.spec.choose(lp.as_deref(), avail, drafts.len());
            drafts.truncate(k);
            item.seq.drafts = drafts;
            item.last_lp = lp;
        }
        for item in retire_finished(&mut self.active, &finished) {
            self.free_slot(item.seq.slot, item.seq.history);
        }
        Ok(())
    }
}

batch_methods!(BatchDFlash, reuse);

impl<'a> RoundHost<'a> for BatchDFlash<'a> {
    fn core(&mut self) -> &mut BatchCore<'a> {
        &mut self.core
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
        // Retire cancelled slots before admitting so ranks match.
        if self.active.iter().any(|s| s.seq.cancelled) {
            return Vec::new();
        }
        let mut out = Vec::new();
        while let Some((req, _)) = self.core.waiting.front() {
            let limit = self.admit_limit(req.tokens.len());
            let (slot, skip) = if let Some(kept) = &mut self.core.kept {
                if self.core.target.lanes.is_some() {
                    let busy =
                        !self.active.is_empty() || !self.prefilling.is_empty() || !out.is_empty();
                    let pool = kept.totals().pool;
                    let free_rows = pool.size.saturating_sub(pool.active);
                    if busy && free_rows < req.tokens.len() + PAGE_HEADROOM {
                        break;
                    }
                }
                let Some((slot, skip)) = kept.admit(&req.tokens, limit) else {
                    break;
                };
                (slot, skip)
            } else {
                let Some(slot) = self.core.free.pop() else {
                    break;
                };
                (slot, 0)
            };
            self.admitted_skips.push_back(skip);
            out.push(self.core.pop_admitted(slot, 0));
        }
        out
    }
    fn retire_slots(&mut self, slots: &[u32]) {
        for s in retire_from(&mut self.active, slots, |s| s.seq.slot) {
            self.free_slot(s.seq.slot, s.seq.history);
        }
    }
    fn run_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        self.exec_round(admitted, follower)
    }
}
