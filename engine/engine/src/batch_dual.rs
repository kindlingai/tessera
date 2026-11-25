//! Solo DFlash acceleration, demotion, and tail promotion for `BatchMtp`.

use crate::batch_spec::{BatchMtp, DraftJob};
use crate::dual::{DualSelector, run_dflash, split_taps};
use crate::mtp_draft::MtpDrafter;
use crate::spec_loop::{TapsBuf, TapsIn};
use crate::speculate::Drafter;
use crate::{Engine, Error};
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
