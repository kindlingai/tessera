//! Generic continuous-batching MTP and dual speculative decoding.

use crate::batch_dual::{DualPeer, PROMOTE_MIN_REM, SoloState, push_tail};
use crate::mtp_draft::{MtpDrafter, MtpSpan};
use crate::sample::{Target, accept, take};
use crate::schedule::Sink;
use crate::spec_loop::{TapsBuf, TapsIn, fnv_u32, span};
use crate::{Engine, Error, Event, Reason, Request, Row};
use std::collections::VecDeque;
use tessera_engine_comm::{Reader, Writer};
use tessera_run::Error as RunError;

/// Wire tag identifying a batched speculative round header.
const BATCH_STEP_TAG: u32 = 2;

/// Default maximum rows per prefill chunk.
pub(crate) const DEFAULT_CHUNK: usize = tessera_device::FAST_MAX_ROWS;

/// One active request decoded inside a shared speculative batch.
pub(crate) struct ActiveSeq {
    pub(crate) slot: u32,
    pub(crate) request: Request,
    pub(crate) sink: Option<Sink>,
    /// Next target position (prompt length plus emitted tokens).
    pub(crate) start: usize,
    /// Last committed token, verified at row 0 next round.
    pub(crate) anchor: u32,
    /// Current round's proposed draft tokens.
    pub(crate) drafts: Vec<u32>,
    pub(crate) emitted: usize,
    /// Set on rank 0 when client drops the stream.
    pub(crate) cancelled: bool,
    /// Active solo DFlash/MTP state when decoding alone.
    pub(crate) solo: Option<SoloState>,
    /// Rolling recent DFlash taps for tail promotion.
    pub(crate) d_tail: Vec<f32>,
}

impl ActiveSeq {
    /// Applies target verify outputs, emitting tokens and returning finish status.
    pub(crate) fn apply_verify(
        &mut self,
        k: usize,
        ids: &[u32],
        max_ctx: usize,
        hash: &mut u32,
    ) -> (usize, Vec<u32>, Option<Reason>) {
        let (accepted, bonus) = accept(&self.drafts[..k], Target::Ids(&ids[..1 + k]));
        fnv_u32(hash, accepted as u32);
        fnv_u32(hash, bonus);
        let commit = 1 + accepted;
        let mut committed = Vec::with_capacity(commit);
        committed.extend_from_slice(&self.drafts[..accepted]);
        committed.push(bonus);
        self.start += commit;
        self.anchor = bonus;
        self.drafts.clear();
        let done = self.commit(&committed, max_ctx);
        (accepted, committed, done)
    }

    /// Takes `tokens` and sends them. Returns the end, if any.
    pub(crate) fn commit(&mut self, tokens: &[u32], max_ctx: usize) -> Option<Reason> {
        let mut done = None;
        for &tok in tokens {
            done = take(&self.request, &mut self.emitted, tok, || {
                if !self.cancelled
                    && let Some(cb) = &mut self.sink
                    && !cb(Event::Token(tok))
                {
                    self.cancelled = true;
                    let _ = cb(Event::Done(Reason::Stop));
                }
                true
            });
            if done.is_some() {
                break;
            }
        }
        if done.is_none() && self.start >= max_ctx {
            done = Some(Reason::Context);
        }
        if let Some(reason) = done
            && !self.cancelled
            && let Some(cb) = &mut self.sink
        {
            let _ = cb(Event::Done(reason));
        }
        done
    }

    /// The room for drafts in the next verify.
    pub(crate) fn room(&self, max_ctx: usize) -> usize {
        let rem = self.request.max_tokens.saturating_sub(self.emitted);
        let ctx_rem = max_ctx.saturating_sub(self.start);
        rem.saturating_sub(1).min(ctx_rem.saturating_sub(1))
    }

    /// Appends the anchor's row and `k` draft rows.
    pub(crate) fn verify(&self, rows: &mut Vec<Row>, k: usize) {
        let block = self.drafts[..k].iter().copied();
        span(
            rows,
            self.slot,
            self.start,
            [self.anchor].into_iter().chain(block),
        );
    }
}

/// Surviving session state handed from verify to batched MTP draft.
pub(crate) struct DraftJob {
    pub(crate) active_idx: usize,
    pub(crate) slot: u32,
    pub(crate) old_start: u32,
    pub(crate) new_start: usize,
    pub(crate) committed: Vec<u32>,
    pub(crate) tap_row_off: usize,
    pub(crate) avail: usize,
}

/// Newly admitted session synchronized across ranks.
struct Admitted {
    slot: u32,
    request: Request,
    sink: Option<Sink>,
}

/// Admitted sequence advancing through chunked prompt prefill.
pub(crate) struct PrefillSeq {
    pub(crate) slot: u32,
    pub(crate) request: Request,
    pub(crate) sink: Option<Sink>,
    pub(crate) pos: usize,
    pub(crate) solo_mode: bool,
    pub(crate) d_buf: Vec<f32>,
    pub(crate) m_tok_buf: Vec<u32>,
    pub(crate) m_tap_buf: Vec<f32>,
}

/// Multi-slot continuous-batching MTP and optional DFlash decoder.
pub struct BatchMtp<'a> {
    pub(crate) target: &'a mut Engine,
    pub(crate) drafter: &'a mut MtpDrafter,
    pub(crate) dual: Option<DualPeer<'a>>,
    pub(crate) depth: usize,
    pub(crate) prefill_chunk: usize,
    pub(crate) max_ctx: usize,
    pub(crate) waiting: VecDeque<(Request, Sink)>,
    pub(crate) prefilling: VecDeque<PrefillSeq>,
    pub(crate) active: Vec<ActiveSeq>,
    pub(crate) free: Vec<u32>,
    pub(crate) hash: u32,
}

impl<'a> BatchMtp<'a> {
    /// Pairs a multi-slot tapped target engine with its MTP drafter.
    pub fn new(
        target: &'a mut Engine,
        drafter: &'a mut MtpDrafter,
        depth: usize,
    ) -> Result<Self, Error> {
        if depth == 0 {
            return Err(Error::Rows);
        }
        let width = target.draft_taps()?;
        if width != drafter.hidden() {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        let max_ctx = target.context().min(drafter.context());
        let free = (0..slots as u32).rev().collect();
        Ok(Self {
            target,
            drafter,
            dual: None,
            depth,
            prefill_chunk: DEFAULT_CHUNK,
            max_ctx,
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

    /// Queues an answer. Unfit requests finish at once.
    pub fn submit(&mut self, request: Request, mut sink: Sink) {
        if request.tokens.is_empty() || request.tokens.len() > self.max_ctx {
            sink(Event::Done(Reason::Context));
            return;
        }
        if request.max_tokens == 0 {
            sink(Event::Done(Reason::Length));
            return;
        }
        self.waiting.push_back((request, sink));
    }

    /// Whether any answer is waiting, prefilling, or decoding.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.prefilling.is_empty() || !self.active.is_empty()
    }

    /// Runs one round: admit, prefill, shared verify, and batched draft.
    pub fn step(&mut self) -> Result<bool, Error> {
        if !self.busy() {
            return Ok(false);
        }
        self.target.settle()?;
        let cancelled = self.collect_cancelled();
        let admitted = self.admit_local();
        if self.target.control.is_some() {
            let msg = encode_round(self.hash, &cancelled, &admitted);
            let _ = self.target.told(&msg)?;
        }
        self.retire_slots(&cancelled);
        self.run_round(admitted)?;
        self.target.settle()?;
        Ok(true)
    }

    /// Follower rank loop executing rank 0's batched rounds.
    pub fn follow(&mut self) -> Result<(), Error> {
        loop {
            let mine = self
                .target
                .held
                .take()
                .map_or_else(Vec::new, |(_, p)| p.to_bytes());
            let Some(told) = self.target.told(&mine)? else {
                return Ok(());
            };
            let Some((want_hash, cancelled, admitted)) = decode_round(&told) else {
                return Ok(());
            };
            if want_hash != self.hash {
                return Err(RunError::OutputsDisagree.into());
            }
            self.retire_slots(&cancelled);
            self.run_round(admitted)?;
        }
    }

    fn collect_cancelled(&self) -> Vec<u32> {
        self.active
            .iter()
            .filter(|s| s.cancelled)
            .map(|s| s.slot)
            .collect()
    }

    fn admit_local(&mut self) -> Vec<Admitted> {
        let mut out = Vec::new();
        while !self.waiting.is_empty() && !self.free.is_empty() {
            let slot = self.free.pop().expect("free slot");
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
        if slots.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.active.len() {
            if slots.contains(&self.active[i].slot) {
                let seq = self.active.remove(i);
                self.free.push(seq.slot);
            } else {
                i += 1;
            }
        }
    }

    fn run_round(&mut self, admitted: Vec<Admitted>) -> Result<(), Error> {
        if !admitted.is_empty() {
            if self.active.len() == 1 && self.active[0].solo.is_some() {
                self.demote_solo()?;
            }
            if self.prefilling.len() == 1 && self.prefilling[0].solo_mode {
                self.demote_prefill()?;
            }
            let solo_admit = self.dual.is_some()
                && self.active.is_empty()
                && self.prefilling.is_empty()
                && admitted.len() == 1;
            for item in admitted {
                self.prefilling.push_back(PrefillSeq {
                    slot: item.slot,
                    request: item.request,
                    sink: item.sink,
                    pos: 0,
                    solo_mode: solo_admit,
                    d_buf: Vec::new(),
                    m_tok_buf: Vec::new(),
                    m_tap_buf: Vec::new(),
                });
            }
        }
        if !self.prefilling.is_empty() {
            self.prefill_step()?;
        }
        self.active.sort_by_key(|s| s.slot);
        if self.active.is_empty() {
            return Ok(());
        }
        if self.active.len() == 1 && self.active[0].solo.is_some() {
            return self.verify_solo();
        }
        self.verify_and_draft()
    }

    /// Advances the head prefilling sequence by one chunk.
    fn prefill_step(&mut self) -> Result<(), Error> {
        let mut seq = self.prefilling.pop_front().expect("prefill seq");
        let n = seq.request.tokens.len();
        let pos0 = seq.pos;
        let len = (n - pos0).min(self.prefill_chunk);
        let wants = self.drafter.wants();
        let dual_widths = self.dual.as_ref().map(|p| (p.d_width, p.m_width));
        let mut rows = Vec::with_capacity(len);
        let chunk = &seq.request.tokens[pos0..pos0 + len];
        span(&mut rows, seq.slot, pos0, chunk.iter().copied());
        let (ids, taps) = self.target.step_taps_local(&rows)?;
        if pos0 + len < n {
            let shifted = &seq.request.tokens[pos0 + 1..pos0 + len + 1];
            if let Some((d_w, m_w)) = dual_widths {
                if seq.solo_mode {
                    let (dt, mt) = self.split_dual_taps(taps, len)?;
                    seq.d_buf.extend_from_slice(&dt);
                    seq.m_tok_buf.extend_from_slice(shifted);
                    seq.m_tap_buf.extend_from_slice(&mt);
                } else {
                    if self.active.len() + self.prefilling.len() < 2 {
                        let dt = self.read_dual_d_taps(&taps, len)?;
                        push_tail(&mut seq.d_buf, &dt, d_w);
                    }
                    self.run_mtp_prefill_chunk(seq.slot, pos0 as u32, shifted, &taps, d_w, m_w)?;
                }
            } else {
                self.drafter
                    .run_taps(seq.slot, pos0 as u32, shifted, taps.as_in(), wants, true)?;
            }
            seq.pos = pos0 + len;
            self.prefilling.push_front(seq);
            return Ok(());
        }
        let anchor = *ids.last().expect("non-empty chunk");
        fnv_u32(&mut self.hash, anchor);
        let mut shifted = Vec::with_capacity(len);
        shifted.extend_from_slice(&seq.request.tokens[pos0 + 1..]);
        shifted.push(anchor);
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
            self.free.push(act.slot);
            return Ok(());
        }
        if seq.solo_mode {
            let (dt, mt) = self.split_dual_taps(taps, len)?;
            seq.d_buf.extend_from_slice(&dt);
            seq.m_tok_buf.extend_from_slice(&shifted);
            seq.m_tap_buf.extend_from_slice(&mt);
            let selector = self.dual.as_ref().expect("dual peer").selector.fresh();
            act.solo = Some(SoloState {
                selector,
                d_base: 0,
                d_ctx_pos: 0,
                d_pending: seq.d_buf,
                m_start: 0,
                m_tokens: seq.m_tok_buf,
                m_taps: seq.m_tap_buf,
                round_mtp: false,
                last_pred: 1.0,
                last_lp: None,
                last_mtp_d1: None,
            });
            self.active.push(act);
            let idx = self.active.len() - 1;
            self.solo_draft(idx)?;
            return Ok(());
        }
        if let Some((d_w, _)) = dual_widths
            && self.active.len() + self.prefilling.len() < 2
        {
            let dt = self.read_dual_d_taps(&taps, len)?;
            push_tail(&mut seq.d_buf, &dt, d_w);
        }
        let avail = self.depth.min(act.room(self.max_ctx));
        act.drafts = if avail > 0 {
            let job = DraftJob {
                active_idx: 0,
                slot: act.slot,
                old_start: pos0 as u32,
                new_start: n,
                committed: shifted,
                tap_row_off: 0,
                avail,
            };
            self.draft_batch(&[job], &taps, dual_widths.is_some())?
                .pop()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        act.d_tail = seq.d_buf;
        self.active.push(act);
        Ok(())
    }

    fn verify_and_draft(&mut self) -> Result<(), Error> {
        let s_count = self.active.len();
        let mut rows = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(s_count);
        for seq in &self.active {
            let k = seq.drafts.len().min(seq.room(self.max_ctx));
            spans.push((rows.len(), k));
            seq.verify(&mut rows, k);
        }
        let (ids, raw_taps) = self.target.step_taps_local(&rows)?;
        let dual_info = self.dual.as_ref().map(|p| (p.d_width, p.block));
        let ranked = self.target.control.is_some();
        let mut jobs: Vec<DraftJob> = Vec::with_capacity(s_count);
        let mut finished: Vec<bool> = vec![false; s_count];
        for (i, &(off, k)) in spans.iter().enumerate() {
            let old_start = self.active[i].start as u32;
            let (_, committed, done) = self.active[i].apply_verify(
                k,
                &ids[off..off + 1 + k],
                self.max_ctx,
                &mut self.hash,
            );
            let seq = &mut self.active[i];
            if done.is_some() || (!ranked && seq.cancelled) {
                finished[i] = true;
                continue;
            }
            let avail = self.depth.min(seq.room(self.max_ctx));
            if avail > 0 {
                jobs.push(DraftJob {
                    active_idx: i,
                    slot: seq.slot,
                    old_start,
                    new_start: seq.start,
                    committed,
                    tap_row_off: off,
                    avail,
                });
            }
        }
        let survivors = finished.iter().filter(|&&f| !f).count();
        let promote = survivors == 1
            && self.prefilling.is_empty()
            && jobs.len() == 1
            && dual_info.is_some_and(|(_, block)| {
                let seq = &self.active[jobs[0].active_idx];
                let rem = seq.request.max_tokens.saturating_sub(seq.emitted);
                seq.start + block <= self.max_ctx && rem >= PROMOTE_MIN_REM
            });
        if promote {
            let job = jobs.remove(0);
            let d_w = dual_info.expect("dual info").0;
            let (d_all, m_all) = self.split_dual_taps(raw_taps, rows.len())?;
            let lo = job.tap_row_off * d_w;
            let hi = (job.tap_row_off + job.committed.len()) * d_w;
            push_tail(&mut self.active[job.active_idx].d_tail, &d_all[lo..hi], d_w);
            self.promote_solo(job, &m_all)?;
        } else if !jobs.is_empty() {
            if let Some((d_w, _)) = dual_info
                && survivors + self.prefilling.len() <= 2
            {
                let d_all = self.read_dual_d_taps(&raw_taps, rows.len())?;
                for job in &jobs {
                    let lo = job.tap_row_off * d_w;
                    let hi = (job.tap_row_off + job.committed.len()) * d_w;
                    push_tail(&mut self.active[job.active_idx].d_tail, &d_all[lo..hi], d_w);
                }
            }
            let drafts_all = self.draft_batch(&jobs, &raw_taps, dual_info.is_some())?;
            for (job, drafts) in jobs.into_iter().zip(drafts_all) {
                self.active[job.active_idx].drafts = drafts;
            }
        }
        let mut idx = 0usize;
        let mut orig = 0usize;
        while idx < self.active.len() {
            if finished[orig] {
                let seq = self.active.remove(idx);
                self.free.push(seq.slot);
            } else {
                idx += 1;
            }
            orig += 1;
        }
        Ok(())
    }

    /// Feeds one non-final dual prefill chunk directly into `MtpDrafter`.
    fn run_mtp_prefill_chunk(
        &mut self,
        slot: u32,
        pos0: u32,
        tokens: &[u32],
        taps: &TapsBuf,
        d_w: usize,
        m_w: usize,
    ) -> Result<(), Error> {
        let wants = self.drafter.wants();
        let stride = d_w + m_w;
        match taps {
            TapsBuf::Device { addr, .. } => {
                let h = TapsIn::Device {
                    addr: *addr + (d_w * 4) as u64,
                    rows: tokens.len(),
                    stride,
                };
                self.drafter.run_taps(slot, pos0, tokens, h, wants, true)?;
            }
            TapsBuf::Host(v) => {
                let mut mt = Vec::with_capacity(tokens.len() * m_w);
                for r in 0..tokens.len() {
                    let lo = r * stride + d_w;
                    mt.extend_from_slice(&v[lo..lo + m_w]);
                }
                self.drafter
                    .run_taps(slot, pos0, tokens, TapsIn::Host(&mt), wants, true)?;
            }
        }
        Ok(())
    }

    /// Runs batched first-form MTP plus shared chain steps for `jobs`.
    pub(crate) fn draft_batch(
        &mut self,
        jobs: &[DraftJob],
        taps: &TapsBuf,
        strided: bool,
    ) -> Result<Vec<Vec<u32>>, Error> {
        let n = jobs.len();
        let chain_n = jobs.iter().map(|j| j.avail).min().unwrap_or(0);
        if chain_n == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let hidden = self.drafter.hidden();
        let wants = self.drafter.wants();
        let (col_off, stride) = match (strided, self.dual.as_ref()) {
            (true, Some(p)) => (p.d_width, p.d_width + p.m_width),
            _ => (0, hidden),
        };
        let mut end_rows = Vec::with_capacity(n);
        let mut total_rows = 0usize;
        for j in jobs {
            total_rows += j.committed.len();
            end_rows.push(total_rows - 1);
        }
        let dev_taps = match *taps {
            TapsBuf::Device { slot, addr, .. } => Some((slot, addr)),
            TapsBuf::Host(_) => None,
        };
        let can_gpu_chain =
            chain_n >= 2 && dev_taps.is_some_and(|(slot, _)| slot.bytes() >= n * (chain_n - 1) * 4);
        let host_slices: Vec<Vec<f32>> = match taps {
            TapsBuf::Host(v) if col_off > 0 => jobs
                .iter()
                .map(|j| {
                    let mut buf = Vec::with_capacity(j.committed.len() * hidden);
                    for r in 0..j.committed.len() {
                        let lo = (j.tap_row_off + r) * stride + col_off;
                        buf.extend_from_slice(&v[lo..lo + hidden]);
                    }
                    buf
                })
                .collect(),
            _ => Vec::new(),
        };
        let spans: Vec<MtpSpan<'_>> = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let h = match taps {
                    TapsBuf::Device { addr, .. } => TapsIn::Device {
                        addr: *addr + ((j.tap_row_off * stride + col_off) * 4) as u64,
                        rows: j.committed.len(),
                        stride: if col_off > 0 { stride } else { 0 },
                    },
                    TapsBuf::Host(_) if col_off > 0 => TapsIn::Host(&host_slices[i]),
                    TapsBuf::Host(v) => {
                        let lo = j.tap_row_off * hidden;
                        let hi = lo + j.committed.len() * hidden;
                        TapsIn::Host(&v[lo..hi])
                    }
                };
                MtpSpan {
                    slot: j.slot,
                    start_pos: j.old_start,
                    tokens: &j.committed,
                    hidden: h,
                }
            })
            .collect();
        let first_wait = !can_gpu_chain;
        let first_out = self.drafter.run_batch(&spans, wants, first_wait)?;
        if can_gpu_chain
            && let Some((tap_slot, tap_addr)) = dev_taps
            && let (Some(o0), Some(o1)) = (
                self.drafter.address(first_out[0]),
                self.drafter.address(first_out[1]),
            )
        {
            let slots: Vec<u32> = jobs.iter().map(|j| j.slot).collect();
            let mut tok_addrs: Vec<u64> = end_rows.iter().map(|&r| o0 + (r * 4) as u64).collect();
            let mut hid_addrs: Vec<u64> = end_rows
                .iter()
                .map(|&r| o1 + (r * hidden * 4) as u64)
                .collect();
            let mut last_out = first_out[0];
            for step in 1..chain_n {
                let mut saved_addrs = Vec::with_capacity(n);
                let mut positions = Vec::with_capacity(n);
                for (i, j) in jobs.iter().enumerate() {
                    let off = ((step - 1) * n + i) * 4;
                    self.target
                        .runner
                        .copy_from(tap_slot, off, tok_addrs[i], 4)?;
                    saved_addrs.push(tap_addr + off as u64);
                    positions.push((j.new_start - 1 + step) as u32);
                }
                let wait = step + 1 == chain_n;
                let outs = self.drafter.run_chain_batch(
                    &slots,
                    &positions,
                    &saved_addrs,
                    &hid_addrs,
                    wants,
                    wait,
                )?;
                last_out = outs[0];
                if !wait {
                    let next0 = self
                        .drafter
                        .address(outs[0])
                        .ok_or(RunError::Unaddressed(0))?;
                    let next1 = self
                        .drafter
                        .address(outs[1])
                        .ok_or(RunError::Unaddressed(0))?;
                    for i in 0..n {
                        tok_addrs[i] = next0 + (i * 4) as u64;
                        hid_addrs[i] = next1 + (i * hidden * 4) as u64;
                    }
                }
            }
            let saved = self
                .target
                .runner
                .held()
                .read_u32s(tap_slot, (chain_n - 1) * n)?;
            let final_ids = self.drafter.ids(last_out)?;
            let mut out = vec![Vec::with_capacity(chain_n); n];
            for step in 0..chain_n - 1 {
                for i in 0..n {
                    out[i].push(saved[step * n + i]);
                }
            }
            for i in 0..n {
                out[i].push(final_ids[i]);
            }
            return Ok(out);
        }
        // Host fallback path (CPU runners or chain_n == 1).
        let ids = self.drafter.ids(first_out[0])?;
        let mut out: Vec<Vec<u32>> = end_rows.iter().map(|&r| vec![ids[r]]).collect();
        if chain_n == 1 {
            return Ok(out);
        }
        let mut prev_hid = self.drafter.read_floats(first_out[1])?;
        let mut cur_rows: Vec<Vec<f32>> = end_rows
            .iter()
            .map(|&r| prev_hid[r * hidden..(r + 1) * hidden].to_vec())
            .collect();
        for step in 1..chain_n {
            let toks: Vec<[u32; 1]> = out.iter().map(|v| [*v.last().unwrap()]).collect();
            let step_spans: Vec<MtpSpan<'_>> = jobs
                .iter()
                .enumerate()
                .map(|(i, j)| MtpSpan {
                    slot: j.slot,
                    start_pos: (j.new_start - 1 + step) as u32,
                    tokens: &toks[i],
                    hidden: TapsIn::Host(&cur_rows[i]),
                })
                .collect();
            let outs = self.drafter.run_batch(&step_spans, wants, true)?;
            let step_ids = self.drafter.ids(outs[0])?;
            for i in 0..n {
                out[i].push(step_ids[i]);
            }
            if step + 1 < chain_n {
                prev_hid = self.drafter.read_floats(outs[1])?;
                for i in 0..n {
                    cur_rows[i] = prev_hid[i * hidden..(i + 1) * hidden].to_vec();
                }
            }
        }
        Ok(out)
    }
}

fn encode_round(hash: u32, cancelled: &[u32], admitted: &[Admitted]) -> Vec<u8> {
    let mut w = Writer(Vec::new());
    w.u32(BATCH_STEP_TAG)
        .u32(hash)
        .u32s(cancelled)
        .u32(admitted.len() as u32);
    for a in admitted {
        w.u32(a.slot)
            .u32(a.request.max_tokens as u32)
            .u32s(&a.request.tokens)
            .u32s(&a.request.stop);
    }
    w.0
}

fn decode_round(bytes: &[u8]) -> Option<(u32, Vec<u32>, Vec<Admitted>)> {
    let mut r = Reader(bytes);
    if r.u32()? != BATCH_STEP_TAG {
        return None;
    }
    let hash = r.u32()?;
    let cancelled = r.u32s()?;
    let n = r.u32()? as usize;
    let mut admitted = Vec::with_capacity(n);
    for _ in 0..n {
        let slot = r.u32()?;
        let max_tokens = r.u32()? as usize;
        let tokens = r.u32s()?;
        let stop = r.u32s()?;
        admitted.push(Admitted {
            slot,
            request: Request {
                tokens,
                max_tokens,
                stop,
                ..Request::default()
            },
            sink: None,
        });
    }
    Some((hash, cancelled, admitted))
}
