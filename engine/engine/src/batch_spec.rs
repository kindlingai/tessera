//! Generic continuous-batching MTP and dual speculative decoding.

use crate::batch_dual::DualPeer;
use crate::dual::DualState;
use crate::mtp_draft::{MtpDrafter, MtpJob, draft_mtp_batch, take_cols};
use crate::spec_loop::{FNV_SEED, fnv_u32, span};
use std::collections::VecDeque;
use tessera_engine_hooks::Kept;
use tessera_engine_hooks::Sink;
use tessera_engine_hooks::sample::{Sampler, Target, accept, take};
use tessera_engine_hooks::{Error, Event, Hooks, Reason, Request, Row, TapsBuf, TapsIn, VerifyOut};
use tessera_run::Error as RunError;

pub(crate) mod admit;
#[cfg(test)]
mod admit_test;

/// Default maximum rows per prefill chunk.
pub(crate) const DEFAULT_CHUNK: usize = tessera_engine_settings::FAST_MAX_ROWS;

mod mode;
#[cfg(test)]
mod mode_test;
mod wire;
use wire::{decode_round, encode_round};

/// Shared coordinator and follower lifecycle for batched speculative loops.
pub(crate) trait RoundHost<'a> {
    fn core(&mut self) -> &mut BatchCore<'a>;
    fn is_busy(&self) -> bool;
    fn collect_cancelled(&self) -> Vec<u32>;
    fn admit_local(&mut self) -> Vec<Admitted>;
    fn retire_slots(&mut self, slots: &[u32]);
    fn run_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error>;
}

/// Executes one coordinator round on `host`.
pub(crate) fn drive_step<'a>(host: &mut impl RoundHost<'a>) -> Result<bool, Error> {
    host.core().target.settle()?;
    if !host.is_busy() {
        return Ok(false);
    }
    let cancelled = host.collect_cancelled();
    let admitted = host.admit_local();
    let hash = host.core().hash;
    if host.core().target.rank().is_some() {
        let msg = encode_round(hash, &cancelled, &admitted);
        let _ = host.core().target.told(&msg)?;
    }
    host.retire_slots(&cancelled);
    host.run_round(admitted, false)?;
    host.core().target.settle()?;
    Ok(true)
}

/// Runs the follower rank loop on `host`.
pub(crate) fn drive_follow<'a>(host: &mut impl RoundHost<'a>) -> Result<(), Error> {
    loop {
        let mine = host
            .core()
            .target
            .take_held()
            .map_or_else(Vec::new, |p| p.to_bytes());
        let Some(told) = host.core().target.told(&mine)? else {
            return Ok(());
        };
        let Some((want_hash, cancelled, admitted)) = decode_round(&told) else {
            return Ok(());
        };
        if want_hash != host.core().hash {
            return Err(RunError::OutputsDisagree.into());
        }
        host.retire_slots(&cancelled);
        host.run_round(admitted, true)?;
    }
}

/// Shared target, slot pool, and prefix state for batched decoders.
pub(crate) struct BatchCore<'a> {
    pub(crate) target: &'a mut dyn Hooks,
    pub(crate) prefill_chunk: usize,
    pub(crate) max_ctx: usize,
    pub(crate) waiting: VecDeque<(Request, Sink)>,
    pub(crate) free: Vec<u32>,
    pub(crate) kept: Option<Kept>,
    pub(crate) admit: Box<dyn admit::Admit>,
    pub(crate) hash: u32,
}

impl<'a> BatchCore<'a> {
    pub(crate) fn new(
        target: &'a mut dyn Hooks,
        slots: usize,
        max_ctx: usize,
        paged: bool,
    ) -> Self {
        if paged && let Some(lanes) = target.lanes_mut() {
            lanes.reset();
        }
        let kept = paged
            .then(|| {
                target
                    .lanes()
                    .map(|l| Kept::paged(slots, max_ctx, l.pages(), l.page))
            })
            .flatten();
        let free = if kept.is_some() {
            Vec::new()
        } else {
            (0..slots as u32).rev().collect()
        };
        Self {
            target,
            prefill_chunk: DEFAULT_CHUNK,
            max_ctx,
            waiting: VecDeque::new(),
            free,
            kept,
            admit: Box::new(admit::Headroom::default()),
            hash: FNV_SEED,
        }
    }

    pub(crate) fn reuse(&mut self, slots: usize) -> bool {
        if self.target.stateful() {
            return false;
        }
        if self.kept.is_none() {
            self.kept = Some(Kept::new(slots, self.max_ctx));
            self.free.clear();
        }
        true
    }

    /// Restores extra prefix pages from the KV store past `skip`.
    pub(crate) fn restore_store(
        &mut self,
        slot: u32,
        tokens: &[u32],
        skip: usize,
    ) -> Result<usize, Error> {
        self.target
            .restore_kv(self.kept.as_mut(), slot, tokens, skip, false)
    }

    pub(crate) fn retire(&mut self, slot: u32, history: Vec<u32>) {
        self.target
            .spill_kv(self.kept.as_ref(), slot, &history, false);
        match &mut self.kept {
            Some(kept) => kept.finish(slot, history),
            None => self.free.push(slot),
        }
    }

    /// Picks a slot for the front waiting request.
    pub(crate) fn admit_front(
        &mut self,
        (gated, busy, charge): (bool, bool, bool),
        pick: impl FnOnce(&mut Kept, &[u32]) -> Option<(u32, usize)>,
    ) -> Option<(u32, usize)> {
        let Some(kept) = &mut self.kept else {
            return self.free.pop().map(|slot| (slot, 0));
        };
        let (front, _) = self.waiting.front()?;
        let extra = if charge { front.tokens.len() } else { 0 };
        if gated
            && !self
                .admit
                .admit(admit::Ask::of(kept.totals().pool, busy, extra))
        {
            return None;
        }
        pick(kept, &front.tokens)
    }

    pub(crate) fn pop_admitted(&mut self, slot: u32, skip: usize) -> Admitted {
        let (mut request, sink) = self.waiting.pop_front().expect("waiting request");
        request.sampling = request.sampling.seeded();
        Admitted {
            slot,
            skip,
            request,
            sink: Some(sink),
        }
    }

    /// Builds `ActiveSeq` after picking `anchor` for a finished prompt.
    pub(crate) fn accept_prefill(
        &mut self,
        slot: u32,
        request: Request,
        sink: Option<Sink>,
        sampler: Sampler,
        anchor: u32,
    ) -> (ActiveSeq, bool) {
        let n = request.tokens.len();
        fnv_u32(&mut self.hash, anchor);
        let history = request.tokens.clone();
        let mut act = ActiveSeq {
            slot,
            request,
            sink,
            start: n,
            anchor,
            drafts: Vec::new(),
            emitted: 0,
            cancelled: false,
            solo: None,
            d_tail: Vec::new(),
            history,
            mtp_valid: 0,
            sampler,
        };
        let done = act.commit(&[anchor], self.max_ctx);
        let ended = done.is_some() || (self.target.rank().is_none() && act.cancelled);
        (act, ended)
    }

    /// Runs the final prefill chunk and builds `ActiveSeq`.
    pub(crate) fn finish_prefill(
        &mut self,
        slot: u32,
        request: Request,
        sink: Option<Sink>,
        rows: &[Row],
        mtp_valid: usize,
        push_anchor: bool,
    ) -> Result<(ActiveSeq, TapsBuf, bool), Error> {
        let mut sampler = Sampler::new(request.sampling);
        let (anchor, taps) = self.target.prefill_pick(rows, &mut sampler)?;
        let (mut act, ended) = self.accept_prefill(slot, request, sink, sampler, anchor);
        act.mtp_valid = mtp_valid;
        if push_anchor {
            act.history.push(anchor);
        }
        Ok((act, taps, ended))
    }
}

/// Builds verify rows, spans, and sampling flag.
pub(crate) fn verify_rows<T>(
    active: &[T],
    max_ctx: usize,
    seq_of: impl Fn(&T) -> &ActiveSeq,
) -> (Vec<Row>, Vec<(usize, usize)>, bool) {
    let mut rows = Vec::new();
    let mut spans = Vec::with_capacity(active.len());
    let mut sampling = false;
    for item in active {
        let seq = seq_of(item);
        let k = seq.drafts.len().min(seq.room(max_ctx));
        spans.push((rows.len(), k));
        seq.verify(&mut rows, k);
        sampling |= !seq.sampler.greedy();
    }
    (rows, spans, sampling)
}

/// Removes finished `slots` from `active` and returns them.
pub(crate) fn retire_from<T>(
    active: &mut Vec<T>,
    slots: &[u32],
    slot_of: impl Fn(&T) -> u32,
) -> Vec<T> {
    active
        .extract_if(.., |t| slots.contains(&slot_of(t)))
        .collect()
}

/// Removes `active` entries marked true in `finished` and returns them.
pub(crate) fn retire_finished<T>(active: &mut Vec<T>, finished: &[bool]) -> Vec<T> {
    let mut orig = 0;
    active
        .extract_if(.., |_| {
            orig += 1;
            finished[orig - 1]
        })
        .collect()
}

/// Prefill chunk length, keeping sampled final chunks one row.
pub(crate) fn prefill_len(n: usize, pos: usize, chunk: usize, greedy: bool) -> usize {
    let mut len = (n - pos).min(chunk);
    if !greedy && pos + len == n && len > 1 {
        len = n - 1 - pos;
    }
    len
}

/// Emits a prefix-reuse event when `skip` rows were kept.
pub(crate) fn notify_kept(sink: &mut Option<Sink>, skip: usize) {
    if skip > 0
        && let Some(cb) = sink
    {
        let _ = cb(Event::Kept(skip));
    }
}

/// Validates and appends `request` to `waiting`, or finishes at once.
pub(crate) fn queue_request(
    waiting: &mut VecDeque<(Request, Sink)>,
    max_ctx: usize,
    request: Request,
    mut sink: Sink,
) {
    if request.tokens.is_empty() || request.tokens.len() > max_ctx {
        sink(Event::Done(Reason::Context));
        return;
    }
    if request.max_tokens == 0 {
        sink(Event::Done(Reason::Length));
        return;
    }
    waiting.push_back((request, sink));
}

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
    pub(crate) solo: Option<DualState>,
    /// Rolling recent DFlash taps for tail promotion.
    pub(crate) d_tail: Vec<f32>,
    /// Prompt plus emitted tokens for prefix reuse.
    pub(crate) history: Vec<u32>,
    /// Rows written into the MTP drafter KV cache.
    pub(crate) mtp_valid: usize,
    pub(crate) sampler: Sampler,
}

impl ActiveSeq {
    /// Applies target verify outputs, emitting tokens and returning finish status.
    pub(crate) fn apply_verify(
        &mut self,
        k: usize,
        off: usize,
        out: &mut VerifyOut,
        max_ctx: usize,
        hash: &mut u32,
    ) -> (usize, Vec<u32>, Option<Reason>) {
        let (accepted, bonus) = match out {
            VerifyOut::Ids(ids) => accept(&self.drafts[..k], Target::Ids(&ids[off..off + 1 + k])),
            VerifyOut::Logits(logits, vocab) => {
                let lo = off * *vocab;
                let hi = (off + 1 + k) * *vocab;
                self.sampler
                    .verify(&self.drafts[..k], &mut logits[lo..hi], *vocab, &[], 0)
            }
        };
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
pub(crate) struct Admitted {
    pub(crate) slot: u32,
    pub(crate) skip: usize,
    pub(crate) request: Request,
    pub(crate) sink: Option<Sink>,
}

/// Admitted sequence advancing through chunked prompt prefill.
pub(crate) struct PrefillSeq {
    pub(crate) slot: u32,
    pub(crate) request: Request,
    pub(crate) sink: Option<Sink>,
    pub(crate) pos: usize,
    pub(crate) mtp_valid: usize,
    pub(crate) solo_mode: bool,
    pub(crate) drop_before: usize,
    pub(crate) m_start_init: u32,
    pub(crate) d_buf: Vec<f32>,
    pub(crate) m_tok_buf: Vec<u32>,
    pub(crate) m_tap_buf: Vec<f32>,
}

/// Multi-slot continuous-batching MTP and optional DFlash decoder.
pub struct BatchMtp<'a> {
    pub(crate) core: BatchCore<'a>,
    pub(crate) drafter: &'a mut MtpDrafter,
    pub(crate) dual: Option<DualPeer<'a>>,
    pub(crate) depth: usize,
    pub(crate) prefilling: VecDeque<PrefillSeq>,
    pub(crate) active: Vec<ActiveSeq>,
    pub(crate) reuse: bool,
    mode: Box<dyn mode::Mode>,
}

impl<'a> BatchMtp<'a> {
    pub(crate) fn init(
        target: &'a mut dyn Hooks,
        drafter: &'a mut MtpDrafter,
        dual: Option<DualPeer<'a>>,
        depth: usize,
        max_ctx: usize,
    ) -> Result<Self, Error> {
        if let Some(lanes) = drafter.lanes_mut() {
            lanes.reset();
        }
        if let (Some(tl), Some(dl)) = (target.lanes(), drafter.lanes())
            && (tl.page != dl.page || tl.pages() != dl.pages())
        {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        Ok(Self {
            core: BatchCore::new(target, slots, max_ctx, true),
            drafter,
            dual,
            depth,
            prefilling: VecDeque::new(),
            active: Vec::new(),
            reuse: false,
            mode: Box::new(mode::Lone::default()),
        })
    }

    /// Pairs a multi-slot tapped target engine with its MTP drafter.
    pub fn new(
        target: &'a mut dyn Hooks,
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
        let max_ctx = target.context().min(drafter.context());
        Self::init(target, drafter, None, depth, max_ctx)
    }
}

/// Public driver methods shared by the batched speculative decoders.
macro_rules! batch_methods {
    ($ty:ident $(, $flag:ident)?) => {
        impl $ty<'_> {
            /// Retains finished slots for prefix reuse.
            pub fn reuse(&mut self) {
                let slots = self.core.target.slots().min(self.drafter.slots());
                let _on = self.core.reuse(slots);
                $(self.$flag = _on;)?
            }

            /// Sets the maximum prompt rows per prefill step.
            pub fn with_prefill_chunk(mut self, rows: usize) -> Self {
                self.core.prefill_chunk = rows.max(1);
                self
            }

            /// Queues an answer. Unfit requests finish at once.
            pub fn submit(&mut self, request: Request, sink: Sink) {
                queue_request(&mut self.core.waiting, self.core.max_ctx, request, sink);
            }

            /// Whether any answer is waiting, prefilling, or decoding.
            pub fn busy(&self) -> bool {
                !self.core.waiting.is_empty() || !self.prefilling.is_empty() || !self.active.is_empty()
            }

            /// Runs one round: admit, prefill, verify, and draft.
            pub fn step(&mut self) -> Result<bool, Error> {
                drive_step(self)
            }

            /// Follower rank loop executing rank 0's batched rounds.
            pub fn follow(&mut self) -> Result<(), Error> {
                drive_follow(self)
            }
        }
    };
}
pub(crate) use batch_methods;

batch_methods!(BatchMtp, reuse);

impl<'a> RoundHost<'a> for BatchMtp<'a> {
    fn core(&mut self) -> &mut BatchCore<'a> {
        &mut self.core
    }
    fn is_busy(&self) -> bool {
        self.busy()
    }
    fn collect_cancelled(&self) -> Vec<u32> {
        self.active
            .iter()
            .filter(|s| s.cancelled)
            .map(|s| s.slot)
            .collect()
    }
    fn admit_local(&mut self) -> Vec<Admitted> {
        if self.active.iter().any(|s| s.cancelled) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let can_reuse = self.can_reuse();
        let page = self.core.target.lanes().map_or(0, |l| l.page);
        while !self.core.waiting.is_empty() {
            let busy = !self.active.is_empty() || !self.prefilling.is_empty() || !out.is_empty();
            let Some((slot, skip)) = self
                .core
                .admit_front((page > 0, busy, !can_reuse), |kept, toks| {
                    crate::batch_dual::admit_one(kept, toks, can_reuse, page)
                })
            else {
                break;
            };
            out.push(self.core.pop_admitted(slot, skip));
        }
        out
    }
    fn retire_slots(&mut self, slots: &[u32]) {
        for s in retire_from(&mut self.active, slots, |s| s.slot) {
            self.retire_slot(s.slot, s.history, s.mtp_valid);
        }
    }
    fn run_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        self.exec_round(admitted, follower)
    }
}

impl BatchMtp<'_> {
    pub(crate) fn can_reuse(&self) -> bool {
        self.reuse && (self.core.target.lanes().is_none() || self.drafter.lanes().is_some())
    }

    pub(crate) fn retire_slot(&mut self, slot: u32, mut history: Vec<u32>, mtp_valid: usize) {
        if self.can_reuse() {
            history.truncate(mtp_valid.min(self.core.max_ctx));
            self.core.retire(slot, history);
        } else {
            self.core.retire(slot, Vec::new());
        }
    }

    fn exec_round(&mut self, admitted: Vec<Admitted>, follower: bool) -> Result<(), Error> {
        if !admitted.is_empty() {
            if self.active.len() == 1 && self.active[0].solo.is_some() {
                self.demote_solo()?;
            }
            if self.prefilling.len() == 1 && self.prefilling[0].solo_mode {
                self.demote_prefill()?;
            }
            let solo_admit = self.dual.is_some() && self.mode.solo(self.flight(), admitted.len());
            let can_reuse = self.can_reuse();
            let page = self.core.target.lanes().map_or(0, |l| l.page);
            for mut item in admitted {
                if follower && let Some(kept) = &mut self.core.kept {
                    let Some((slot, _)) =
                        crate::batch_dual::admit_one(kept, &item.request.tokens, can_reuse, page)
                    else {
                        return Err(Error::RowPlace);
                    };
                    debug_assert_eq!(slot, item.slot);
                }
                notify_kept(&mut item.sink, item.skip);
                let n = item.request.tokens.len();
                let win = solo_admit
                    .then(|| self.dual.as_ref()?.dflash.sliding_window())
                    .flatten();
                let drop_before = win.map_or(item.skip, |w| n.saturating_sub(w).max(item.skip));
                self.prefilling.push_back(PrefillSeq {
                    slot: item.slot,
                    request: item.request,
                    sink: item.sink,
                    pos: item.skip,
                    mtp_valid: item.skip,
                    solo_mode: solo_admit,
                    drop_before,
                    m_start_init: item.skip as u32,
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
        let len = prefill_len(
            n,
            pos0,
            self.core.prefill_chunk,
            seq.request.sampling.greedy(),
        );
        let wants = self.drafter.wants();
        let dual_widths = self.dual.as_ref().map(|p| (p.d_width, p.m_width));
        let mut rows = Vec::with_capacity(len);
        let chunk = &seq.request.tokens[pos0..pos0 + len];
        span(&mut rows, seq.slot, pos0, chunk.iter().copied());
        self.ensure_pages(&rows)?;
        if pos0 + len < n {
            let taps = self.core.target.step_taps_only(&rows)?;
            let shifted = &seq.request.tokens[pos0 + 1..pos0 + len + 1];
            if let Some((d_w, m_w)) = dual_widths {
                if seq.solo_mode {
                    if pos0 + len <= seq.drop_before {
                        let (sl, p0, end) = (seq.slot, pos0 as u32, pos0 + len);
                        self.run_mtp_prefill_chunk(sl, p0, shifted, &taps, d_w, m_w)?;
                        (seq.m_start_init, seq.mtp_valid) = (end as u32, end);
                    } else {
                        let (dt, mt) = self.split_dual_taps(taps, len)?;
                        let skip = seq.drop_before.saturating_sub(pos0);
                        seq.d_buf.extend_from_slice(&dt[skip * d_w..]);
                        seq.m_tok_buf.extend_from_slice(shifted);
                        seq.m_tap_buf.extend_from_slice(&mt);
                    }
                } else {
                    self.prefill_tail(&mut seq.d_buf, &taps, len)?;
                    self.run_mtp_prefill_chunk(seq.slot, pos0 as u32, shifted, &taps, d_w, m_w)?;
                    seq.mtp_valid = pos0 + len;
                }
            } else {
                self.drafter
                    .run_taps(seq.slot, pos0 as u32, shifted, taps.as_in(), wants, true)?;
                seq.mtp_valid = pos0 + len;
            }
            seq.pos = pos0 + len;
            self.prefilling.push_front(seq);
            return Ok(());
        }
        let (mut act, taps, ended) = self.core.finish_prefill(
            seq.slot,
            seq.request,
            seq.sink,
            &rows,
            seq.mtp_valid,
            true,
        )?;
        if ended {
            self.retire_slot(act.slot, act.history, act.mtp_valid);
            return Ok(());
        }
        let mut shifted = Vec::with_capacity(len);
        shifted.extend_from_slice(&act.request.tokens[pos0 + 1..]);
        shifted.push(act.anchor);
        if seq.solo_mode {
            let (dt, mt) = self.split_dual_taps(taps, len)?;
            let skip = seq.drop_before.saturating_sub(pos0);
            seq.d_buf
                .extend_from_slice(&dt[skip * dual_widths.expect("dual widths").0..]);
            seq.m_tok_buf.extend_from_slice(&shifted);
            seq.m_tap_buf.extend_from_slice(&mt);
            let selector = self.dual.as_ref().expect("dual peer").selector.fresh();
            act.solo = Some(DualState::new(
                selector,
                (seq.drop_before, 0, seq.d_buf),
                (seq.m_start_init, seq.m_tok_buf, seq.m_tap_buf),
            ));
            self.active.push(act);
            let idx = self.active.len() - 1;
            self.solo_draft(idx)?;
            return Ok(());
        }
        self.prefill_tail(&mut seq.d_buf, &taps, len)?;
        let avail = self.depth.min(act.room(self.core.max_ctx));
        if avail >= 2 && self.drafter.lanes().is_some() {
            self.ensure_slot_pages(act.slot, n + avail - 1)?;
        }
        act.drafts = if avail > 0 {
            let job = MtpJob {
                slot: act.slot,
                old_start: pos0 as u32,
                new_start: n,
                tokens: &shifted,
                tap_row_off: 0,
            };
            let got = self
                .draft_jobs(&[job], &taps, avail)?
                .pop()
                .unwrap_or_default();
            act.mtp_valid = n;
            got
        } else {
            Vec::new()
        };
        act.d_tail = seq.d_buf;
        self.active.push(act);
        Ok(())
    }

    fn verify_and_draft(&mut self) -> Result<(), Error> {
        let s_count = self.active.len();
        let (rows, spans, sampling) = verify_rows(&self.active, self.core.max_ctx, |s| s);
        self.ensure_pages(&rows)?;
        let (mut out, raw_taps) = self.core.target.step_verify_local(&rows, sampling)?;
        let dual_info = self.dual.as_ref().map(|p| (p.d_width, p.block));
        let ranked = self.core.target.rank().is_some();
        let mut jobs: Vec<DraftJob> = Vec::with_capacity(s_count);
        let mut finished: Vec<bool> = vec![false; s_count];
        for (i, &(off, k)) in spans.iter().enumerate() {
            let old_start = self.active[i].start as u32;
            let (_, committed, done) = self.active[i].apply_verify(
                k,
                off,
                &mut out,
                self.core.max_ctx,
                &mut self.core.hash,
            );
            let seq = &mut self.active[i];
            seq.history.extend_from_slice(&committed);
            if done.is_some() || (!ranked && seq.cancelled) {
                finished[i] = true;
                continue;
            }
            let avail = self.depth.min(seq.room(self.core.max_ctx));
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
        let flight = mode::Flight {
            active: survivors,
            prefilling: self.prefilling.len(),
        };
        let promote = jobs.len() == 1
            && dual_info.is_some_and(|(_, block)| {
                let seq = &self.active[jobs[0].active_idx];
                let left = seq.request.max_tokens.saturating_sub(seq.emitted);
                seq.start + block <= self.core.max_ctx && self.mode.promote(flight, left)
            });
        if promote {
            let job = jobs.remove(0);
            let (d_all, m_all) = self.split_dual_taps(raw_taps, rows.len())?;
            let peer = self.dual.as_ref().expect("dual peer");
            let seq = &mut self.active[job.active_idx];
            peer.push_rows(
                &mut seq.d_tail,
                &d_all,
                job.tap_row_off,
                job.committed.len(),
            );
            self.promote_solo(job, &m_all)?;
        } else if !jobs.is_empty() {
            if let Some(peer) = &self.dual
                && self.mode.tail(survivors + self.prefilling.len())
            {
                let d_all = self.read_dual_d_taps(&raw_taps, rows.len())?;
                for job in &jobs {
                    let seq = &mut self.active[job.active_idx];
                    peer.push_rows(
                        &mut seq.d_tail,
                        &d_all,
                        job.tap_row_off,
                        job.committed.len(),
                    );
                }
            }
            let chain_n = jobs.iter().map(|j| j.avail).min().unwrap_or(0);
            let mtp_jobs: Vec<MtpJob<'_>> = jobs
                .iter()
                .map(|j| MtpJob {
                    slot: j.slot,
                    old_start: j.old_start,
                    new_start: j.new_start,
                    tokens: &j.committed,
                    tap_row_off: j.tap_row_off,
                })
                .collect();
            if chain_n >= 2 && self.drafter.lanes().is_some() {
                for j in &jobs {
                    self.ensure_slot_pages(j.slot, j.new_start + chain_n - 1)?;
                }
            }
            let drafts_all = self.draft_jobs(&mtp_jobs, &raw_taps, chain_n)?;
            for (job, drafts) in jobs.into_iter().zip(drafts_all) {
                let seq = &mut self.active[job.active_idx];
                seq.drafts = drafts;
                seq.mtp_valid = job.new_start;
            }
        }
        for seq in retire_finished(&mut self.active, &finished) {
            self.retire_slot(seq.slot, seq.history, seq.mtp_valid);
        }
        Ok(())
    }

    /// Runs the batched MTP draft for `jobs` over `taps`.
    fn draft_jobs(
        &mut self,
        jobs: &[MtpJob<'_>],
        taps: &TapsBuf,
        depth: usize,
    ) -> Result<Vec<Vec<u32>>, Error> {
        let stride = self
            .dual
            .as_ref()
            .map(|p| (p.d_width, p.d_width + p.m_width));
        draft_mtp_batch(self.core.target, self.drafter, jobs, taps, stride, depth)
    }

    /// Keeps the DFlash tail of a prefill chunk.
    fn prefill_tail(&self, buf: &mut Vec<f32>, taps: &TapsBuf, rows: usize) -> Result<(), Error> {
        let Some(peer) = &self.dual else {
            return Ok(());
        };
        if self
            .mode
            .tail(self.active.len() + self.prefilling.len() + 1)
        {
            let dt = self.read_dual_d_taps(taps, rows)?;
            peer.push_rows(buf, &dt, 0, rows);
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
                let mt = take_cols(v, tokens.len(), stride, d_w, m_w);
                self.drafter
                    .run_taps(slot, pos0, tokens, TapsIn::Host(&mt), wants, true)?;
            }
        }
        Ok(())
    }
}
