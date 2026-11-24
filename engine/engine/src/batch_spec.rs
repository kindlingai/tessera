//! Generic continuous-batching MTP speculative decoding over many slots.

use crate::mtp_draft::{MtpDrafter, MtpSpan};
use crate::schedule::Sink;
use crate::spec_loop::{TapsBuf, TapsIn, fnv_u32, tell_ranks};
use crate::speculate::accept_greedy;
use crate::{Engine, Error, Event, Reason, Request, Row};
use std::collections::VecDeque;
use tessera_engine_comm::{Reader, Writer};
use tessera_run::Error as RunError;

/// Wire tag identifying a batched speculative round header.
const BATCH_STEP_TAG: u32 = 2;

/// Default maximum rows per prefill chunk.
const DEFAULT_CHUNK: usize = 64;

/// One active request decoded inside a shared speculative batch.
struct ActiveSeq {
    slot: u32,
    request: Request,
    sink: Option<Sink>,
    /// Next target position (prompt length plus emitted tokens).
    start: usize,
    /// Last committed token, verified at row 0 next round.
    anchor: u32,
    /// Current round's proposed draft tokens.
    drafts: Vec<u32>,
    emitted: usize,
    /// Set on rank 0 when client drops the stream.
    cancelled: bool,
}

/// Surviving session state handed from verify to batched MTP draft.
struct DraftJob {
    active_idx: usize,
    slot: u32,
    old_start: u32,
    new_start: usize,
    committed: Vec<u32>,
    tap_row_off: usize,
    avail: usize,
}

/// Newly admitted session synchronized across ranks.
struct Admitted {
    slot: u32,
    request: Request,
    sink: Option<Sink>,
}

/// Multi-slot continuous-batching MTP speculative decoder.
pub struct BatchMtp<'a> {
    target: &'a mut Engine,
    drafter: &'a mut MtpDrafter,
    depth: usize,
    prefill_chunk: usize,
    max_ctx: usize,
    waiting: VecDeque<(Request, Sink)>,
    active: Vec<ActiveSeq>,
    free: Vec<u32>,
    hash: u32,
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
        let Some(width) = target.tap_width() else {
            return Err(Error::Taps);
        };
        if width != drafter.hidden() {
            return Err(Error::Rows);
        }
        let slots = target.slots().min(drafter.slots());
        let max_ctx = target.context().min(drafter.context());
        let free = (0..slots as u32).rev().collect();
        Ok(Self {
            target,
            drafter,
            depth,
            prefill_chunk: DEFAULT_CHUNK,
            max_ctx,
            waiting: VecDeque::new(),
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

    /// Whether any answer is waiting or decoding.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Runs one round: admit, prefill, shared verify, and batched draft.
    pub fn step(&mut self) -> Result<bool, Error> {
        if !self.busy() {
            return Ok(false);
        }
        let cancelled = self.collect_cancelled();
        let admitted = self.admit_local();
        if self.target.control.is_some() {
            let msg = encode_round(self.hash, &cancelled, &admitted);
            let _ = tell_ranks(self.target, &msg)?;
        }
        self.retire_slots(&cancelled);
        self.run_round(admitted)?;
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
            let Some(told) = tell_ranks(self.target, &mine)? else {
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
        for item in admitted {
            self.prefill_one(item)?;
        }
        self.active.sort_by_key(|s| s.slot);
        if self.active.is_empty() {
            return Ok(());
        }
        self.verify_and_draft()
    }

    fn prefill_one(&mut self, item: Admitted) -> Result<(), Error> {
        let Admitted {
            slot,
            request,
            mut sink,
        } = item;
        let n = request.tokens.len();
        let wants = self.drafter.wants();
        let mut pos0 = 0usize;
        while pos0 < n {
            let len = (n - pos0).min(self.prefill_chunk);
            let rows: Vec<Row> = (0..len)
                .map(|i| Row {
                    slot,
                    token: request.tokens[pos0 + i],
                    position: (pos0 + i) as u32,
                })
                .collect();
            let (ids, taps) = self.target.step_taps_local(&rows)?;
            let last_chunk = pos0 + len == n;
            if !last_chunk {
                let shifted = &request.tokens[pos0 + 1..pos0 + len + 1];
                self.drafter
                    .run_taps(slot, pos0 as u32, shifted, taps.as_in(), wants, true)?;
                pos0 += len;
                continue;
            }
            let anchor = *ids.last().expect("non-empty chunk");
            fnv_u32(&mut self.hash, anchor);
            let mut cancelled = false;
            let mut done = None;
            if request.stop.contains(&anchor) {
                done = Some(Reason::Stop);
            } else {
                if let Some(cb) = &mut sink
                    && !cb(Event::Token(anchor))
                {
                    cancelled = true;
                    let _ = cb(Event::Done(Reason::Stop));
                }
                if 1 >= request.max_tokens {
                    done = Some(Reason::Length);
                } else if n >= self.max_ctx {
                    done = Some(Reason::Context);
                }
            }
            if let Some(reason) = done {
                if !cancelled && let Some(cb) = &mut sink {
                    let _ = cb(Event::Done(reason));
                }
                self.free.push(slot);
                return Ok(());
            }
            if self.target.control.is_none() && cancelled {
                self.free.push(slot);
                return Ok(());
            }
            let mut shifted = Vec::with_capacity(len);
            shifted.extend_from_slice(&request.tokens[pos0 + 1..]);
            shifted.push(anchor);
            let rem = request.max_tokens - 1;
            let ctx_rem = self.max_ctx - n;
            let avail = self
                .depth
                .min(rem.saturating_sub(1))
                .min(ctx_rem.saturating_sub(1));
            let drafts = if avail > 0 {
                let job = DraftJob {
                    active_idx: 0,
                    slot,
                    old_start: pos0 as u32,
                    new_start: n,
                    committed: shifted,
                    tap_row_off: 0,
                    avail,
                };
                self.draft_batch(&[job], &taps)?.pop().unwrap_or_default()
            } else {
                Vec::new()
            };
            self.active.push(ActiveSeq {
                slot,
                request,
                sink,
                start: n,
                anchor,
                drafts,
                emitted: 1,
                cancelled,
            });
            return Ok(());
        }
        Ok(())
    }

    fn verify_and_draft(&mut self) -> Result<(), Error> {
        let s_count = self.active.len();
        let mut rows = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(s_count);
        for seq in &self.active {
            let rem = seq.request.max_tokens.saturating_sub(seq.emitted);
            let ctx_rem = self.max_ctx.saturating_sub(seq.start);
            let max_d = rem.saturating_sub(1).min(ctx_rem.saturating_sub(1));
            let k = seq.drafts.len().min(max_d);
            let off = rows.len();
            rows.push(Row {
                slot: seq.slot,
                token: seq.anchor,
                position: seq.start as u32,
            });
            for j in 0..k {
                rows.push(Row {
                    slot: seq.slot,
                    token: seq.drafts[j],
                    position: (seq.start + 1 + j) as u32,
                });
            }
            spans.push((off, k));
        }
        let (ids, taps) = self.target.step_taps_local(&rows)?;
        let ranked = self.target.control.is_some();
        let mut jobs: Vec<DraftJob> = Vec::with_capacity(s_count);
        let mut finished: Vec<bool> = vec![false; s_count];
        for (i, &(off, k)) in spans.iter().enumerate() {
            let seq = &mut self.active[i];
            let (accepted, bonus) = accept_greedy(&seq.drafts[..k], &ids[off..off + 1 + k]);
            fnv_u32(&mut self.hash, accepted as u32);
            fnv_u32(&mut self.hash, bonus);
            let commit = 1 + accepted;
            let mut committed = Vec::with_capacity(commit);
            committed.extend_from_slice(&seq.drafts[..accepted]);
            committed.push(bonus);
            let mut done = None;
            for &tok in &committed {
                if seq.request.stop.contains(&tok) {
                    done = Some(Reason::Stop);
                    break;
                }
                seq.emitted += 1;
                if !seq.cancelled
                    && let Some(cb) = &mut seq.sink
                    && !cb(Event::Token(tok))
                {
                    seq.cancelled = true;
                    let _ = cb(Event::Done(Reason::Stop));
                }
                if seq.emitted >= seq.request.max_tokens {
                    done = Some(Reason::Length);
                    break;
                }
            }
            let old_start = seq.start as u32;
            seq.start += commit;
            seq.anchor = bonus;
            seq.drafts.clear();
            if done.is_none() && seq.start >= self.max_ctx {
                done = Some(Reason::Context);
            }
            if let Some(reason) = done {
                if !seq.cancelled
                    && let Some(cb) = &mut seq.sink
                {
                    let _ = cb(Event::Done(reason));
                }
                finished[i] = true;
                continue;
            }
            if !ranked && seq.cancelled {
                finished[i] = true;
                continue;
            }
            let rem = seq.request.max_tokens.saturating_sub(seq.emitted);
            let ctx_rem = self.max_ctx.saturating_sub(seq.start);
            let avail = self
                .depth
                .min(rem.saturating_sub(1))
                .min(ctx_rem.saturating_sub(1));
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
        if !jobs.is_empty() {
            let drafts_all = self.draft_batch(&jobs, &taps)?;
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

    /// Runs batched first-form MTP plus shared chain steps for `jobs`.
    fn draft_batch(&mut self, jobs: &[DraftJob], taps: &TapsBuf) -> Result<Vec<Vec<u32>>, Error> {
        let n = jobs.len();
        let chain_n = jobs.iter().map(|j| j.avail).min().unwrap_or(0);
        if chain_n == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let hidden = self.drafter.hidden();
        let wants = self.drafter.wants();
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
        let spans: Vec<MtpSpan<'_>> = jobs
            .iter()
            .map(|j| {
                let h = match taps {
                    TapsBuf::Device { addr, .. } => TapsIn::Device {
                        addr: *addr + (j.tap_row_off * hidden * 4) as u64,
                        rows: j.committed.len(),
                    },
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
