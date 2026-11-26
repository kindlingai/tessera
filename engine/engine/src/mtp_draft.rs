//! Native MTP speculative decoding: one-layer MLA drafter and loop.

use crate::ranks::open_rank;
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{D2dCopy, SpecReq, TapsBuf, TapsIn, run_speculate, spec_entry};
use crate::speculate::{DraftCore, draft_methods};
use crate::{Engine, Error, Event, Place, SpecStats};
use tessera_engine_comm::Transport;
use tessera_graph::DimName::{Context, Hidden, Rows, Segments, Seqs};
use tessera_graph::{Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Recipe, Settings, Slot, Wants, Weight};
use tessera_models::mtp;
use tessera_run::{Run, open};

/// One session's rows fed to a batched MTP launch.
pub(crate) struct MtpSpan<'a> {
    pub slot: u32,
    pub start_pos: u32,
    pub tokens: &'a [u32],
    pub hidden: TapsIn<'a>,
}

/// One sequence job handed to batched MTP drafting.
pub(crate) struct MtpJob<'a> {
    pub slot: u32,
    pub old_start: u32,
    pub new_start: usize,
    pub tokens: &'a [u32],
    pub tap_row_off: usize,
}

/// Pads `v` to `filled` by repeating its last item.
fn pad_tail<T: Copy>(v: &mut Vec<T>, filled: usize) {
    if let Some(&last) = v.last() {
        v.resize(filled, last);
    }
}

/// Extracts `width` floats at `col_off` from each strided row.
pub(crate) fn take_cols(
    src: &[f32],
    rows: usize,
    stride: usize,
    col_off: usize,
    width: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        let lo = r * stride + col_off;
        out.extend_from_slice(&src[lo..lo + width]);
    }
    out
}

/// Runs an MTP one-layer MLA drafter with kept KV caches.
pub struct MtpDrafter {
    core: DraftCore,
    hidden: usize,
}

draft_methods!(MtpDrafter);

impl MtpDrafter {
    /// Opens a compiled runner for an MTP stepper graph.
    pub fn open(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
    ) -> Result<Self, Error> {
        let run = |plan| Ok(open(plan, settings)?);
        Self::placed(graph, context, recipe, settings, None, run)
    }

    /// Opens one rank of a TP-sharded MTP drafter.
    pub fn open_ranked<T: Transport + 'static>(
        graph: Graph,
        context: usize,
        (recipe, settings): (&Recipe, &Settings),
        mut place: Place,
        sums: T,
    ) -> Result<Self, Error> {
        place.rank = sums.rank();
        let grid = place.grid.take();
        place.settle(grid.as_deref());
        let run = |plan| open_rank(plan, settings, sums, grid);
        Self::placed(graph, context, recipe, settings, Some(&place), run)
    }

    fn placed(
        graph: Graph,
        context: usize,
        recipe: &Recipe,
        settings: &Settings,
        place: Option<&Place>,
        run: impl FnOnce(tessera_lower::Plan) -> Result<Box<dyn Run>, Error>,
    ) -> Result<Self, Error> {
        if context < 2 {
            return Err(Error::Rows);
        }
        tessera_run::configure(settings);
        let graph = match place {
            Some(place) => {
                let shares = recipe.pins.shares(&graph);
                let sharded = mtp::shard(&graph, &place.parts, &shares, place.moe_tp)?;
                place.log_whole(&graph, "mtp ", &sharded.whole);
                place.resolve(sharded.graph, &recipe.sums)
            }
            None => graph,
        };
        let hidden = graph
            .dims
            .iter()
            .find(|d| d.name == Hidden)
            .map(|d| d.size.max() as usize)
            .ok_or(Error::Taps)?;
        let wants = Wants::roles(&graph, 0, &[Role::Tokens, Role::Taps]);
        let core = DraftCore::open(
            graph, context, 1, wants, recipe, settings, place, "mtp", run,
        )?;
        Ok(Self { core, hidden })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Number of cached plan hits in the MTP drafter.
    pub fn plan_hits(&self) -> usize {
        self.core.plans.hits
    }

    pub(crate) fn wants(&self) -> Wants {
        Wants::roles(&self.core.graph, 0, &[Role::Tokens, Role::Taps])
    }

    fn exec_mtp(
        &mut self,
        dims: [usize; 3],
        feeds: [Option<Tensor>; 5],
        d2d: &[D2dCopy],
        wants: Wants,
        wait: bool,
    ) -> Result<Vec<Slot>, Error> {
        let [filled, max_end, segments] = dims;
        let sizes = [
            (Seqs, self.core.slots as u64),
            (Rows, filled as u64),
            (Context, crate::context_bucket(max_end, self.core.context)),
            (Segments, segments as u64),
        ];
        self.core.exec_tap(&feeds, d2d, &sizes, wants, wait)
    }

    pub(crate) fn run_taps(
        &mut self,
        slot: u32,
        start_pos: u32,
        tokens: &[u32],
        hidden: TapsIn<'_>,
        wants: Wants,
        wait: bool,
    ) -> Result<Vec<Slot>, Error> {
        let span = MtpSpan {
            slot,
            start_pos,
            tokens,
            hidden,
        };
        self.run_batch(&[span], wants, wait)
    }

    /// Runs one batched MTP pass across session spans.
    pub(crate) fn run_batch(
        &mut self,
        spans: &[MtpSpan<'_>],
        wants: Wants,
        wait: bool,
    ) -> Result<Vec<Slot>, Error> {
        let m: usize = spans.iter().map(|s| s.tokens.len()).sum();
        if m == 0 {
            return Err(Error::Rows);
        }
        let mut max_end = 0usize;
        for s in spans {
            if s.tokens.is_empty() {
                return Err(Error::Rows);
            }
            let end = s.start_pos as usize + s.tokens.len();
            if s.slot as usize >= self.core.slots || end > self.core.context {
                return Err(Error::RowPlace);
            }
            max_end = max_end.max(end);
        }
        let filled = crate::row_bucket(m).min(self.core.context);
        let mut tok_buf = Vec::with_capacity(filled);
        let mut pos_buf = Vec::with_capacity(filled);
        let mut slot_buf = Vec::with_capacity(filled);
        let mut scale_buf = Vec::with_capacity(filled);
        for s in spans {
            for (i, &t) in s.tokens.iter().enumerate() {
                let p = s.start_pos + i as u32;
                tok_buf.push(t);
                pos_buf.push(p);
                slot_buf.push(s.slot);
                scale_buf.push(if p == 0 { 0.0 } else { 1.0 });
            }
        }
        pad_tail(&mut tok_buf, filled);
        pad_tail(&mut pos_buf, filled);
        pad_tail(&mut slot_buf, filled);
        pad_tail(&mut scale_buf, filled);
        let all_dev = spans
            .iter()
            .all(|s| matches!(s.hidden, TapsIn::Device { .. }));
        let mut d2d = Vec::new();
        let hid_tensor = if all_dev {
            let mut off = 0usize;
            for (idx, s) in spans.iter().enumerate() {
                let TapsIn::Device { addr, rows, stride } = s.hidden else {
                    unreachable!();
                };
                if rows != s.tokens.len() {
                    return Err(Error::Rows);
                }
                let is_last_span = idx + 1 == spans.len();
                if stride > self.hidden {
                    for r in 0..rows {
                        d2d.push(D2dCopy {
                            input: 3,
                            src: addr + (r * stride * 4) as u64,
                            dst_row: off + r,
                            rows: 1,
                            pad_to: if is_last_span && r + 1 == rows {
                                filled
                            } else {
                                0
                            },
                            width: self.hidden,
                        });
                    }
                } else {
                    d2d.push(D2dCopy {
                        input: 3,
                        src: addr,
                        dst_row: off,
                        rows,
                        pad_to: if is_last_span { filled } else { 0 },
                        width: self.hidden,
                    });
                }
                off += rows;
            }
            None
        } else {
            let mut hid_buf = Vec::with_capacity(filled * self.hidden);
            for s in spans {
                let TapsIn::Host(h) = s.hidden else {
                    return Err(Error::Rows);
                };
                if h.len() != s.tokens.len() * self.hidden {
                    return Err(Error::Rows);
                }
                hid_buf.extend_from_slice(h);
            }
            let last_start = (m - 1) * self.hidden;
            let last_hid = hid_buf[last_start..last_start + self.hidden].to_vec();
            for _ in m..filled {
                hid_buf.extend_from_slice(&last_hid);
            }
            Some(Tensor::f32(vec![filled, self.hidden], hid_buf))
        };
        let feeds = [
            Some(Tensor::u32(vec![filled], tok_buf)),
            Some(Tensor::u32(vec![filled], pos_buf)),
            Some(Tensor::u32(vec![filled], slot_buf)),
            hid_tensor,
            Some(Tensor::f32(vec![filled, 1], scale_buf)),
        ];
        self.exec_mtp([filled, max_end, spans.len()], feeds, &d2d, wants, wait)
    }

    /// Runs one chained MTP step from device buffers.
    pub(crate) fn run_chain_batch(
        &mut self,
        slots: &[u32],
        positions: &[u32],
        tok_addrs: &[u64],
        hid_addrs: &[u64],
        wants: Wants,
        wait: bool,
    ) -> Result<Vec<Slot>, Error> {
        let m = slots.len();
        if m == 0 || positions.len() != m || tok_addrs.len() != m || hid_addrs.len() != m {
            return Err(Error::Rows);
        }
        let mut max_end = 0usize;
        for (&slot, &pos) in slots.iter().zip(positions) {
            if slot as usize >= self.core.slots || pos as usize + 1 > self.core.context {
                return Err(Error::RowPlace);
            }
            max_end = max_end.max(pos as usize + 1);
        }
        let filled = crate::row_bucket(m).min(self.core.context);
        let mut pos_buf = Vec::with_capacity(filled);
        let mut slot_buf = Vec::with_capacity(filled);
        let mut scale_buf = Vec::with_capacity(filled);
        let mut d2d = Vec::with_capacity(m * 2);
        for i in 0..m {
            let p = positions[i];
            pos_buf.push(p);
            slot_buf.push(slots[i]);
            scale_buf.push(if p == 0 { 0.0f32 } else { 1.0f32 });
            let pad = if i + 1 == m { filled } else { 0 };
            d2d.push(D2dCopy {
                input: 3,
                src: hid_addrs[i],
                dst_row: i,
                rows: 1,
                pad_to: pad,
                width: self.hidden,
            });
            d2d.push(D2dCopy {
                input: 0,
                src: tok_addrs[i],
                dst_row: i,
                rows: 1,
                pad_to: pad,
                width: 1,
            });
        }
        pad_tail(&mut pos_buf, filled);
        pad_tail(&mut slot_buf, filled);
        pad_tail(&mut scale_buf, filled);
        let feeds = [
            None,
            Some(Tensor::u32(vec![filled], pos_buf)),
            Some(Tensor::u32(vec![filled], slot_buf)),
            None,
            Some(Tensor::f32(vec![filled, 1], scale_buf)),
        ];
        self.exec_mtp([filled, max_end, m], feeds, &d2d, wants, wait)
    }

    pub(crate) fn ids(&self, slot: Slot) -> Result<Vec<u32>, Error> {
        Ok(self.core.runner.ids(slot)?)
    }

    pub(crate) fn read_floats(&self, slot: Slot) -> Result<Vec<f32>, Error> {
        let raw = self.core.runner.read(slot)?;
        Ok(raw.floats().map_err(tessera_run::Error::from)?.to_vec())
    }

    pub(crate) fn address(&self, slot: Slot) -> Option<u64> {
        self.core.runner.address(slot)
    }
}

/// Feeds deferred `(tokens, taps)` chunks into `drafter` for `slot`.
pub(crate) fn flush_mtp(
    drafter: &mut MtpDrafter,
    slot: u32,
    mut pos0: u32,
    tokens: &[u32],
    taps: &[f32],
    chunk: usize,
) -> Result<Option<u32>, Error> {
    if tokens.is_empty() {
        return Ok(None);
    }
    let width = drafter.hidden();
    let wants = drafter.wants();
    let mut off = 0usize;
    let mut last_tok = None;
    while off < tokens.len() {
        let len = (tokens.len() - off).min(chunk);
        let h = &taps[off * width..(off + len) * width];
        let outs = drafter.run_taps(
            slot,
            pos0,
            &tokens[off..off + len],
            TapsIn::Host(h),
            wants,
            true,
        )?;
        if off + len == tokens.len() {
            let ids = drafter.ids(outs[0])?;
            last_tok = Some(ids[len - 1]);
        }
        pos0 += len as u32;
        off += len;
    }
    Ok(last_tok)
}

enum ChainHid {
    Dev(Vec<u64>),
    Host(Vec<Vec<f32>>),
}

impl ChainHid {
    fn read(
        drafter: &MtpDrafter,
        slot: Slot,
        rows: &[usize],
        hidden: usize,
    ) -> Result<Self, Error> {
        if slot.form.name() == "f32"
            && let Some(base) = drafter.address(slot)
        {
            return Ok(Self::Dev(
                rows.iter()
                    .map(|&r| base + (r * hidden * 4) as u64)
                    .collect(),
            ));
        }
        let floats = drafter.read_floats(slot)?;
        Ok(Self::Host(
            rows.iter()
                .map(|&r| floats[r * hidden..(r + 1) * hidden].to_vec())
                .collect(),
        ))
    }

    fn at(&self, i: usize) -> TapsIn<'_> {
        match self {
            Self::Dev(addrs) => TapsIn::Device {
                addr: addrs[i],
                rows: 1,
                stride: 0,
            },
            Self::Host(rows) => TapsIn::Host(&rows[i]),
        }
    }
}

/// Runs batched first-form MTP plus shared chain steps for `jobs`.
pub(crate) fn draft_mtp_batch(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    jobs: &[MtpJob<'_>],
    taps: &TapsBuf,
    dual_stride: Option<(usize, usize)>,
    chain_n: usize,
) -> Result<Vec<Vec<u32>>, Error> {
    let n = jobs.len();
    if n == 0 || chain_n == 0 {
        return Ok(vec![Vec::new(); n]);
    }
    let hidden = drafter.hidden();
    let wants = drafter.wants();
    let (col_off, stride) = dual_stride.unwrap_or((0, hidden));
    let mut end_rows = Vec::with_capacity(n);
    let mut total_rows = 0usize;
    for j in jobs {
        total_rows += j.tokens.len();
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
                take_cols(
                    &v[j.tap_row_off * stride..],
                    j.tokens.len(),
                    stride,
                    col_off,
                    hidden,
                )
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
                    rows: j.tokens.len(),
                    stride: if col_off > 0 { stride } else { 0 },
                },
                TapsBuf::Host(_) if col_off > 0 => TapsIn::Host(&host_slices[i]),
                TapsBuf::Host(v) => {
                    let lo = j.tap_row_off * hidden;
                    let hi = lo + j.tokens.len() * hidden;
                    TapsIn::Host(&v[lo..hi])
                }
            };
            MtpSpan {
                slot: j.slot,
                start_pos: j.old_start,
                tokens: j.tokens,
                hidden: h,
            }
        })
        .collect();
    let first_out = drafter.run_batch(&spans, wants, !can_gpu_chain)?;
    if can_gpu_chain
        && let Some((tap_slot, tap_addr)) = dev_taps
        && let (Some(o0), Some(o1)) = (drafter.address(first_out[0]), drafter.address(first_out[1]))
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
                target.runner.copy_from(tap_slot, off, tok_addrs[i], 4)?;
                saved_addrs.push(tap_addr + off as u64);
                positions.push((j.new_start - 1 + step) as u32);
            }
            let wait = step + 1 == chain_n;
            let outs = drafter.run_chain_batch(
                &slots,
                &positions,
                &saved_addrs,
                &hid_addrs,
                wants,
                wait,
            )?;
            last_out = outs[0];
            if !wait {
                let next0 = drafter
                    .address(outs[0])
                    .ok_or(tessera_run::Error::Unaddressed(0))?;
                let next1 = drafter
                    .address(outs[1])
                    .ok_or(tessera_run::Error::Unaddressed(0))?;
                for i in 0..n {
                    tok_addrs[i] = next0 + (i * 4) as u64;
                    hid_addrs[i] = next1 + (i * hidden * 4) as u64;
                }
            }
        }
        let saved = target
            .runner
            .held()
            .read_u32s(tap_slot, (chain_n - 1) * n)?;
        let final_ids = drafter.ids(last_out)?;
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
    let ids = drafter.ids(first_out[0])?;
    let mut out: Vec<Vec<u32>> = end_rows.iter().map(|&r| vec![ids[r]]).collect();
    if chain_n == 1 {
        return Ok(out);
    }
    let seq_rows: Vec<usize> = (0..n).collect();
    let mut hid = ChainHid::read(drafter, first_out[1], &end_rows, hidden)?;
    for step in 1..chain_n {
        let toks: Vec<[u32; 1]> = out.iter().map(|v| [*v.last().unwrap()]).collect();
        let step_spans: Vec<MtpSpan<'_>> = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| MtpSpan {
                slot: j.slot,
                start_pos: (j.new_start - 1 + step) as u32,
                tokens: &toks[i],
                hidden: hid.at(i),
            })
            .collect();
        let outs = drafter.run_batch(&step_spans, wants, true)?;
        let step_ids = drafter.ids(outs[0])?;
        for i in 0..n {
            out[i].push(step_ids[i]);
        }
        if step + 1 < chain_n {
            hid = ChainHid::read(drafter, outs[1], &seq_rows, hidden)?;
        }
    }
    Ok(out)
}

/// Runs an MTP first-form pass plus adaptive chain steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draft_mtp(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    spec: &SpecLen,
    slot: u32,
    start_pos: u32,
    tokens: &[u32],
    taps: &TapsBuf,
    start: usize,
    avail: usize,
) -> Result<Vec<u32>, Error> {
    let chain_n = if !spec.is_adaptive() {
        avail
    } else {
        let mut n = 1;
        while n < avail && spec.more(None, n) {
            n += 1;
        }
        n
    };
    let job = MtpJob {
        slot,
        old_start: start_pos,
        new_start: start,
        tokens,
        tap_row_off: 0,
    };
    let mut drafts = draft_mtp_batch(target, drafter, &[job], taps, None, chain_n)?
        .pop()
        .unwrap_or_default();
    let k = spec.choose(None, drafts.len(), drafts.len());
    drafts.truncate(k);
    Ok(drafts)
}

/// Native MTP speculative decoding over a target and MTP pair.
pub struct MtpSpeculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut MtpDrafter,
    depth: usize,
    spec_len: SpecLen,
    prefill_chunk: Option<usize>,
}

impl<'a> MtpSpeculate<'a> {
    /// Pairs a tapped target engine with its MTP drafter.
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
        let spec_len = SpecLen::fixed(depth)
            .with_cost(CostModel::mtp())
            .with_mtp_seeds();
        Ok(Self {
            target,
            drafter,
            depth,
            spec_len,
            prefill_chunk: None,
        })
    }

    /// Configures adaptive or truncated draft length selection.
    pub fn with_spec_len(mut self, spec_len: SpecLen) -> Self {
        self.spec_len = spec_len;
        self
    }
}

spec_entry!(
    MtpSpeculate<'a>,
    SpecLen,
    spec_len,
    |s, target, req, spec, emit| run_mtp_req(target, s.drafter, s.depth, req, spec, emit)
);

fn run_mtp_req(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    depth: usize,
    req: SpecReq,
    spec: SpecLen,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Result<SpecStats, Error> {
    let hidden = drafter.hidden();
    let max_ctx = target.context().min(drafter.context());
    run_speculate(
        target,
        drafter,
        req,
        emit,
        |_, drafter, pos, prompt, len, taps| {
            if prompt.len() <= max_ctx {
                let shifted = &prompt[pos + 1..pos + len + 1];
                let wants = drafter.wants();
                drafter.run_taps(0, pos as u32, shifted, taps.as_in(), wants, true)?;
            }
            Ok(())
        },
        |_, _, pos, prompt, anchor, taps| {
            let pending: Vec<u32> = prompt[pos + 1..].iter().copied().chain([anchor]).collect();
            Ok((pos as u32, pending, taps, spec))
        },
        |target,
         drafter,
         (pending_start, pending_tokens, pending_taps, spec),
         start,
         _,
         remaining| {
            let avail = depth
                .min(spec.max_k())
                .min(remaining.saturating_sub(1))
                .min(max_ctx.saturating_sub(start + 1));
            if avail == 0 {
                return Ok(None);
            }
            let drafts = draft_mtp(
                target,
                drafter,
                spec,
                0,
                *pending_start,
                pending_tokens,
                pending_taps,
                start,
                avail,
            )?;
            Ok(Some(drafts))
        },
        |_,
         _,
         (pending_start, pending_tokens, pending_taps, spec),
         start,
         drafted,
         accepted,
         posterior,
         mut taps,
         stats| {
            spec.observe(drafted, accepted, None);
            stats.mtp_steps += 1;
            let commit = 1 + accepted;
            *pending_start = start as u32;
            *pending_tokens = posterior[..commit].to_vec();
            taps.keep_rows(commit, hidden);
            *pending_taps = taps;
            Ok(())
        },
    )
}
