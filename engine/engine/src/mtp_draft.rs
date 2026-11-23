//! Native MTP speculative decoding: one-layer MLA drafter and loop.

use crate::ranks::open_rank;
use crate::spec_len::{CostModel, SpecLen};
use crate::spec_loop::{D2dCopy, TapsBuf, TapsIn, run_speculate, spec_entry};
use crate::speculate::{DraftCore, draft_methods};
use crate::{Engine, Error, Event, Place, Request, SpecStats};
use tessera_engine_comm::Transport;
use tessera_graph::DimName::{Context, Hidden, Rows, Segments, Seqs};
use tessera_graph::{Graph, Role};
use tessera_interp::Tensor;
use tessera_lower::{Recipe, Settings, Slot, Wants, Weight};
use tessera_models::mtp;
use tessera_run::{Run, open};

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
                for w in sharded.whole.iter().filter(|_| place.rank == 0) {
                    let path = &graph.weights[w.weight as usize].path;
                    eprintln!("tessera: mtp {path:?} stays whole: {}", w.why);
                }
                sharded.graph
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

    pub(crate) fn read_last(
        &mut self,
        outputs: &[Slot],
        rows: usize,
        keep_hidden: bool,
    ) -> Result<(u32, TapsBuf), Error> {
        let ids = self.core.runner.ids(outputs[0])?;
        let token = ids[rows - 1];
        if !keep_hidden {
            return Ok((token, TapsBuf::Host(Vec::new())));
        }
        let slot = outputs[1];
        if slot.form.name() == "f32"
            && let Some(base) = self.core.runner.address(slot)
        {
            let addr = base + ((rows - 1) * self.hidden * 4) as u64;
            return Ok((
                token,
                TapsBuf::Device {
                    slot,
                    addr,
                    rows: 1,
                },
            ));
        }
        let raw = self.core.runner.read(slot)?;
        let floats = raw.floats().map_err(tessera_run::Error::from)?;
        let from = (rows - 1) * self.hidden;
        Ok((
            token,
            TapsBuf::Host(floats[from..from + self.hidden].to_vec()),
        ))
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
        let m = tokens.len();
        if m == 0 {
            return Err(Error::Rows);
        }
        if slot as usize >= self.core.slots || start_pos as usize + m > self.core.context {
            return Err(Error::RowPlace);
        }
        let filled = m.next_power_of_two().min(self.core.context);
        let mut tok_buf = Vec::with_capacity(filled);
        let mut pos_buf = Vec::with_capacity(filled);
        let mut scale_buf = Vec::with_capacity(filled);
        for (i, &t) in tokens.iter().enumerate() {
            let p = start_pos + i as u32;
            tok_buf.push(t);
            pos_buf.push(p);
            scale_buf.push(if p == 0 { 0.0 } else { 1.0 });
        }
        let last_tok = *tokens.last().expect("non-empty");
        let last_pos = start_pos + (m - 1) as u32;
        let last_scale = if last_pos == 0 { 0.0 } else { 1.0 };
        for _ in m..filled {
            tok_buf.push(last_tok);
            pos_buf.push(last_pos);
            scale_buf.push(last_scale);
        }
        let (hid_tensor, d2d) = match hidden {
            TapsIn::Host(h) => {
                if h.len() != m * self.hidden {
                    return Err(Error::Rows);
                }
                let mut hid_buf = Vec::with_capacity(filled * self.hidden);
                hid_buf.extend_from_slice(h);
                let last_hid = &h[(m - 1) * self.hidden..];
                for _ in m..filled {
                    hid_buf.extend_from_slice(last_hid);
                }
                (Some(Tensor::f32(vec![filled, self.hidden], hid_buf)), None)
            }
            TapsIn::Device { addr, rows } => {
                if rows != m {
                    return Err(Error::Rows);
                }
                (
                    None,
                    Some(D2dCopy {
                        input: 3,
                        src: addr,
                        rows,
                        filled,
                        width: self.hidden,
                    }),
                )
            }
        };
        let feeds = [
            Some(Tensor::u32(vec![filled], tok_buf)),
            Some(Tensor::u32(vec![filled], pos_buf)),
            Some(Tensor::u32(vec![filled], vec![slot; filled])),
            hid_tensor,
            Some(Tensor::f32(vec![filled, 1], scale_buf)),
        ];
        let sizes = [
            (Seqs, self.core.slots as u64),
            (Rows, filled as u64),
            (
                Context,
                crate::context_bucket(start_pos as usize + m, self.core.context),
            ),
            (Segments, 1),
        ];
        self.core
            .exec_tap(&feeds, d2d.as_slice(), &sizes, wants, wait)
    }

    /// Runs one chained MTP step from device token and hidden.
    fn run_chain(
        &mut self,
        slot: u32,
        pos: u32,
        tok_addr: u64,
        hid_addr: u64,
        wants: Wants,
        wait: bool,
    ) -> Result<Vec<Slot>, Error> {
        if slot as usize >= self.core.slots || pos as usize + 1 > self.core.context {
            return Err(Error::RowPlace);
        }
        let scale = if pos == 0 { 0.0f32 } else { 1.0f32 };
        let d2d = [
            D2dCopy {
                input: 3,
                src: hid_addr,
                rows: 1,
                filled: 1,
                width: self.hidden,
            },
            D2dCopy {
                input: 0,
                src: tok_addr,
                rows: 1,
                filled: 1,
                width: 1,
            },
        ];
        let feeds = [
            None,
            Some(Tensor::u32(vec![1], vec![pos])),
            Some(Tensor::u32(vec![1], vec![slot])),
            None,
            Some(Tensor::f32(vec![1, 1], vec![scale])),
        ];
        let sizes = [
            (Seqs, self.core.slots as u64),
            (Rows, 1),
            (
                Context,
                crate::context_bucket(pos as usize + 1, self.core.context),
            ),
            (Segments, 1),
        ];
        self.core.exec_tap(&feeds, &d2d, &sizes, wants, wait)
    }
}

/// Runs one local MTP step, keeping output hidden when requested.
pub(crate) fn run_mtp(
    drafter: &mut MtpDrafter,
    start_pos: u32,
    tokens: &[u32],
    hid: TapsIn<'_>,
    keep: bool,
) -> Result<(u32, TapsBuf), Error> {
    let wants = drafter.wants();
    let outputs = drafter.run_taps(0, start_pos, tokens, hid, wants, true)?;
    drafter.read_last(&outputs, tokens.len(), keep)
}

/// Queues `chain_n` MTP steps on GPU with one trailing sync.
#[allow(clippy::too_many_arguments)]
fn chain_gpu(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    start_pos: u32,
    tokens: &[u32],
    tap_slot: Slot,
    tap_addr: u64,
    start: usize,
    chain_n: usize,
) -> Result<Vec<u32>, Error> {
    let wants = drafter.wants();
    let m = tokens.len();
    let hidden = drafter.hidden();
    let first_in = TapsIn::Device {
        addr: tap_addr,
        rows: m,
    };
    let outputs = drafter.run_taps(0, start_pos, tokens, first_in, wants, false)?;
    let addr_of = |r: &dyn Run, s: Slot| r.address(s).ok_or(tessera_run::Error::Unaddressed(0));
    let mut tok_addr = addr_of(&*drafter.core.runner, outputs[0])? + ((m - 1) * 4) as u64;
    let mut hid_addr = addr_of(&*drafter.core.runner, outputs[1])? + ((m - 1) * hidden * 4) as u64;
    let mut last_out = outputs[0];
    for j in 1..chain_n {
        let off = (j - 1) * 4;
        target.runner.copy_from(tap_slot, off, tok_addr, 4)?;
        let saved_tok = tap_addr + off as u64;
        let pos = (start - 1 + j) as u32;
        let wait = j + 1 == chain_n;
        let outs = drafter.run_chain(0, pos, saved_tok, hid_addr, wants, wait)?;
        last_out = outs[0];
        if !wait {
            tok_addr = addr_of(&*drafter.core.runner, outs[0])?;
            hid_addr = addr_of(&*drafter.core.runner, outs[1])?;
        }
    }
    let mut drafts = target.runner.held().read_u32s(tap_slot, chain_n - 1)?;
    drafts.push(drafter.core.runner.ids(last_out)?[0]);
    Ok(drafts)
}

/// Runs an MTP first-form pass plus adaptive chain steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draft_mtp(
    target: &mut Engine,
    drafter: &mut MtpDrafter,
    spec: &SpecLen,
    start_pos: u32,
    tokens: &[u32],
    taps: &TapsBuf,
    start: usize,
    avail: usize,
) -> Result<(Vec<u32>, f64), Error> {
    let chain_n = if !spec.is_adaptive() {
        avail
    } else {
        let mut n = 1;
        while n < avail && spec.more(None, n) {
            n += 1;
        }
        n
    };
    let mut drafts = if chain_n >= 2
        && let TapsBuf::Device { slot, addr, rows } = *taps
        && rows == tokens.len()
        && slot.bytes() >= chain_n * 4
    {
        chain_gpu(
            target, drafter, start_pos, tokens, slot, addr, start, chain_n,
        )?
    } else {
        let mut out = Vec::with_capacity(chain_n);
        let (first, mut prev_hid) = run_mtp(drafter, start_pos, tokens, taps.as_in(), chain_n > 1)?;
        out.push(first);
        for j in 1..chain_n {
            let pos = (start - 1 + j) as u32;
            let (tok, next_hid) = run_mtp(
                drafter,
                pos,
                &[out[j - 1]],
                prev_hid.as_in(),
                j + 1 < chain_n,
            )?;
            out.push(tok);
            prev_hid = next_hid;
        }
        out
    };
    let (k, pred) = spec.choose(None, drafts.len(), drafts.len());
    drafts.truncate(k);
    Ok((drafts, pred))
}

/// Native MTP speculative decoding over a target and MTP pair.
pub struct MtpSpeculate<'a> {
    target: &'a mut Engine,
    drafter: &'a mut MtpDrafter,
    depth: usize,
    spec_len: SpecLen,
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
        let Some(width) = target.tap_width() else {
            return Err(Error::Taps);
        };
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
    request: Request,
    spec: SpecLen,
    emit: &mut dyn FnMut(Event) -> bool,
) -> Result<SpecStats, Error> {
    let hidden = drafter.hidden();
    let max_ctx = target.context().min(drafter.context());
    run_speculate(
        target,
        request,
        emit,
        |_, prompt, anchor, taps| {
            let pending: Vec<u32> = prompt[1..].iter().copied().chain([anchor]).collect();
            Ok((0u32, pending, taps, spec, 1.0f64))
        },
        |target,
         (pending_start, pending_tokens, pending_taps, spec, last_pred),
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
            let (drafts, pred) = draft_mtp(
                target,
                drafter,
                spec,
                *pending_start,
                pending_tokens,
                pending_taps,
                start,
                avail,
            )?;
            *last_pred = pred;
            Ok(Some(drafts))
        },
        |_,
         (pending_start, pending_tokens, pending_taps, spec, last_pred),
         start,
         drafted,
         accepted,
         posterior,
         mut taps,
         stats| {
            spec.observe(drafted, accepted, None, *last_pred);
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
